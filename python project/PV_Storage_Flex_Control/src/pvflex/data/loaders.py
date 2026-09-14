# =============================================================================
# data/loaders.py  ——  数据读取模块（把原始数据文件读成能用的时间序列）
# -----------------------------------------------------------------------------
# 【这个文件解决什么问题】
#   整个工程的"进货口"。数据资料文件夹里放的是一堆"给人看的"文件
#   （PVSYST 导出的发电量文本、能耗平台导出的 Excel 式 CSV、电费账单 PDF），
#   格式杂乱、编码不一、还夹着说明文字。本文件负责把它们读成"给程序看的"
#   整齐时间序列，后面的预测和调度才有原料可用。
#
#   它对外提供这些函数（按重要性排序）：
#     load_pv_generation()           → 逐小时光伏发电量（8760 小时/年）
#     load_load_series()             → 逐日园区总用电量（每天一个数）
#     load_self_consumption_hourly() → 逐小时 源-荷-消纳 三列（8760 小时/年）★实测
#     load_real_load_shape()         → 24 小时归一化负荷形状（占比）
#     load_net_purchase_hourly()     → 逐小时净购电量（账单锚定还原）
#
# 【给零基础同学的背景小课堂】
#   - pandas 是 Python 里专门处理表格 / 时间序列的库。
#   - Series    = "带时间索引的一列数"，比如每天一个用电量。
#   - DataFrame = "多列表格"，比如各用电点 × 各天。
#   - 本文件把数据都整理成 Series 或 DataFrame，方便后面加减、按月汇总。
#
# 【单位约定（🔴 非常重要，本项目最容易搞错的地方）】
#   电量 → kWh（度）      功率 → kW        电价 → 元/kWh
#   碳因子 → kgCO2/kWh    占比/形状 → 无量纲（和为 1）
#   ⚠️ 凡是名字里带 shape 的，都是"比例"而不是"电量"——具体量纲见各函数说明。
#
# ⚠️ 【已知限制（务必如实告知，不要回避）】
#   1) 编码混杂：能耗平台 CSV 有 GBK 与 UTF-8 两批（见 _read_csv_any_encoding）
#   2) 高压总表缺失：2025-11~2026-04 能耗平台未导出"高压进线级总表"，
#      只剩末端照明/插座表，负荷被严重低估 → 故引入"可靠区间截断"
#      （见 load_load_series 与 config.LOAD_RELIABLE_START/END）
#   3) 负荷只有日粒度：能耗平台给的是"每天合计"，没有逐时电量，
#      逐时曲线是用"日总量 × 形状"摊出来的，小时级绝对量存在分辨率上限
# =============================================================================

import os
import re            # 正则表达式库：用来从杂乱文本里挑出“日期,数值”这样的行
import glob          # 用来按通配符批量找文件，比如找出所有 25.03.csv
import logging

import numpy as np
import pandas as pd

import pvflex.config as cfg

# 关闭 pandas 2.x 关于“自动降类型(downcasting)”的 FutureWarning，
# 让运行输出更干净（不影响计算结果）。
pd.set_option('future.no_silent_downcasting', True)

# 用 logging 记录“读到了什么、跳过了什么”，比满屏 print 更专业，也方便以后排查。
# level=INFO 表示普通信息都打印；你以后想安静一点可以改成 WARNING。
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 0) 通用小工具：多编码 CSV 读取
# -----------------------------------------------------------------------------
# ⚠️ 踩坑记录（2026-09-14）：能耗平台导出的 CSV 编码不统一——
#    早期文件（如 25.03.csv、20250804.csv）是 GBK/GB18030，
#    较新文件（25.04.csv 起）是 UTF-8 (带 BOM)。
#    原代码只按 'utf-8-sig' 打开，遇到 GBK 文件会抛
#    UnicodeDecodeError 导致整月数据被 try/except 静默跳过（丢数据且不报警）。
#    下面这个函数按“UTF-8-sig → GBK → GB18030 → latin1”顺序尝试，
#    保证两种编码都能读进来；并清洗掉“表头前多一列空索引”的行列错位。
_CSV_ENCODINGS = ('utf-8-sig', 'gbk', 'gb18030', 'latin1')


def _read_csv_any_encoding(path, **kwargs):
    """
    按多种编码依次尝试读取 CSV，返回 (DataFrame, 实际使用的编码)。

    💡 初学者小课堂：文本文件存到硬盘上是一串字节，需要“编码表”才能翻译成汉字。
       同一份数据，不同软件导出的编码可能不同（GBK 是中文 Windows 常用，
       UTF-8 是国际通用）。我们挨个试，哪个能读通用哪个，就不用管文件到底啥编码。

    读进来后做一步“错位修正”：
      部分文件（GBK 那批）首列没有列名，pandas 会造一个 'Unnamed: 0' 空列，
      导致真正的“名称”列被挤到第 2 列。这里检测到就把它扶正，
      保证所有文件的列结构一致，后续按列名取值不会错位。
    """
    last_err = None
    for enc in _CSV_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            # ---- 错位修正：首列为 'Unnamed: 0' 且第 2 列名为 '名称' 时，把它设为索引 ----
            cols = [str(c) for c in df.columns]
            if len(cols) >= 2 and cols[0].startswith('Unnamed') and cols[1].strip() == '名称':
                df = df.iloc[:, 1:]          # 丢掉那个空列
            return df, enc
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    # 全部编码都失败：抛最后一个错，由调用方决定是否跳过该文件
    raise last_err if last_err else RuntimeError(f'无法读取 CSV：{path}')


def _parse_pvsyst_time(text, year=2025):
    """
    解析 PVSYST / 消纳率表里的时间字符串，返回 pandas.Timestamp；失败返回 None。

    💡 初学者小课堂：这些表的时间长这样 '01/01/90 06:00'——
       格式是【日/月/年(两位) 时:分】，且年份 '90' 是 PVSYST 的占位年份（不是 1990 年）。
       我们统一换成固定年份（默认 2025）只为能画出“几月几日几时”，不影响统计。
       注意：是“日/月”而不是常见的“月/日”，别搞反。

    另外兼容无时间的纯日期形式（有些导出行只有 '01/01/90'）。
    """
    if not text:
        return None
    txt = str(text).strip()
    try:
        parts = txt.split()
        d = parts[0]
        hm = parts[1] if len(parts) > 1 else '00:00'
        dd, mm, _yy = d.split('/')
        hh, mi = (hm.split(':') + ['0'])[:2]
        return pd.Timestamp(year=year, month=int(mm), day=int(dd),
                            hour=int(hh), minute=int(mi))
    except Exception:
        return None


# -----------------------------------------------------------------------------
# 1) 读取 PVSYST 年发电量 CSV（每小时发电量）
# -----------------------------------------------------------------------------
def load_pv_generation(path=None):
    """
    读取 PVSYST 导出的全年发电量文件，返回逐小时光伏发电量。

    这类文件的特点（为什么不能直接用 pd.read_csv）：
      - 前面好几行是软件说明（不是数据），例如 'PVSYST 7.4.0'、'Simulation date'；
      - 真正的数据行长这样：  01/01/90 06:00,62.884
        意思是 "1月1日 06:00，发电量 62.884 kWh"；
      - 时间格式是【日/月/年(两位) 时:分】，注意是"日/月"而不是常见的"月/日"；
      - 年份 '90' 是 PVSYST 的占位年份（不是 1990 年），我们忽略它。
      因为格式不规则，这里用正则逐行挑出数据行，而不是当成标准 CSV 读。

    参数
    ----
    path : str, 可选
        发电量文件路径；不传则用 config.PV_GEN_CSV。

    返回
    ----
    pandas.Series
        索引 = 逐小时时间戳（DatetimeIndex，年份统一为 2025）
        值   = 该小时发电量，单位 kWh
        名字 = 'pv_generation'
        读不到文件时返回【空 Series】（不抛异常，避免整个流程崩掉）。

    ⚠️ 时区/年份说明：年份被统一改写成 2025，只为让"几月几日几时"能画出来，
       不影响任何统计量（年发电量、月度分布都不依赖具体年份）。
    """
    path = path or cfg.PV_GEN_CSV

    # 文件不存在时，友好提示并返回空序列，而不是抛一堆看不懂的错误
    if not os.path.exists(path):
        logger.warning('未找到光伏发电量文件：%s，请检查 config.REPO_DATA_DIR', path)
        return pd.Series(dtype=float, name='pv_generation')

    # 正则：匹配 “两位数字/两位数字/两位数字 两位数字:两位数字,任意数值”
    # 拆开看：(\d{2})/(\d{2})/\d{2}\s+(\d{2}):(\d{2}),(.+)
    #        日      月      年(忽略)  时    分      数值
    pattern = re.compile(r'^(\d{2})/(\d{2})/\d{2}\s+(\d{2}):(\d{2}),(.+)$')

    times = []   # 用来存每一行对应的时间
    values = []  # 用来存每一行对应的发电量

    # encoding='utf-8-sig' 能正确处理文件开头的 BOM 标记；
    # errors='ignore' 让偶尔的乱码字符不至于让程序崩溃。
    with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            line = line.strip()
            m = pattern.match(line)
            if not m:
                continue  # 不是数据行（是说明文字），跳过
            day, month, hour, minute, val = m.groups()
            # 用固定年份 2025 只是为了能画出“几月几日”，真实年份不影响统计
            try:
                dt = pd.Timestamp(year=2025, month=int(month), day=int(day),
                                  hour=int(hour), minute=int(minute))
            except ValueError:
                continue  # 个别非法日期就跳过，不中断整个程序
            times.append(dt)
            try:
                values.append(float(val))
            except ValueError:
                values.append(np.nan)

    # 组装成 Series（一列带时间索引的数据）
    s = pd.Series(values, index=times, name='pv_generation')
    s = s.sort_index()              # 按时间排好序
    s = s.clip(lower=0)             # 夜间有 -0.0854 这种数值噪声，按 0 处理（夜里不发电）
    logger.info('读取光伏发电量：%d 个小时（%s ~ %s）',
                len(s), s.index.min(), s.index.max())
    return s


# -----------------------------------------------------------------------------
# 2) 读取能耗平台“月”CSV（宽表），并聚合成“每天总负荷”
# -----------------------------------------------------------------------------
def _parse_one_monthly_file(path):
    """
    解析单个能耗平台月文件（例如 25.03.csv），返回该月【逐日】园区总用电量。

    这种文件的样子（宽表 —— 注意"行"和"列"的方向与直觉相反）：
        名称,2025-03-01,2025-03-02,...,2025-03-31,合计
        门卫一,/,/,...,/
        JGAL2,12.3,15.6,...,/
    也就是：每一【行】是一个用电点，每一【列】是一天，格子是当天用电量，
    没数据的地方用 "/" 表示。

    我们要的是"园区每天的总用电量"，所以处理思路是：
      设"名称"为行索引 → 转置(行列互换，变成 日期×用电点) → 把各用电点横向加总。

    参数
    ----
    path : str
        月文件路径（文件名形如 '25.03.csv'）。

    返回
    ----
    pandas.Series
        索引 = 该月的日期，值 = 当天全园区各计量点电量之和（kWh）。

    💡 初学者小课堂：函数名以 _ 开头表示"内部辅助函数"——
       它不是给外部调用的，只是本文件内部的分工，你不需要在别处 import 它。
    """
    # ⚠️ 编码兼容（2026-09-14 修）：能耗平台 CSV 有 GBK 与 UTF-8 两批，
    #    统一走 _read_csv_any_encoding，避免 GBK 文件被静默跳过丢整月数据。
    df, enc = _read_csv_any_encoding(path)
    logger.debug('读取 %s（编码 %s，%d 行 × %d 列）',
                 os.path.basename(path), enc, df.shape[0], df.shape[1])

    # 第一列叫“名称”，把它设为行索引（即每一行代表一个用电点）
    if '名称' in df.columns:
        df = df.set_index('名称')

    # 有的文件末尾有一行“合计”（各用电点每天的合计），我们要自己加总，
    # 所以如果有这行就先删掉，避免重复计算。
    if '合计' in df.index:
        df = df.drop(index='合计')

    # 有的文件还有“合计”列（每个用电点各月的合计），删掉它，只留日期列
    if '合计' in df.columns:
        df = df.drop(columns='合计')

    # 转置：原来是 用电点×日期，转置后变成 日期×用电点
    df = df.T

    # “/” 不是数字，pandas 不会自动当成缺失值，要手动替换成 NaN（空值）
    # 加 infer_objects() 是为了消除 pandas 2.x 关于“自动降类型”的 FutureWarning
    df = df.replace('/', np.nan).infer_objects(copy=False)
    # 再把剩下的文本尽量转成数字；转不成的变成 NaN（errors='coerce'）
    df = df.apply(pd.to_numeric, errors='coerce')

    # 按“天”把各用电点的用电量加总（axis=1 表示沿“列”方向，即横向相加）
    # skipna=True：某个用电点当天没数据(/)就跳过它，不影响其他点相加
    daily = df.sum(axis=1, skipna=True)

    # 把字符串日期（如 '2025-03-01'）变成真正的时间类型，方便后续按月统计
    daily.index = pd.to_datetime(daily.index, errors='coerce')
    # 丢掉日期解析失败的行：⚠️ Series.dropna() 只删 NaN 值、不删 NaT 索引，
    # 必须显式按 notna() 过滤索引，否则"备注"类非日期列会带 NaT 索引混进来。
    daily = daily[daily.index.notna()]
    return daily.dropna()


def load_load_series(folder=None):
    """
    读取“能耗管理平台数据”文件夹里所有月文件，拼成一个连续的“每天总负荷”序列。

    我们只挑名字像 '25.03.csv'、'26.01.csv' 这样的“月文件”
    （用通配符 '??.??.csv' 匹配：两个字符.两个字符.csv）。
    像 '26.01.01.csv'（日文件）、'25年.csv'（年文件）就不在这里处理。

    参数
    ----
    folder : str, 可选
        数据文件夹路径；不传则用 config.LOAD_DIR。

    返回
    ----
    pandas.Series
        索引 = 日期（DatetimeIndex，精确到"天"）
        值   = 园区当天总用电量，单位 kWh
        什么都没读到 / 文件夹不存在时返回【空 Series】并打 WARNING。

    ⚠️ 本函数做了三层清洗（每层都会打 WARNING 日志，务必留意）：
       ① 剔除"缺失日"：日用电 ≤ 0 视为未采集，不是真的不用电；
       ② 【可靠区间截断】：只保留 config.LOAD_RELIABLE_START ~ END 之间的日期
          —— 区间外月份能耗平台缺高压总表，纳入会污染统计（详见下方代码注释）；
       ③ 剔除量级异常的月 / 日（中位数 ± MAD 稳健判据）。
       所以返回的天数通常【少于】文件里的实际天数，这是刻意为之。
    """
    folder = folder or cfg.LOAD_DIR

    if not os.path.isdir(folder):
        logger.warning('未找到负荷数据文件夹：%s，请检查 config.REPO_DATA_DIR', folder)
        return pd.Series(dtype=float)

    # glob 按通配符找文件；sorted 让结果按文件名排序（1月、2月……顺序拼）
    files = sorted(glob.glob(os.path.join(folder, '??.??.csv')))

    pieces = []
    for fp in files:
        try:
            d = _parse_one_monthly_file(fp)
            pieces.append(d)
            logger.info('读取负荷文件 %s（%d 天）', os.path.basename(fp), len(d))
        except Exception as e:
            # 某个文件格式不对就跳过，并提示，不中断整体
            logger.warning('跳过文件 %s，原因：%s', os.path.basename(fp), e)

    if not pieces:
        logger.warning('没有读到任何负荷数据，请检查 LOAD_DIR 路径。')
        return pd.Series(dtype=float)

    # 把各月拼成一条长序列，再按日期排序
    s = pd.concat(pieces).sort_index()

    # ---- 缺失日清洗：日用电 ≤ 0 是能耗平台“未采集/抄表缺失”，不是真实 0 用电 ----
    # 这些 0 值若不剔除，会拉低年/月用电量统计，并污染“代表日”选取（落在缺失日
    # 时 pv>0 而 load≈0，优化器会算出负的电费/反升的碳排，纯属数据缺口假象）。
    n_missing = int((s <= 0).sum())
    if n_missing > 0:
        logger.warning('剔除负荷缺失日 %d 个（日用电≤0，视为未采集/抄表缺失）', n_missing)
        s = s[s > 0]

    # ---- 【2026-09-14 修正】可靠区间截断 --------------------------------------
    # ⚠️ 这是本版最重要的数据口径修正，务必理解：
    #   旧版代码用“某月中位数 < 全局中位数 × 25%”判定“疑似换表/改单位”，把
    #   2025-12~2026-03 整月剔除。经逐月核对【计量点构成】后确认该判定【是误判】：
    #     正常月(如 25.09) 含「配电房-低压柜(47.5万) / 1AA1-进线(19.8万) / 2AA1-进线」
    #       等【高压柜级总表】，共 175+ 个有效计量点，月合计 145 万 kWh；
    #     25.12~26.03 这些【高压总表全部缺失】，只剩末端照明/插座表（107/105/103/104 个点），
    #       月合计骤降到 4~6 万 kWh；
    #     26.04 高压表【恢复】，月合计回到 27 万 kWh。
    #   即：这几个月不是“用电量崩了”，而是【数据源没导出高压侧总表】——
    #       用它算出来的“负荷”只覆盖末端小负荷，会造成年用电量严重低估、
    #       光伏消纳率虚高、储能优化结论失真。
    #   故改为【按可靠区间显式截断】，并明确告知用户原因，而不是靠统计阈值猜。
    if getattr(cfg, 'LOAD_DROP_UNRELIABLE', True) and len(s) > 0:
        t0 = pd.Timestamp(cfg.LOAD_RELIABLE_START)
        t1 = pd.Timestamp(cfg.LOAD_RELIABLE_END) + pd.Timedelta(days=1)  # 含末日
        outside = s[(s.index < t0) | (s.index >= t1)]
        if len(outside) > 0:
            bad_m = sorted({str(p) for p in outside.index.to_period('M').unique()})
            logger.warning(
                '按可靠区间截断：丢弃 %d 天（区间外月份：%s）——这些月份能耗平台缺失'
                '高压进线级总表，只剩末端表，负荷被严重低估，纳入会污染年用电量与消纳率。'
                '如需保留请在 config 里改 LOAD_RELIABLE_START/END 或设 LOAD_DROP_UNRELIABLE=False',
                len(outside), ', '.join(bad_m))
            s = s[(s.index >= t0) & (s.index < t1)]

    # ---- 稳健尺度基准：各月日负荷中位数之中位数（对少数异常月不敏感）----
    # 用它作参照，比“全局中位数/全局 MAD”更稳：异常月会把全局 MAD 撑大，
    # 导致低位下界被拉到 0、漏掉所有低位脏数据（这正是上一版没抓到故障的原因）。
    if len(s) > 10:
        monthly_med = s.resample('ME').median()
        typical = float(np.nanmedian(monthly_med.values)) if monthly_med.notna().any() else 0.0

        # ---- 月级尺度异常兜底 ----
        # 可靠区间截断后，正常情况下不应再有“量级骤降”的月份。此处保留一层保守
        # 兜底（阈值放宽到 10%），只在本区间内仍出现明显异常月时告警+剔除，
        # 不再是原来 25% 那种会误伤“部分缺失”月份的激进阈值。
        if typical > 0:
            bad_months = monthly_med[monthly_med < 0.10 * typical]
            if len(bad_months) > 0:
                bad_periods = set(bad_months.index.to_period('M'))
                for ym, v in bad_months.items():
                    logger.warning(f'剔除异常月 {ym.strftime("%Y-%m")}（日负荷中位 {v:,.0f} '
                                   f'≈ 正常的 {v/typical*100:.1f}%，量级异常，整月排除）')
                # 注意：bad_months.index 是“月末标签”，而 s 是日级索引，
                #       不能用 s.drop(bad_months.index)（那只删月末那一天）。
                #       改用“月份周期”匹配，整月剔除。
                s = s[~s.index.to_period('M').isin(bad_periods)]

        # ---- 日级清洗 ----
        med = s.median()
        mad = (s - med).abs().median()
        if mad and mad > 0:
            # 高位：中位数 + 20×MAD（只在明显错乱时剔除，避免误删真实高峰）
            upper = med + 20 * mad
            bad_high = s[s > upper]
            if len(bad_high) > 0:
                logger.warning('清洗负荷高位异常值 %d 个（超出中位数+20×MAD，示例：%s）',
                               len(bad_high),
                               ', '.join(f'{d.date()}={v:.0f}' for d, v in bad_high.items()[:3]))
                s = s.drop(bad_high.index)
        if typical > 0:
            # 低位：绝对地板 typical × LOAD_DAY_MIN_FRAC（异常月已排除，typical 不受污染）。
            # 专门抓“抄表缺失/部分计量”的极小日（如 2025-06 个别日仅数百 kWh）。
            floor = cfg.LOAD_DAY_MIN_FRAC * typical
            bad_low = s[s < floor]
            if len(bad_low) > 0:
                logger.warning('清洗负荷低位异常值 %d 个（< %.0f kWh≈正常的%.0f%%，'
                               '疑似抄表缺失/部分计量）',
                               len(bad_low), floor, cfg.LOAD_DAY_MIN_FRAC * 100)
                s = s.drop(bad_low.index)
    return s


# -----------------------------------------------------------------------------
# 3) 读取电费“每小时电量”表，还原为“园区净购电”小时曲线（账单锚定法）
# -----------------------------------------------------------------------------
def _bill_monthly_net_purchase(folder=None):
    """
    从“电费清单/总账单”下所有 pdf 提取每期“本期电量”，按 'YYYY-MM' 汇总，
    得到园区每月“净购电量”（kWh）。净购电量 = 从电网实际买入的电量
    （不含光伏就地消纳部分，光伏就地消纳不进电表）。
    返回：dict {'YYYY-MM': kwh}

    💡 初学者小课堂：这个函数名字以 _ 开头，表示它是“内部辅助函数”。
       后面 cli.py 做交叉验证时也会用到它，所以我在模块底部专门加了一个
       不带下划线的 bill_monthly_net_purchase() 当“公开接口”，方便外部调用。
    """
    folder = folder or os.path.join(cfg.REPO_DATA_DIR, '电费清单', '总账单')
    files = sorted(glob.glob(os.path.join(folder, '**', '*.pdf'), recursive=True))
    if not files:
        logger.warning('未找到电费账单 PDF：%s', folder)
        return {}
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning('未安装 pypdf，无法读取账单做锚定')
        return {}
    bill = {}
    for fp in files:
        try:
            r = PdfReader(fp)
            txt = '\n'.join((p.extract_text() or '') for p in r.pages)
        except Exception as e:
            logger.warning('读取账单失败 %s：%s', os.path.basename(fp), e)
            continue
        m = re.search(r'本期电量\s*([\d,.]+)\s*千瓦时', txt)
        if not m:
            continue
        kwh = float(m.group(1).replace(',', ''))
        sub = os.path.basename(os.path.dirname(fp))   # 子目录形如 '2025'
        yr = re.search(r'(\d{4})', sub)
        mo = re.search(r'(\d+)月', os.path.basename(fp))
        if not (yr and mo):
            continue
        key = '%s-%02d' % (yr.group(1), int(mo.group(1)))
        bill[key] = bill.get(key, 0.0) + kwh
    return bill


def _hourly_file_year(basename, tag):
    """
    推断小时表文件对应的 'YYYY-MM'。

    💡 初学者小课堂：文件名里藏着年月信息，但写得不规范——
        有的是 '01_041820.xls'（前两位是月），有的是 '26年1月.xls'（年是汉字）。
        Python 没法自动猜，所以这里专门写规则把文件名翻译成标准的 '2025-01' 这种格式，
        方便后面和账单按"年-月"对上号。
        tag 参数（表计编号 '2' 或 '3'）目前不影响年份判定（2026 月由 '26年' 前缀
        识别，其余均按 2025 处理），保留参数是为兼容调用处签名。
    """
    if '26年' in basename:
        # 形如 '26年1月.xls'：月份在 '年' 与 '月' 之间
        mo = re.search(r'(\d+)月', basename)
        mm = int(mo.group(1)) if mo else 1
        return '2026-%02d' % mm
    mm = int(basename[:2])
    return '2025-%02d' % mm


def load_net_purchase_hourly(folder=None):
    """
    读取“电费清单/每小时电量/*.xls”，还原“园区净购电”每小时曲线。

    💡 初学者小课堂（为什么需要这个函数）：
      电费账单只告诉我们“这个月一共从电网买了多少电”，但没有“每小时买多少”。
      而小时电量表里每个小时都有数字，可惜它的“倍率没还原”（数值偏小好几倍），
      而且有些月份文件还缺几天。所以我们用“账单的月总量”当尺子，
      把小时表这一串数字整体放大/缩小，让它的月合计刚好等于账单的数字——
      这样既保住了“一天里哪小时高哪小时低”的真实形状，量级又对了。
      这个办法叫“锚定缩放”（用可靠的月总量，给不可靠的小时数据定标）。

    背景（重要口径说明）：
      - 这些 xls 里的 [2]/[3] 表计（户号（已隐去），已确认是园区总进线表）
        记录的是“正向有功总电量”，即园区从电网净购入的电量（不含光伏就地消纳）。
      - 其数值量级偏小且逐月递增，是“缺少 CT/PT 倍率 + 个别文件不完整(如9月仅7天)”
        所致，不能直接当真实负荷。
      - 本函数用“电费账单月净购电量”逐月做锚定缩放：
            scale(month) = bill_net(month) / sum(hourly_show(month))
        再用 scale 把该月每天 24h 示值乘回，得到“量级正确、形状保留”的净购电曲线。
      - 还原结果是“净购电”，不是“园区总用电”。园区总用电请仍用 load_load_series()
        （能耗平台，含光伏就地消纳）。两者差 = 光伏就地消纳量。

    返回：pandas.Series，索引为小时级时间，值为净购电量(kWh)。
          读不到或锚定失败返回空 Series。
    """
    folder = folder or os.path.join(cfg.REPO_DATA_DIR, '电费清单', '每小时电量')
    if not os.path.isdir(folder):
        logger.warning('未找到每小时电量文件夹：%s', folder)
        return pd.Series(dtype=float, name='net_purchase')

    bill = _bill_monthly_net_purchase()
    if not bill:
        logger.warning('账单锚定失败，无法还原净购电曲线')
        return pd.Series(dtype=float, name='net_purchase')

    hfiles = sorted(glob.glob(os.path.join(folder, '*.xls')))
    # 先按 (年-月) 收集每个文件的“天数”，同月只保留天数最多的文件，
    # 避免片段文件（如 '10_041809.xls' 16天）与完整月文件（'10月.xls' 31天）
    # 重复缩放、重叠小时互相抵消导致月总量失真。
    file_days = {}
    meta = {}
    for fp in hfiles:
        bn = os.path.basename(fp)
        df = pd.read_excel(fp, header=None)
        hr = None
        for i in range(8):
            if '测量点名称' in df.iloc[i].astype(str).tolist():
                hr = i
                break
        if hr is None:
            continue
        rows = df.iloc[hr + 1:]
        pts = rows.iloc[:, 3].astype(str).tolist()
        tag = '2' if any(p.strip() == '2' for p in pts) else '3'
        sub = rows[rows.iloc[:, 3].astype(str).str.strip() == tag]
        if len(sub) == 0:
            continue
        ym = _hourly_file_year(bn, tag)
        ndays = sum(1 for _, rr in sub.iterrows()
                    if pd.to_numeric(rr.iloc[8:32], errors='coerce').notna().sum() >= 20)
        meta[bn] = (fp, tag, ym, ndays)
        if ym not in file_days or ndays > file_days[ym][1]:
            file_days[ym] = (bn, ndays)
    keep = {v[0] for v in file_days.values()}           # 每个 ym 保留的文件名

    pieces = []
    skipped_no_bill = []                                 # 小时表在、但无账单可锚定的月份
    for fp in hfiles:
        bn = os.path.basename(fp)
        if bn not in keep:
            logger.info('同月去重：跳过 %s（保留天数更多的同月文件）', bn)
            continue
        fp, tag, ym, _ = meta[bn]
        if ym not in bill:
            logger.warning('跳过 %s（%s）：账单无对应月净购电量', bn, ym)
            skipped_no_bill.append(ym)
            continue
        df = pd.read_excel(fp, header=None)
        hr = None
        for i in range(8):
            if '测量点名称' in df.iloc[i].astype(str).tolist():
                hr = i
                break
        rows = df.iloc[hr + 1:]
        sub = rows[rows.iloc[:, 3].astype(str).str.strip() == tag]
        # 逐日 24h 示值
        day_arrays = []
        day_dates = []
        for _, rr in sub.iterrows():
            v = pd.to_numeric(rr.iloc[8:32], errors='coerce')
            if v.notna().sum() >= 20:
                day_arrays.append(v.to_numpy(float))
                dstr = rr.iloc[6] if len(rr) > 6 else None
                day_dates.append(dstr)
        if not day_arrays:
            continue
        arr = np.vstack(day_arrays)                       # (days, 24)
        month_show = arr.sum()
        if month_show <= 0:
            continue
        scale = bill[ym] / month_show                    # 锚定缩放因子
        arr_scaled = arr * scale                          # 量级还原

        idx = []
        for di, dstr in enumerate(day_dates):
            dt = pd.to_datetime(dstr, errors='coerce') if dstr is not None else pd.NaT
            if pd.isna(dt):
                dt = pd.Timestamp(ym + '-01') + pd.Timedelta(days=di)
            for h in range(24):
                idx.append(dt + pd.Timedelta(hours=h))
        s = pd.Series(arr_scaled.flatten(), index=idx, name='net_purchase')
        pieces.append(s)
        logger.info('还原净购电 %s（表计[%s]）：缩放因子=%.3f，天数=%d',
                    ym, tag, scale, len(day_arrays))

    if not pieces:
        return pd.Series(dtype=float, name='net_purchase')
    out = pd.concat(pieces).sort_index()
    # 极个别跨文件边界仍重叠的小时，取平均
    dup = out.index.duplicated(keep=False)
    if dup.any():
        avg = out[out.index.duplicated(keep=False)].groupby(level=0).mean()
        out = out[~out.index.duplicated(keep=False)]
        out = pd.concat([out, avg]).sort_index()
    out = out.clip(lower=0)
    logger.info('净购电小时曲线：%d 小时（%s ~ %s）', len(out), out.index.min(), out.index.max())
    if skipped_no_bill:
        uniq = sorted(set(skipped_no_bill))
        logger.warning('净购电曲线存在缺口：以下 %d 个月有小时表但无账单可锚定，已跳过不还原：%s',
                       len(uniq), ', '.join(uniq))
    return out


# -----------------------------------------------------------------------------
# 公开接口：把内部 _bill_monthly_net_purchase 暴露给外部（如 cli 做交叉验证）
# -----------------------------------------------------------------------------
def bill_monthly_net_purchase(folder=None):
    """
    公开版本的“账单月净购电量”查询（内部实现见 _bill_monthly_net_purchase）。
    供 cli.py 的 netload 模式做交叉验证时直接调用，避免在外部用下划线私有函数。
    """
    return _bill_monthly_net_purchase(folder=folder)


# -----------------------------------------------------------------------------
# 4) 【2026-09-14 新增】读取“真实逐时源-荷-消纳”全年表（8760 小时）
# -----------------------------------------------------------------------------
def load_self_consumption_hourly(path=None):
    """
    读取 `福州烟草-消纳率计算.xlsx` 的「福州烟草-发电量」sheet，
    还原【8760 小时】逐时序列。

    💡 初学者小课堂（为什么需要这个函数）：
      能耗平台给的负荷只有“每天总用电量”，而 24 小时调度需要逐时曲线。
      常规做法是用一条“形状”把日总量摊开——但形状若是拍脑袋定的，分时细节就是猜的。
      这份表是目前唯一“逐时 × 全年完整”的数据，含三列关键信息：
        · E_Grid        —— 逐时光伏发电量（与 PVSYST 年 CSV 同源）
        · 用电量        —— 逐时用电量
        · 被消纳的用电量 —— 光伏被园区就地用掉的部分
      本函数把它读出来，供 preprocess 提取【实测 24h 形状】，
      从而把“摊开用的形状”从估算升级为实测。

    ⚠️ 口径局限（必须知道，避免误用）：
      该表“用电量”全年合计约 333 万 kWh，只有能耗平台园区总量(≈931 万 kWh/年)的 ~36%。
      原因是它只覆盖【部分计量点】，不含高压进线级总表。
      所以：
        · 做“分时形状参考 / 消纳率分析”可以放心用；
        · 不能直接当“园区总负荷”去算年电费、储能容量。
      园区总量口径请继续用 load_load_series()。
      ⚠️ 另注意：本表自算的消纳率约 41.91%，与园区全量口径的 18.35% 不是同一个概念，
         报告中引用时必须注明口径，不可混用。

    参数
    ----
    path : str, 可选
        消纳率计算表路径；不传则用 config.SELF_CONSUMPTION_XLSX。

    返回
    ----
    pandas.DataFrame
        索引 = 逐小时时间戳（2025 全年，DatetimeIndex）
        列   = ['pv', 'load', 'self_consumed']，三列【单位均为 kWh】
               · pv            —— 该小时光伏发电量
               · load          —— 该小时用电量
               · self_consumed —— 该小时光伏被园区就地用掉的电量
        读不到 / 未装 openpyxl / sheet 名不匹配时返回【空 DataFrame】。

    ⚠️ 列的单位与口径：三列都直接取自原表，未做任何换算。
       其中 load 列【不是园区全量负荷】（原因见上方口径局限），
       用它算总量会严重偏低，只能用来提取"分时形状"。
    """
    path = path or cfg.SELF_CONSUMPTION_XLSX
    if not os.path.exists(path):
        logger.warning('未找到消纳率计算表：%s', path)
        return pd.DataFrame()

    try:
        import openpyxl
    except ImportError:
        logger.warning('未安装 openpyxl，无法读取消纳率计算表')
        return pd.DataFrame()

    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    # 找到“含逐时数据的 sheet”：优先按名找，找不到就取第一个有 8760 行数据的表
    sheet_name = None
    for cand in ('福州烟草-发电量',):
        if cand in wb.sheetnames:
            sheet_name = cand
            break
    if sheet_name is None:
        logger.warning('消纳率表中未找到“福州烟草-发电量”sheet，现有：%s', wb.sheetnames)
        wb.close()
        return pd.DataFrame()

    ws = wb[sheet_name]
    times, pv, load, used = [], [], [], []
    for r in ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=True):
        t = r[0]
        if t is None:
            continue
        ts = _parse_pvsyst_time(str(t).strip())
        if ts is None:
            continue
        times.append(ts)
        pv.append(pd.to_numeric(r[1], errors='coerce'))
        load.append(pd.to_numeric(r[3], errors='coerce'))
        used.append(pd.to_numeric(r[6], errors='coerce'))
    wb.close()

    if not times:
        logger.warning('消纳率表未解析出任何逐时数据行')
        return pd.DataFrame()

    df = pd.DataFrame({'pv': pv, 'load': load, 'self_consumed': used},
                      index=pd.DatetimeIndex(times)).sort_index()
    # 同一小时若重复出现（表里有合并单元格残留），取均值
    df = df.groupby(level=0).mean()
    df['pv'] = df['pv'].clip(lower=0)
    logger.info('读取真实逐时源-荷-消纳：%d 小时（%s ~ %s），用电量合计 %.0f kWh、'
                '发电量合计 %.0f kWh、被消纳合计 %.0f kWh',
                len(df), df.index.min(), df.index.max(),
                np.nansum(df['load']), np.nansum(df['pv']), np.nansum(df['self_consumed']))
    return df


def load_real_load_shape(path=None):
    """
    从真实逐时数据里算出【24 小时归一化负荷形状】，供"日总量 × 形状"摊分使用。

    💡 为什么要用中位数而不是均值：
       均值会被个别极端日拉偏——比如某天检修停产（用电骤降到 1/10），
       或某天抄表异常（数据翻倍），这些"坏日子"会把均值拽走。
       逐小时取【中位数】得到的形状更能代表"平常的一天"，
       这也是能源行业做典型日曲线时的常规做法。

    参数
    ----
    path : str, 可选
        消消纳率计算表路径；不传则用 config.SELF_CONSUMPTION_XLSX。

    返回
    ----
    numpy.ndarray（长度 24）
        【已归一化】的小时占比，shape.sum() == 1，shape[h] 表示第 h 小时
        占全天用电量的比例。
        样本不足（< 7 天）/ 数据异常时返回 None（由调用方退回其他来源）。

    ⚠️ 量纲提醒：本函数返回的是【占比】，不是电量。
       需要真实电量量纲请用 preprocess.TYPICAL_LOAD_SHAPE_RAW。
    """
    path = path or cfg.SELF_CONSUMPTION_XLSX
    df = load_self_consumption_hourly(path)
    if df.empty or 'load' not in df:
        return None
    s = df['load'].dropna()
    if len(s) < 24 * 7:
        logger.warning('真实逐时负荷样本不足（%d 小时），不使用其形状', len(s))
        return None
    by_hour = s.groupby(s.index.hour).median().reindex(range(24))
    if by_hour.isna().any() or by_hour.sum() <= 0:
        return None
    shape = by_hour.to_numpy(float)
    return shape / shape.sum()

