# =============================================================================
# data/preprocess.py  ——  数据预处理（为“预测”和“优化”准备整齐的输入）
# -----------------------------------------------------------------------------
# 原始数据有两个“分辨率”问题：
#   1) 发电量(PV)是“每小时”的（一年 8760 个点）；
#   2) 园区负荷(load)原始是“每天合计”的（一天一个数）。
# 但研究方向3要做“日前 24 小时”的源-荷预测和调控，需要“每小时”的源和荷
# 对齐到同一张表上。所以本文件负责：
#   - 把“每天总负荷”摊成“每小时负荷”（用一个典型日形状去分配）；
#   - 把 PV 和 负荷 按时间对齐成一张表；
#   - 给每行造一些“特征”（小时、星期几、是否周末……），供预测模型使用。
#
# -----------------------------------------------------------------------------
# ⚠️ 关于“负荷预测精度”的重要说明（本工程的已知限制，非模型问题）：
#   我们手里的负荷“日总量”来自能耗平台（每天合计，日采集粒度）。
#   当前的做法是：把每日总量 × 一条【实测 24h 形状】摊成逐小时负荷。
#   这意味着“小时级的真实逐时波动”无法被复原——模型只能学到
#   “历史同时刻的形状 × 当日总量水平”，故负荷预测的相对 RMSE 必然高于 PV。
#   —— 这是输入数据分辨率的物理上限，不是模型失效。
#   形状来源演进：
#     2026-08-27 前：福建工业园区双班“估算形态”（人为假设）
#     2026-08-27 起：电费清单 286 天逐时实测日均曲线
#     2026-09-14 起：消纳率表【8760 小时全年实测】（逐小时中位数，最权威）
#   三级来源的选择逻辑见本文件下半部分的 TYPICAL_LOAD_SHAPE。
#   —— 若想真正降低负荷预测误差，需拿到【园区逐时电量表】（替代日合计），
#      届时可直接喂真实小时序列，而不再需要“形状摊开”这一步。
# =============================================================================

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1) 典型日负荷形状（把“一天总用电量”分配到 24 个小时的比例）
# -----------------------------------------------------------------------------
# 2026-09-14 更新：负荷形状改为【三级来源自动选择】，优先级从高到低：
#   ① 消纳率表 8760 小时全年实测（福州烟草-消纳率计算.xlsx）→ 最权威
#   ② 电费清单 286 天逐时实测日均曲线（福州真实逐时负荷_日均曲线.csv）
#   ③ 福建工业园区双班估算形态（兜底）
#   数据源说明见 config.py 的 REAL_LOAD_SHAPE_CSV / SELF_CONSUMPTION_XLSX。
# ⚠️ 两个加载函数返回的都是【未归一】的原始数组（量纲为 kWh，不是占比）；
#    归一化在模块末尾统一完成，见 TYPICAL_LOAD_SHAPE 处的说明。
# -----------------------------------------------------------------------------

# 兜底估算形态（仅当实测 CSV 缺失/损坏时使用）：福建工业园区双班生产负荷形状。
_TYPICAL_LOAD_SHAPE_ESTIMATE = np.array([
    0.022, 0.020, 0.018, 0.017, 0.017, 0.019,   # 0-5 点（夜班/待机，低）
    0.030, 0.040, 0.052, 0.056, 0.057, 0.050,   # 6-11 点（早班爬坡+上午高峰）
    0.045, 0.052, 0.055, 0.056, 0.057, 0.054,   # 12-17 点（午休小凹+下午高峰）
    0.050, 0.046, 0.040, 0.034, 0.028, 0.024,   # 18-23 点（收工回落）
])


def _load_real_load_shape():
    """
    从“福州真实逐时负荷_日均曲线.csv”读取 24 小时实测形状(未归一)。

    返回长度 24 的 numpy 数组；读取失败/数据异常时返回 None（调用方退回估算）。
    CSV 结构：首行=小时表头(00:00..23:00)，数据行首列=指标名，
    其中 '日均负荷_kWh' 行为 24 小时日均电量(kWh)，即真实分时分布。
    """
    try:
        import os
        # 延迟导入 config，避免包导入顺序问题；REPO_DATA_DIR 已正确指向仓库根/数据资料
        from pvflex import config as _cfg
        csv_path = os.path.join(_cfg.REPO_DATA_DIR, '福州真实逐时负荷_日均曲线.csv')
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        if df.shape[1] < 25:
            return None
        row = df[df.iloc[:, 0].astype(str).str.strip() == '日均负荷_kWh']
        if len(row) == 0:
            return None
        vals = row.iloc[0, 1:25].to_numpy(dtype=float)
        if len(vals) == 24 and np.all(np.isfinite(vals)) and vals.sum() > 0:
            return vals
        return None
    except Exception:
        # 任何异常都退回估算，不让负荷形状影响主流程
        return None


def _load_shape_from_self_consumption():
    """
    【2026-09-14 新增】从“福州烟草-消纳率计算.xlsx”读取【8760 小时全年】实测负荷形状。

    这是比日均曲线更权威的来源：覆盖全年 365 天、8760 小时，逐小时取【中位数】，
    能有效抵抗个别极端日（检修停产/抄表异常）的干扰。

    ⚠️ 口径局限：该表“用电量”只覆盖部分计量点（全年约 333 万 kWh，为园区总量的
       ~36%，不含高压进线级总表）。但【归一化后的 24h 形状】只关心“每小时占比”，
       与总量口径无关，故用它作形状是可靠的。
       若该表不可用（文件缺失/openpyxl 未装），返回 None 由调用方退回其他来源。

    返回：长度 24 的 numpy 数组（未归一化，值为各小时中位用电量）；失败返回 None。
    """
    try:
        from pvflex.data.loaders import load_self_consumption_hourly
        df = load_self_consumption_hourly()
        if df is None or df.empty or 'load' not in df.columns:
            return None
        s = df['load'].dropna()
        if len(s) < 24 * 90:          # 至少 3 个月数据才认为可信
            return None
        by_hour = s.groupby(s.index.hour).median().reindex(range(24))
        if by_hour.isna().any() or by_hour.sum() <= 0:
            return None
        return by_hour.to_numpy(float)
    except Exception:
        return None


# 对外暴露的“典型日负荷形状”：按可靠性依次尝试三级来源，取第一个可用的。
#   ① 消纳率表 8760 小时【全年】实测（最权威；逐小时取中位数，抗极端日干扰）
#   ② 电费清单 286 天逐时实测日均曲线（次之）
#   ③ 福建工业园区双班估算形态（兜底，保证代码在任何数据缺失下都能跑）
# 下游(rolling_eval / mining / to_hourly_load) 直接引用本常量即可获得真实曲线。
#
# ⚠️⚠️ 2026-09-14 重要修正（必须归一化，否则“直接引用本常量的地方”会得到荒谬结果）：
#   三级来源返回的数组【量纲完全不同】，只有估算形态是“占比之和≈1”：
#     ③ 估算形态        → 24 个占比，sum≈1
#     ① 消纳率表 8760h  → 各小时【中位用电量】kWh，sum≈8240.8
#     ② 电费日均曲线    → 各小时【日均电量】kWh，  sum≈当日总电量
#   to_hourly_load() 内部有 shape/shape.sum() 归一化，走它的路径不受影响；
#   但【直接使用 TYPICAL_LOAD_SHAPE 的地方】（如 plots/stats）若以为它是占比，
#   就会把 8240.8 当成 100%（相当于把 712.8 当占比），使小时级负荷被放大数百倍：
#       错误：38901 kWh（日总量） × 712.8      = 27,728,633 kWh/h
#       正确：38901 kWh（日总量） × 0.0866     = 3,365 kWh/h
#   故此处【统一归一化为占比】，让 TYPICAL_LOAD_SHAPE 的语义在全工程内保持一致：
#   “长度 24、各元素为占比、sum=1”。这样任何下游用法都不会因来源切换而改变量纲。
#   原始（未归一）数组保留在 TYPICAL_LOAD_SHAPE_RAW，供需要真实 kWh 量纲的场景使用。
# -----------------------------------------------------------------------------
_LOAD_SHAPE_SOURCE = 'estimate'
_shape_full_year = _load_shape_from_self_consumption()
if _shape_full_year is not None:
    TYPICAL_LOAD_SHAPE = _shape_full_year
    _LOAD_SHAPE_SOURCE = 'self_consumption_8760h'
else:
    _real_shape = _load_real_load_shape()
    if _real_shape is not None:
        TYPICAL_LOAD_SHAPE = _real_shape
        _LOAD_SHAPE_SOURCE = 'hourly_bill_ndays'
    else:
        TYPICAL_LOAD_SHAPE = _TYPICAL_LOAD_SHAPE_ESTIMATE

# 统一归一化：保证对外暴露的 TYPICAL_LOAD_SHAPE 恒为“占比形态、sum=1”，
# 与来源无关。原始（未归一）数组保留在 TYPICAL_LOAD_SHAPE_RAW 里备查。
TYPICAL_LOAD_SHAPE_RAW = np.asarray(TYPICAL_LOAD_SHAPE, dtype=float).copy()
_ts = np.asarray(TYPICAL_LOAD_SHAPE, dtype=float).sum()
if _ts > 0:
    TYPICAL_LOAD_SHAPE = np.asarray(TYPICAL_LOAD_SHAPE, dtype=float) / _ts


def load_shape_source():
    """返回当前 TYPICAL_LOAD_SHAPE 的实际来源标识（便于报告/日志标注数据口径）。"""
    return _LOAD_SHAPE_SOURCE


def to_hourly_load(daily_load, typical_shape=None):
    """
    把“每天总负荷”摊成“每小时负荷”。

    参数
    ----
    daily_load : pandas.Series
        索引是日期、值是当天园区总用电量(kWh) 的序列。
    typical_shape : 长度 24 的数组，可选
        每个小时占总量的比例；不传就用本文件上面的 TYPICAL_LOAD_SHAPE。
        ⚠️ 传入的数组【不必是归一化的占比】——本函数内部会再归一一次，
           所以直接传 TYPICAL_LOAD_SHAPE_RAW（kWh 量纲）也能得到正确结果。

    返回
    ----
    pandas.Series，索引是“逐小时时间”，值是该小时负荷(kWh)，名字叫 'load_hourly'。
    """
    if typical_shape is None:
        typical_shape = TYPICAL_LOAD_SHAPE
    shape = np.asarray(typical_shape, dtype=float)
    # 归一化，保证 24 小时加起来 = 1。
    # ⚠️ 这一步使本函数对 TYPICAL_LOAD_SHAPE 是否已归一【不敏感】——
    #    无论传入占比还是 kWh 量纲，结果都正确。这也是它不受来源切换影响的原因。
    shape = shape / shape.sum()

    if daily_load is None or len(daily_load) == 0:
        return pd.Series(dtype=float, name='load_hourly')

    times = []
    values = []
    # 逐个日期处理：把这一天的“总量”按 shape 拆成 24 个小时
    for day, total in daily_load.items():
        for h in range(24):
            # 用日期 + 小时数拼出这一小时的时间戳
            ts = pd.Timestamp(day) + pd.Timedelta(hours=h)
            times.append(ts)
            values.append(float(total) * shape[h])
    s = pd.Series(values, index=times, name='load_hourly')
    return s.sort_index()


def align_pv_load(pv, load_hourly):
    """
    把 PV（每小时）和 负荷（每小时）按时间对齐成一张表。

    只保留两者都有数据的重叠时间段（外连接后删掉缺任一侧的行）。
    返回的 DataFrame 有两列：'pv' 和 'load'。

    实现说明（2026-09-14 核实）：pd.DataFrame({'pv':…, 'load':…}) 会**按索引对齐**，
    即使两侧索引长度不同也不会退化为按位置对齐（pandas 2.3.3 实测：长度 100 与 50
    对齐后仍是 DatetimeIndex，各自缺失处为 NaN，dropna 后取交集）。
    故本实现正确，无需改动。
    """
    df = pd.DataFrame({'pv': pv, 'load': load_hourly})
    df = df.dropna()          # 去掉任一侧缺失的小时
    df = df.sort_index()
    return df


def build_features(df):
    """
    给对齐后的“每小时”数据造特征，供预测模型使用。

    特征包括：
      hour      当前小时(0-23)
      dayofweek 星期几(0=周一 … 6=周日)
      month     月份
      is_weekend 是否周末(周六/周日)
      pv_yest   昨天同一时刻的 PV（滞后特征，帮助捕捉“连续晴天”之类规律）
      load_yest 昨天同一时刻的负荷

    返回：特征矩阵 X（numpy 数组）和对应的目标列 pv/load 仍在传入的 df 里。
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # 滞后一天同一小时：按"时间戳减 24h"查表（而非 shift(24) 按位置平移）。
    # ⚠️ 原因：数据有缺口时（如负荷缺失日），shift(24) 取到的是"往前 24 个位置"
    #    而非"昨日同时刻"——缺口后第一天的滞后特征会错位到更早的日期。
    #    按时间查表则严格取"昨日同一小时"，取不到（缺口后首日）就是 NaN，
    #    后续 dropna 会诚实剔除这些行，而不是用错值污染训练。
    lag_idx = df.index - pd.Timedelta(hours=24)
    df['pv_yest'] = df['pv'].reindex(lag_idx).to_numpy(float)
    df['load_yest'] = df['load'].reindex(lag_idx).to_numpy(float)

    # 真正用于建模的特征列
    feat_cols = ['hour', 'dayofweek', 'month', 'is_weekend', 'pv_yest', 'load_yest']
    X = df[feat_cols].to_numpy(dtype=float)
    return df, X, feat_cols


def split_train_test(df, test_ratio=0.15):
    """
    把数据按时间顺序切分：前面大部分当训练，最后一段当测试（检验预测准不准）。
    """
    if df is None or len(df) == 0:
        return df.iloc[0:0], df.iloc[0:0]
    n = len(df)
    n_test = max(1, int(round(n * test_ratio)))
    n_train = n - n_test
    train = df.iloc[:n_train]
    test = df.iloc[n_train:]
    return train, test
