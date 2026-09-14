# =============================================================================
# analysis/stats.py  ——  项目数据统计分析（方案 B：发电/负荷/消纳率/电费）
# -----------------------------------------------------------------------------
# 【这个文件解决什么问题】
#   把“项目真实数据”跑出一组能直接写进报告/汇报的直观结论：
#     1) 光伏发电量分析（年发电、月发电、容量系数）
#     2) 园区负荷分析（年用电、日均、峰谷、日负荷率）
#     3) 光伏消纳率（发出来的电自己用掉多少）
#     4) 电费与光伏收益（用平均电价做简化估算）
#   它是全工程最"朴素"的一环 —— 不做预测、不做优化、不训练模型，
#   就是老老实实把原始数据加加减减成结论。但它是后面所有高级分析的基础，
#   而且这些数字是你汇报时最先被问到的东西，所以准确性最重要。
#
# 【设计约定：每个“分析”都拆成两个函数】
#   analyze_xxx() / calc_xxx()   负责计算，返回 dict（数据，方便别处复用）
#   print_xxx()                  负责把结果打印成好看的一段文字
#   为什么这么拆？因为"算"和"显示"是两件不同的事：
#     算出来的 dict 可以被别的模块拿去继续用（比如 cli.py 把消纳率的结果
#     再传给电费计算），而 print 只管给人看。
#     如果混在一起写，别的模块想复用数字就只能去解析打印出来的文本，非常脆弱。
#   这样你以后想改某一项，只动对应的函数即可。
#
# 【本文件在整个工程里的位置】
#   数据流的第一站：读进来的原始 Series → 这里算出统计结论。
#   被 cli.py 的 run_stats() 调用。
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # 无界面也能存图
import matplotlib.pyplot as plt
import logging

import pvflex.config as cfg

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 小工具：让 matplotlib 能显示中文（否则中文变方框）
# -----------------------------------------------------------------------------
def set_chinese_font():
    """尝试设置中文字体；失败也不影响出图。

    【为什么需要这个函数？】
        matplotlib 默认字体里没有汉字，画图时所有中文会变成一个个空方框。
        解决办法是告诉它"用系统里的中文字体"。
        但不同电脑装的中文字体不一定一样，所以这里给了一个候选清单，
        从前往后试，谁在就用手谁。

    返回
    ----
    None（副作用：修改 matplotlib 的全局字体设置）

    ⚠️ 注意：如果候选清单里的字体一个都没装，函数不会报错，
       只是画出来的中文会是方框。这时你可以自己往清单里加字体名。
    """
    candidates = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Arial Unicode MS']
    for font in candidates:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示成方块
            return
        except Exception:
            continue


def save_fig(fig, filename):
    """把画好的图 fig 保存到 config.FIG_OUT_DIR 目录下。

    参数
    ----
    fig : matplotlib.figure.Figure
        已经画好的图对象。
    filename : str
        文件名，例如 'pv_monthly.png'。

    返回
    ----
    None（副作用：在磁盘上生成一个 png 文件，并释放图对象内存）
    """
    os.makedirs(cfg.FIG_OUT_DIR, exist_ok=True)      # 目录不存在就建，已存在也不报错
    path = os.path.join(cfg.FIG_OUT_DIR, filename)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    # dpi=120 控制清晰度；bbox_inches='tight' 表示"把多余的空白边裁掉"，
    # 否则图四周会留一大圈白边，插进 Word 里很难看。
    plt.close(fig)
    # ⚠️ 这行很重要：把图从内存里释放掉。如果不关，画几十张图后
    #    matplotlib 会警告"打开了太多图"并可能卡死。
    logger.info('已保存图片：%s', path)


# -----------------------------------------------------------------------------
# 1) 光伏发电量分析
# -----------------------------------------------------------------------------
def analyze_pv(pv):
    """输入逐小时发电量，算出光伏的各项统计指标。

    参数
    ----
    pv : pandas.Series
        每小时光伏发电量(kWh)，索引为时间戳。

    返回
    ----
    dict，字段含义：
        '年发电量_kWh'      : float，全期发电量合计（注意：是"全期"不是自然年）
        '日均发电量_kWh'    : float，按天汇总后的平均值
        '最大日发电量_kWh'  : float，发电最多的一天
        '最小日发电量_kWh'  : float，发电最少的一天
        '月发电量_kWh'      : pandas.Series，按月汇总的发电量
        '容量系数'          : float，年发电量 ÷ (额定容量 × 8760)；未配置额定容量时为 nan

    ⚠️ 如果传入空数据，不会报错，而是返回一串 0（"优雅降级"）。
       这样上层调用者不用到处写 if 判断。
    """
    if pv is None or len(pv) == 0:
        return {'年发电量_kWh': 0.0, '日均发电量_kWh': 0.0,
                '最大日发电量_kWh': 0.0, '最小日发电量_kWh': 0.0,
                '月发电量_kWh': pd.Series(dtype=float), '容量系数': float('nan')}
    daily = pv.resample('D').sum()      # 'D' = 按天(Day)汇总，sum() 求和
    monthly = pv.resample('ME').sum()   # 'ME' = 按月末(Month End)汇总
    # ⚠️ 为什么月度用 'ME' 而不是 'M'？新版 pandas 里 'M' 已被弃用，
    #    会打一堆警告。'ME' 表示"以月末为标签"，等价于原来的 'M'。
    result = {
        '年发电量_kWh': float(daily.sum()),
        '日均发电量_kWh': float(daily.mean()),
        '最大日发电量_kWh': float(daily.max()),
        '最小日发电量_kWh': float(daily.min()),
        '月发电量_kWh': monthly,
    }
    # 容量系数 = 年发电 / (额定容量 × 8760小时)
    # 8760 = 365 天 × 24 小时，也就是一年的总小时数。
    if cfg.PV_RATED_KW and not np.isnan(cfg.PV_RATED_KW):
        result['容量系数'] = float(result['年发电量_kWh'] / (cfg.PV_RATED_KW * 8760.0))
    else:
        result['容量系数'] = float('nan')   # nan = "不是一个数"，表示无法计算
    return result


def print_pv_summary(result):
    """把 analyze_pv() 的结果打印成人能读的一段文字。

    参数
    ----
    result : dict
        analyze_pv() 的返回值。

    返回
    ----
    None（只往屏幕打印）
    """
    print('\n========== 光伏发电量分析 ==========')
    print(f"  年发电量        : {result['年发电量_kWh']:,.1f} kWh")
    # 格式说明 {x:,.1f} = 千位加逗号 + 保留 1 位小数，例如 2,268,001.5
    print(f"  日均发电量      : {result['日均发电量_kWh']:,.1f} kWh")
    print(f"  最大日发电量    : {result['最大日发电量_kWh']:,.1f} kWh")
    print(f"  最小日发电量    : {result['最小日发电量_kWh']:,.1f} kWh")
    if not np.isnan(result['容量系数']):
        print(f"  容量系数        : {result['容量系数']*100:,.2f} %  "
              f"(额定容量 {cfg.PV_RATED_KW:g} kW)")
        # 上面 :g 表示"用最简洁的方式显示数字"，例如 1800.0 会显示成 1800
    else:
        print('  容量系数        : 未计算（请在 config.py 设置 PV_RATED_KW）')


# -----------------------------------------------------------------------------
# 2) 园区负荷分析
# -----------------------------------------------------------------------------
def analyze_load(load):
    """输入每天用电量，算出负荷的各项统计指标。

    参数
    ----
    load : pandas.Series
        每天园区总用电量(kWh)，索引为日期。

    返回
    ----
    dict，字段含义：
        '年用电量_kWh'      : float，全期用电量合计
        '日均用电量_kWh'    : float，日平均值
        '最大日用电量_kWh'  : float，用电最多的一天
        '最小日用电量_kWh'  : float，用电最少的一天
        '月用电量_kWh'      : pandas.Series，按月汇总
        '日负荷率'          : float，日均 ÷ 最大日，越接近 1 表示用电越平稳

    【"日负荷率"是什么意思？为什么它有用？】
        日负荷率 = 平均一天的用电 ÷ 最忙一天的用电。
        如果园区每天都在 3.8 万度左右波动，这个比例就接近 1（很平稳）；
        如果平时 1 万度、偶尔冲到 10 万度，比例就只有 0.1（波动很大）。
        波动大意味着变压器、线路要按最高峰值来配，平时却大量闲置 —— 不经济。
        这个指标也解释了为什么"储能削峰填谷"有价值：把高峰削下来能省容量费。
    """
    if load is None or len(load) == 0:
        return {'年用电量_kWh': 0.0, '日均用电量_kWh': 0.0,
                '最大日用电量_kWh': 0.0, '最小日用电量_kWh': 0.0,
                '月用电量_kWh': pd.Series(dtype=float), '日负荷率': 0.0}
    monthly = load.resample('ME').sum()
    daily_mean = float(load.mean())
    daily_max = float(load.max())
    return {
        '年用电量_kWh': float(load.sum()),
        '日均用电量_kWh': daily_mean,
        '最大日用电量_kWh': daily_max,
        '最小日用电量_kWh': float(load.min()),
        '月用电量_kWh': monthly,
        # 下面这个 if-else 是分母保护：最大日为 0 时不能做除法（会崩溃），
        # 所以检查一下，为 0 就返回 0。
        '日负荷率': daily_mean / daily_max if daily_max > 0 else 0.0,
    }


def print_load_summary(result):
    """把 analyze_load() 的结果打印出来。

    参数
    ----
    result : dict
        analyze_load() 的返回值。

    返回
    ----
    None（只往屏幕打印）
    """
    print('\n========== 园区负荷分析 ==========')
    print(f"  年用电量        : {result['年用电量_kWh']:,.1f} kWh")
    print(f"  日均用电量      : {result['日均用电量_kWh']:,.1f} kWh")
    print(f"  最大日用电量    : {result['最大日用电量_kWh']:,.1f} kWh")
    print(f"  最小日用电量    : {result['最小日用电量_kWh']:,.1f} kWh")
    print(f"  日负荷率        : {result['日负荷率']*100:,.2f} %  "
          f"(=日均/最大日，越大越平稳)")


# -----------------------------------------------------------------------------
# 3) 光伏消纳率
# -----------------------------------------------------------------------------
def calc_consumption_rate(pv_daily, load_daily):
    """计算光伏消纳率（发出来的电有多大比例被园区自己用掉了）。

    参数
    ----
    pv_daily : pandas.Series
        每天光伏发电量(kWh)，索引为日期。
    load_daily : pandas.Series
        每天园区用电量(kWh)，索引为日期。

    返回
    ----
    dict，字段含义：
        '对齐天数'              : int，两边都有数据的共同天数
        '期内总用电量_kWh'      : float，共同天数内的用电合计
        '期内被消纳发电量_kWh'  : float，共同天数内被园区用掉的光伏电量
        '消纳率'                : float，被消纳 ÷ 总用电（不是 ÷ 总发电！）

    ------ 详细说明（这段很重要，算法与项目方已对齐）------
    ✅ 消纳率口径已与项目方对齐（见“福州烟草-消纳率计算.xlsx”）：
        被消纳的用电量 = 用电量 - max(0, 用电量 - 发电量)
        消纳率         = 被消纳的用电量 / 用电量
    含义：光伏发电优先自用；自用不完的上网部分不计入“被消纳”，
         剩余用电由电网补充。这样光伏大发、用电小的日子，消纳率=100%
         （全部用电都被绿电覆盖），与项目方算法一致。

    用大白话再讲一遍这个公式：
        每天同时有"光伏发了多少"和"园区用了多少"两个数。分两种情况：
        ① 光伏发得比用电少（比如发 100 度、用 300 度）：
           100 度全被自己用掉 → 被消纳 = 100 度
        ② 光伏发得比用电多（比如发 500 度、用 300 度）：
           自己最多只能吃下 300 度，多出的 200 度只能上网卖 → 被消纳 = 300 度
        写成公式就是"被消纳 = 用电 - max(0, 用电 - 光伏)"，
        也就是 min(用电, 光伏) —— 取两者中的小者。

    ⚠️ 分母是"用电量"而不是"发电量"，这是本项目与项目方一致的约定。
        分母换成发电量会得到另一套数字（通常更小），两者不能混着比。
    """
    # 先取两个时间轴的交集：只比较"双方都有数据"的那些天。
    # 为什么必须这样做？因为光伏数据是 2025-01 开始的，用电量是 2025-03 开始的，
    # 不取交集的话，缺失的那一边会被当作 0，把消纳率算得面目全非。
    common_days = pv_daily.index.intersection(load_daily.index)
    pv_c = pv_daily.loc[common_days]
    load_c = load_daily.loc[common_days]
    pv_vals = pv_c.values        # 转成 numpy 数组，便于逐元素运算
    load_vals = load_c.values
    # 每天“被消纳的用电量” = 用电量 - max(0, 用电量 - 发电量)
    #   np.maximum(0.0, A-B) 是"逐元素取较大者"，即 max(0, A-B) 的数组版本。
    #   普通 Python 的 max() 一次只能比一个数，数组要用 np.maximum。
    consumed_by_day = load_vals - np.maximum(0.0, load_vals - pv_vals)
    consumed = float(consumed_by_day.sum())
    total_load = float(load_vals.sum())
    # 分母保护：总用电为 0 时不能做除法，返回 nan 而不是让程序崩掉。
    rate = consumed / total_load if total_load > 0 else float('nan')
    return {'对齐天数': len(common_days),
            '期内总用电量_kWh': total_load,
            '期内被消纳发电量_kWh': consumed,
            '消纳率': rate}


def print_consumption_summary(result):
    """把 calc_consumption_rate() 的结果打印出来。

    参数
    ----
    result : dict
        calc_consumption_rate() 的返回值。

    返回
    ----
    None（只往屏幕打印）
    """
    print('\n========== 光伏消纳率分析 ==========')
    print(f"  参与计算的天数  : {result['对齐天数']} 天")
    print(f"  期内总用电量    : {result['期内总用电量_kWh']:,.1f} kWh")
    print(f"  被园区自消纳量  : {result['期内被消纳发电量_kWh']:,.1f} kWh")
    if not np.isnan(result['消纳率']):
        print(f"  消纳率          : {result['消纳率']*100:,.2f} %")
    else:
        print('  消纳率          : 无法计算（用电量或发电量为 0）')


# -----------------------------------------------------------------------------
# 4) 电费与光伏收益（简化版）
# -----------------------------------------------------------------------------
def calc_cost(annual_load_kwh, pv_self_consumed_kwh):
    """用平均电价做一次最简单的电费/收益估算。

    参数
    ----
    annual_load_kwh : float
        统计期内的总用电量(kWh)。
        ⚠️ 注意是"期内"不是"全年" —— 必须与 pv_self_consumed_kwh 用同一时间段，
           否则口径不一致（详见 cli.py 里 run_stats 的注释）。
    pv_self_consumed_kwh : float
        同一时间段内被园区自消纳的光伏电量(kWh)。

    返回
    ----
    dict，字段含义：
        '平均电价_元每kWh'       : float，取自 config.TARIFF_AVG_YUAN_PER_KWH
        '年电费_元'              : float，用电量 × 平均电价
        '光伏自消纳量_kWh'       : float，原样传回，方便打印
        '光伏自发自用年节省_元'  : float，自消纳量 × 平均电价

    【这个估算有多"简化"？有两点要清楚】
      ① 用"平均电价"而不是"分时电价"。实际用电在峰段多还是谷段多，
         会明显影响电费，但这里不区分。
      ② 只用"自发自用"省下的钱，没算"余电上网"能卖多少钱。
      所以这个数字偏保守，属于"给个量级"的估算。
      精确的分时电费在 optimization 模块里算（那样才能体现储能的削峰填谷价值）。
    """
    tariff = cfg.TARIFF_AVG_YUAN_PER_KWH
    total_cost = annual_load_kwh * tariff
    saving = pv_self_consumed_kwh * tariff
    return {'平均电价_元每kWh': tariff,
            '年电费_元': float(total_cost),
            '光伏自消纳量_kWh': float(pv_self_consumed_kwh),
            '光伏自发自用年节省_元': float(saving)}


def print_cost_summary(result):
    """把 calc_cost() 的结果打印出来。

    参数
    ----
    result : dict
        calc_cost() 的返回值。

    返回
    ----
    None（只往屏幕打印）
    """
    print('\n========== 电费与光伏收益（简化） ==========')
    print(f"  采用平均电价    : {result['平均电价_元每kWh']:.2f} 元/kWh（示例值，见 config.py）")
    print(f"  估算电费(同口径): {result['年电费_元']:,.1f} 元（与消纳同口径的期内用电）")
    print(f"  光伏自消纳量    : {result['光伏自消纳量_kWh']:,.1f} kWh")
    print(f"  自发自用节省    : {result['光伏自发自用年节省_元']:,.1f} 元（=自消纳量×电价）")


# -----------------------------------------------------------------------------
# 5) 画 5 张图并保存
# -----------------------------------------------------------------------------
def make_figures(pv, load, pv_res, load_res, cons_res):
    """生成 5 张分析图到 config.FIG_OUT_DIR。

    参数
    ----
    pv : pandas.Series         逐小时光伏发电量
    load : pandas.Series       每天园区用电量
    pv_res : dict              analyze_pv() 的结果
    load_res : dict            analyze_load() 的结果
    cons_res : dict            calc_consumption_rate() 的结果

    返回
    ----
    None（副作用：在 figures/ 下生成 5 个 png 文件）

    5 张图分别是：
        1. pv_monthly.png          光伏发电量（按月柱状）
        2. load_monthly.png        园区用电量（按月柱状）
        3. pv_vs_load_monthly.png  发电 vs 用电（按月对比，最容易看出缺口）
        4. consumption_rate.png    消纳率（横向条）
        5. cost.png                电费 vs 光伏节省
    """
    set_chinese_font()

    # 图1：光伏发电量（按月）
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    # plt.subplots() 一次返回两个东西：整个画布(fig1) 和 坐标轴(ax1)。
    # 画图时主要操作 ax1（它可以理解成"这张纸上的绘图区域"）。
    pv_monthly = pv_res['月发电量_kWh']
    # X 轴标签用 strftime('%Y-%m') 把时间戳变成 '2025-09' 这种字符串。
    ax1.bar([d.strftime('%Y-%m') for d in pv_monthly.index], pv_monthly.values)
    ax1.set_title('光伏发电量（按月）'); ax1.set_ylabel('发电量 (kWh)')
    plt.xticks(rotation=45)     # X 轴文字转 45 度，否则月份多了会挤成一团
    save_fig(fig1, 'pv_monthly.png')

    # 图2：园区用电量（按月）
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    load_monthly = load_res['月用电量_kWh']
    ax2.bar([d.strftime('%Y-%m') for d in load_monthly.index], load_monthly.values, color='orange')
    ax2.set_title('园区用电量（按月）'); ax2.set_ylabel('用电量 (kWh)')
    plt.xticks(rotation=45)
    save_fig(fig2, 'load_monthly.png')

    # 图3：发电 vs 用电（按月对比）
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    x = np.arange(len(pv_monthly))     # [0,1,2,...] 作为柱子的位置编号
    # 两组柱子左右各错开 0.2、宽度 0.4：这样并排显示而不是叠在一起。
    ax3.bar(x - 0.2, pv_monthly.values, width=0.4, label='光伏发电')
    # reindex(..).fillna(0)：让用电量的月份顺序与发电量对齐；
    # 某些月份用电量没有数据，填 0 而不是留空（避免画图时报错）。
    ax3.bar(x + 0.2, load_monthly.reindex(pv_monthly.index).fillna(0).values,
            width=0.4, label='园区用电', color='orange')
    ax3.set_xticks(x)                  # 把刻度位置设成 [0,1,2,...]
    ax3.set_xticklabels([d.strftime('%Y-%m') for d in pv_monthly.index], rotation=45)
    # 因为刻度位置用了编号，所以要再用 set_xticklabels 把编号换成月份文字。
    ax3.set_title('光伏发电 vs 园区用电（按月）'); ax3.set_ylabel('电量 (kWh)')
    ax3.legend()      # 显示图例（那个小方框，标明哪个颜色是什么）
    save_fig(fig3, 'pv_vs_load_monthly.png')

    # 图4：消纳率水平条
    fig4, ax4 = plt.subplots(figsize=(6, 2.5))
    rate = cons_res['消纳率'] * 100     # 0.1835 → 18.35
    ax4.barh(['消纳率'], [rate], color='green')   # barh = 横向条形图（h 是 horizontal）
    ax4.set_xlim(0, 100); ax4.set_xlabel('%')     # X 轴固定 0~100，图才稳定
    ax4.set_title(f"光伏消纳率 {rate:.1f}%")
    save_fig(fig4, 'consumption_rate.png')

    # 图5：电费 vs 光伏节省
    # 与 run_stats 同口径：电费基数用"消纳同口径的期内总用电"，避免"全年电费配部分期消纳"的口径错配
    cost_res = calc_cost(cons_res['期内总用电量_kWh'], cons_res['期内被消纳发电量_kWh'])
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    ax5.bar(['年电费', '光伏年节省'], [cost_res['年电费_元'], cost_res['光伏自发自用年节省_元']],
            color=['red', 'green'])
    # 注意这里第一次出现"红=钱的花费、绿=省下来"的配色习惯，
    # 是普通的图表配色（不是股票涨跌那套红涨绿跌）。
    ax5.set_title('电费与光伏收益（简化）'); ax5.set_ylabel('金额 (元)')
    save_fig(fig5, 'cost.png')

    logger.info('5 张分析图已保存到 %s', cfg.FIG_OUT_DIR)
