# =============================================================================
# carbon/carbon.py  ——  碳排放相关计算（“能—碳”里的“碳”）
# -----------------------------------------------------------------------------
# 【这个文件解决什么问题】
#   研究方向3的核心是“能—碳”指标。前面 optimization 里已经把碳折算成钱，
#   这里单独把“碳排放量”的计算抽出来，方便你做碳报表、碳强度分析。
#   为什么要把"碳"单独成模块？因为汇报时"减了多少吨碳"往往是比"省了多少钱"
#   更有说服力的指标（尤其对烟草这类有双碳考核要求的行业），
#   需要能独立输出，而不是埋在优化结果里。
#
# 【核心概念】
#   - 园区用电来自两方面：自己光伏发的 + 从电网买的。
#   - 光伏是绿电，碳排≈0；从电网买的电才有碳排。
#   - 每小时碳因子不同（傍晚峰段煤电多，碳更高），这就是“碳的时变性”。
#
# 【本文件在整个工程里的位置】
#   被 optimization/dispatch.py（算目标函数里的碳成本）和报告输出引用。
#   单位约定：电量 kWh、碳排 kgCO2、碳因子 kgCO2/kWh。
# =============================================================================

import numpy as np

import pvflex.config as cfg


def grid_carbon_of_buy(buy_array, carbon_array):
    """计算“从电网买电”带来的碳排放量。

    参数
    ----
    buy_array    : 长度 24 的数组，每小时从电网买的电量(kWh)
    carbon_array : 长度 24 的数组，每小时的电网碳因子(kgCO2/kWh)

    返回
    ----
    float
        总碳排(kgCO2) = Σ(买电量 × 该小时碳因子)

    【为什么要"逐小时相乘"而不是"总量 × 平均碳因子"？】
        因为碳因子是时变的（这正是方向3强调的"时变性"）。
        假设一天买了 1000 度电，其中：
          夜里 500 度 × 0.379 = 189.5 kg
          傍晚 500 度 × 0.476 = 238.0 kg
          合计 427.5 kg
        如果偷懒用平均值 0.42 算：1000 × 0.42 = 420 kg，少了 7.5 kg。
        数字看着差别不大，但优化的目的恰恰就是"把买电时段从高碳挪到低碳"，
        用平均值就把这个可优化的空间抹平了 —— 优化器会失去减碳的动力。
    """
    # clip(min=0) 把负数截成 0：电量不可能是负的，负数说明上游算错了，
    # 这里做一道保险，避免出现"负碳排"这种荒谬结果。
    buy = np.asarray(buy_array, float).clip(min=0)
    carbon = np.asarray(carbon_array, float)
    return float(np.sum(buy * carbon))
    # np.sum(数组A * 数组B) 是"逐元素相乘再全部相加"，也就是数学上的点积，
    # 正好对应 Σ(买电量 × 碳因子)。


def net_grid_carbon(buy_array, sell_array, carbon_array):
    """计算园区“净购电”带来的碳排放量（与 optimization/dispatch 同一口径）。

    参数
    ----
    buy_array    : 长度 24 的数组，每小时从电网买的电量(kWh)
    sell_array   : 长度 24 的数组，每小时卖给电网的电量(kWh，绿电上网)
    carbon_array : 长度 24 的数组，每小时电网碳因子(kgCO2/kWh)

    返回
    ----
    float
        净碳排(kgCO2)；卖电多于买电的时段会自然产生碳信用(负值)。

    按企业碳核算 范围2 市场法：向电网买电计碳排，把光伏卖给电网（上网）计为
    碳信用冲减（绿电替代火电）。故净碳排 = Σ((买电量 − 卖电量) × 该小时碳因子)。

    【什么是"范围2"？】
        企业温室气体核算把排放分成三个范围：
          范围1 = 自己烧的（锅炉、车辆尾气）
          范围2 = 外购电力和热力带来的间接排放 ← 本模块算的就是这个
          范围3 = 上下游供应链
        本项目只涉及范围2（用电），因为园区自己不发电烧燃料。
    """
    buy = np.asarray(buy_array, float).clip(min=0)
    sell = np.asarray(sell_array, float).clip(min=0)
    carbon = np.asarray(carbon_array, float)
    return float(np.sum(carbon * (buy - sell)))
    # buy - sell 是"净购电量"，可能为负（卖得比买的多），
    # 此时该小时碳排为负 —— 表示"帮电网消纳了绿电、减少了火电出力"，
    # 这在碳核算里是一种可交易的信用额度。


def carbon_intensity(load_array, carbon_array):
    """计算这一天的“平均碳强度”（kgCO2 / kWh 用电）。

    参数
    ----
    load_array   : 长度 24 的数组，每小时园区总用电量(kWh)
    carbon_array : 长度 24 的数组，每小时电网碳因子(kgCO2/kWh)

    返回
    ----
    float
        碳强度 = 总碳排 / 总用电；若总用电为 0 返回 nan。

    【碳强度和碳排量有什么区别？】
        碳排量 = 排了多少（绝对量，单位 kgCO2）
        碳强度 = 每用一度电排多少（相对量，单位 kgCO2/kWh）
        举例：大厂碳排量大但碳强度可能低（因为用电效率高、绿电比例高）。
        做考核时两个指标都要看：碳排量看总量控制，碳强度看效率水平。
    """
    load = np.asarray(load_array, float).clip(min=0)
    carbon = np.asarray(carbon_array, float)
    total_load = float(load.sum())
    if total_load <= 0:
        return float('nan')     # 分母保护：没用电就谈不上"每度电的碳"
    # 简化：假设所有用电都来自电网（无光伏时的最坏碳强度）
    # ▲ 注意这个"最坏情况"假设：如果光伏同时也在供电，真实碳强度会比这个数低。
    #   本函数的定位是"给一个上限参考值"，不是精确核算。
    total_carbon = float(np.sum(load * carbon))
    return total_carbon / total_load


def annual_carbon_summary(buy_daily_kwh, grid_carbon_factor=None):
    """快速估算全年碳排（简化版，用固定平均碳因子）。

    参数
    ----
    buy_daily_kwh : pandas.Series 或 float
        全年每天从电网买的电量序列(Series) 或 标量年总量(kWh)。
    grid_carbon_factor : float 或 None
        平均电网碳因子(kgCO2/kWh)。
        默认 None → 取 config.CARBON_FACTOR_24H 的均值（与方向3一致）；
        也可显式传入（如 0.57 全国均值）做敏感性对比。

    返回
    ----
    dict，字段含义：
        '年电网买电量_kWh' : float
        '年碳排_kgCO2'     : float
        '等效种树_棵'      : float，粗略换算（1 棵树年吸 ~18kgCO2）

    ⚠️ 与 grid_carbon_of_buy 的区别：
        本函数用"全年平均碳因子"，精度低，只适合做"给外行看的量级换算"。
        需要精确数字（尤其要体现时变性）时必须用 grid_carbon_of_buy。
    """
    if grid_carbon_factor is None:
        grid_carbon_factor = float(np.mean(cfg.CARBON_FACTOR_24H))
    # hasattr(x, 'sum') 是"检查这个对象有没有 sum 方法"。
    # 有 → 说明传进来的是 Series（能求和）；没有（如普通数字）→ 直接当总量用。
    # 这样两种传法（Series 或 单个数字）都能兼容。
    if hasattr(buy_daily_kwh, 'sum'):
        annual_buy = float(buy_daily_kwh.sum())
    else:
        annual_buy = float(buy_daily_kwh)
    annual_carbon = annual_buy * grid_carbon_factor
    # "等效种树"是给非专业读者直观感受用的换算。18 kg/棵·年 是常见的粗略系数，
    # ⚠️ 不同树种、树龄差别很大，只能当"形象说法"，不能写进正式核算报告。
    trees = annual_carbon / 18.0
    return {
        '年电网买电量_kWh': annual_buy,
        '年碳排_kgCO2': annual_carbon,
        '等效种树_棵': trees,
    }
