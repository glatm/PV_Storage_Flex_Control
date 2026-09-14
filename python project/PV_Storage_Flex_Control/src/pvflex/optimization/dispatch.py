# =============================================================================
# optimization/dispatch.py  ——  方面2：基于鲁棒优化和随机规划的"能—碳"调控
# -----------------------------------------------------------------------------
# 对应申报书研究方向3原文：
#   "建立基于鲁棒优化和随机规划的'能—碳'指标提升调控模型，
#    在日前和实时两个时间尺度实现对'光储直柔'系统的主动调控"
#
# 【这个文件解决什么问题】
#   给定"今天 24 小时的光伏发电、园区负荷、电价、碳因子"，算出"电池每小时该
#   充电还是放电、该从电网买多少电、多余的电卖多少"，使得【电费 + 碳成本】最小。
#   这就是整个项目最核心的"调度"环节。
#
# 【决策量（每小时一个）】
#   charge    电池充电功率（kW）      discharge 电池放电功率（kW）
#   soc       电池剩余电量（kWh）     buy       向电网买电（kW）
#   sell      向电网卖电（kW）
#
# 【约束（必须满足的物理/商业规则）】
#   ① 能量守恒：光伏 + 放电 + 买电 = 负荷 + 充电 + 卖电
#      （任何一刻，电从哪来、到哪去都必须配平）
#   ② SOC 递推：这一小时的电池电量 = 上一小时 + 充进去的 − 放出来的
#      （充电乘效率 eff、放电除效率 eff——充放电都有损耗）
#   ③ 电池容量/功率上下限、SOC 上下限
#   ④ 买与卖不能同时发生（物理上不可能一边买一边卖）
#
# 【两个时间尺度（这是本课题的关键设计）】
#   日前(day-ahead)：用"预测值"一次性排好明天 24 小时的计划。
#                    因为预测不准，所以看多种可能场景 → 【随机规划】
#  实时(real-time)：边上边改。用"已经发生的真实值"，每隔几小时把剩下的重新排
#                    一遍 → 【MPC 模型预测控制】；并且给未来留"最坏情况"余量
#                    → 【鲁棒优化】
#
#   💡 MPC 就像开车：眼睛看前方路况（预测未来），手握方向盘（决定这一刻怎么动），
#      每开一小段就重新看一眼路（滚动更新），而不是出发前把全程方向打死。
#
# 【用什么算】
#   优先 PuLP（纯 Python 的线性/整数规划库，免费好装，配 CBC 求解器）→ 精确解；
#   若环境没装 PuLP 或求解失败，退回一条"见好就收"的启发式规则，保证也能跑。
#
# ⚠️ 【目标函数的两个口径必须一致（🔴 改这里要非常小心）】
#      目标：min Σ [ 买电价·buy − 上网价·sell + 碳价·碳因子·(buy−sell) ]
#      碳排口径：范围2 市场法——按【净购电】(buy−sell) 计碳排，绿电上网算碳信用冲减。
#      ⚠️ 目标函数里的碳成本项 与 返回的 carbon_total 必须用【同一口径】，
#         否则会"按一种口径优化、按另一种口径汇报"，结论就假了。
# =============================================================================

import numpy as np
import logging

import pvflex.config as cfg

logger = logging.getLogger(__name__)

# 把“碳排(kg)”折算成“钱(元)”的权重（见 config.CARBON_PRICE_YUAN_PER_KG）。
# 这样“电费”和“碳成本”就能加到一起，成为一个统一的“能—碳”目标。


# -----------------------------------------------------------------------------
# 核心：求解“给定一组 24 小时预测”下的最优调度
# -----------------------------------------------------------------------------
def solve_dispatch(pv_arr, load_arr, price_arr, carbon_arr, soc_init=None, robust_load_margin=0.0):
    """
    对一个 24 小时的 (PV, 负荷, 电价, 碳因子) 场景，求最优电池调度（MILP 精确解）。

    这是"标准版"求解器：用 PuLP + CBC 求混合整数线性规划（MILP），
    因为"买与卖互斥"用了二进制变量，所以是 MILP 而不是纯 LP。
    结果最权威，适合作为论文主结果的校验基准。

    参数
    ----
    pv_arr : 长度 24 的数组
        逐小时光伏发电量，单位 kWh（每小时的"千瓦时"等同于该小时的平均 kW）
    load_arr : 长度 24 的数组
        逐小时园区负荷，单位 kWh
    price_arr : 长度 24 的数组
        逐小时【买电】电价，单位 元/kWh
    carbon_arr : 长度 24 的数组
        逐小时电网碳因子，单位 kgCO2/kWh
    soc_init : float, 可选
        起始 SOC 比例（0~1）；不传则用 config.BATTERY_SOC_INIT
    robust_load_margin : float, 可选
        给未来负荷加的安全余量比例（0~1）。>0 即"鲁棒"——把负荷估计得更满一些，
        宁可多备点电，防止实际负荷比预测高时不够用。默认 0 表示不保守。

    返回
    ----
    dict，字段如下（数组均为长度 24）：
        'charge' / 'discharge' : 充 / 放电功率（kW）
        'soc'                  : 各小时末电池剩余电量（kWh）
        'buy' / 'sell'         : 向电网买 / 卖电（kW）
        'cost'                 : 全天电费，单位 元
        'carbon'               : 全天碳排，单位 kgCO2（净购电口径）
        'energy_carbon_index'  : "能—碳"综合指标 = 电费 + 碳价×碳排，单位 元
        'solver'               : 实际使用的求解器名（'PuLP' 或启发式）

    ⚠️ 求解失败时的行为：若 CBC 返回"未达最优"，本函数【自动退回启发式调度】
       并打 WARNING（而不是返回垃圾值）。所以拿到结果建议看一眼 'solver' 字段。
    """
    pv = np.asarray(pv_arr, float).clip(min=0)
    load = np.asarray(load_arr, float).clip(min=0)
    price = np.asarray(price_arr, float)
    carbon = np.asarray(carbon_arr, float)
    n = len(pv)

    cap = cfg.BATTERY_CAPACITY_KWH
    power = cfg.BATTERY_POWER_KW
    eff = cfg.BATTERY_EFF
    soc0 = (soc_init if soc_init is not None else cfg.BATTERY_SOC_INIT) * cap
    soc_min = cfg.BATTERY_SOC_MIN * cap
    soc_max = cfg.BATTERY_SOC_MAX * cap

    # ---- 尝试用 PuLP 精确求解 ----
    try:
        import pulp
    except Exception:
        logger.info('未安装 PuLP，使用启发式兜底调度。')
        return _heuristic_dispatch(pv, load, price, carbon, cap, power, eff,
                                   soc0, soc_min, soc_max)

    prob = pulp.LpProblem('energy_carbon_dispatch', pulp.LpMinimize)

    # 决策变量
    charge = [pulp.LpVariable(f'ch_{h}', lowBound=0, upBound=power) for h in range(n)]
    discharge = [pulp.LpVariable(f'dis_{h}', lowBound=0, upBound=power) for h in range(n)]
    soc = [pulp.LpVariable(f'soc_{h}', lowBound=soc_min, upBound=soc_max) for h in range(n)]
    buy = [pulp.LpVariable(f'buy_{h}', lowBound=0) for h in range(n)]
    sell = [pulp.LpVariable(f'sell_{h}', lowBound=0) for h in range(n)]
    y = [pulp.LpVariable(f'y_{h}', cat='Binary') for h in range(n)]   # 1=卖 0=买（与下方约束一致：buy<=BIG*(1-y), sell<=BIG*y）
    BIG = float(cap + np.max(load) + 1e3)

    # 目标：电费 + 碳成本。
    # 碳排按“净购电”口径（企业碳核算 范围2 市场法）：向电网买电计碳排，
    # 把光伏卖给电网（上网）计为碳信用冲减（绿电替代火电）。故碳成本用 (buy-sell)。
    # ⚠️ 目标函数与下方 carbon_total 必须用同一口径，否则求解方向与评估不一致。
    prob += pulp.lpSum(
        price[h] * buy[h] - cfg.GRID_SELL_PRICE_YUAN_PER_KWH * sell[h]
        + cfg.CARBON_PRICE_YUAN_PER_KG * carbon[h] * (buy[h] - sell[h])
        for h in range(n)
    )

    # 约束
    load_eff = load * (1 + robust_load_margin)   # 鲁棒：把未来负荷估计得更满一点
    for h in range(n):
        # 能量守恒：光伏 + 放电 + 买电 = 负荷 + 充电 + 卖电
        prob += (pv[h] + discharge[h] + buy[h] == load_eff[h] + charge[h] + sell[h])
        # SOC 递推：充电进 eff，放电出 1/eff
        if h == 0:
            prob += (soc[h] == soc0 + eff * charge[h] - discharge[h] / eff)
        else:
            prob += (soc[h] == soc[h - 1] + eff * charge[h] - discharge[h] / eff)
        # 买和卖不能同时（用二进制互斥）
        prob += buy[h] <= BIG * (1 - y[h])
        prob += sell[h] <= BIG * y[h]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # ⚠️ 必须检查求解状态：不可行/求解失败时 CBC 给的 varValue 是无意义值（甚至 None），
    #    若不拦截会静默流出"全零/垃圾调度"伪装成最优解。
    if prob.status != pulp.LpStatusOptimal:
        logger.warning('LP 求解未达最优（status=%s），退回启发式调度。',
                       pulp.LpStatus.get(prob.status, str(prob.status)))
        return _heuristic_dispatch(pv, load, price, carbon, cap, power, eff,
                                   soc0, soc_min, soc_max)

    def vals(vars_):
        return np.array([v.varValue or 0.0 for v in vars_])

    charge_v = vals(charge); discharge_v = vals(discharge)
    soc_v = vals(soc); buy_v = vals(buy); sell_v = vals(sell)

    cost = np.sum(price * buy_v - cfg.GRID_SELL_PRICE_YUAN_PER_KWH * sell_v)
    # 净购电碳排 = Σ 碳因子 × (买电 − 卖电)；卖绿电上网计碳信用冲减。
    carbon_total = np.sum(carbon * (buy_v - sell_v))
    z = cost + cfg.CARBON_PRICE_YUAN_PER_KG * carbon_total
    return {
        'charge': charge_v, 'discharge': discharge_v, 'soc': soc_v,
        'buy': buy_v, 'sell': sell_v,
        'cost': float(cost), 'carbon': float(carbon_total), 'energy_carbon_index': float(z),
        'solver': 'PuLP',
    }


def solve_dispatch_lp(pv_arr, load_arr, price_arr, carbon_arr, soc_init=None, robust_load_margin=0.0):
    """
    连续 LP 版调度求解（scipy.linprog / HiGHS），毫秒级，专用于全年滚动评估
    与大规模敏感性扫描。

    参数与返回：同 solve_dispatch（见其说明）。差别只在"怎么算"和"多快"。

    💡 【初学者小课堂：为什么需要这个"快版"，以及它凭什么和 MILP 等价】
       上一版 solve_dispatch 用 PuLP 求 MILP（含二进制变量），解一次要几百毫秒。
       但全年滚动评估要解 219 天 × 每天多次，敏感性扫描更是上千次——
       用 MILP 会慢到无法接受。所以这里用纯 LP（无二进制变量），快几百倍。

       那"买与卖不能同时"这个互斥条件怎么表达？
       关键洞察：因为【买电价 > 上网价】，最优解本来就不会又买又卖——
       那样等于"高价买进、低价卖出"，纯亏钱。所以互斥约束在最优解处
       【自动满足】，不需要显式写。

       技巧：把 (buy, sell) 两个变量合并成"有符号净交换量" g = buy − sell。
         · g > 0 表示净买电（要从电网买）
         · g < 0 表示净卖电（往电网上卖）
       目标函数对 g 是【分段线性的凸函数】：
         · g > 0 时斜率 = 买电价
         · g < 0 时斜率 = 上网价
       引入辅助变量 pos ≥ g 就能在纯 LP 里精确表达这个凸折线，
       于是 LP 与 MILP 在本问题上【严格等价】（校验差异 0.000%，见 validate_lp_vs_milp）。

       ⚠️ 等价性依赖"上网价 ≤ 最低零售价"这个凸性条件（已在 config 中保证）：
          如果哪天把上网价调得比买电价还高，就会"又买又卖套利"，
          凸性被破坏，LP 与 MILP 不再等价 —— 改电价时必须回来检查这一点。

    目标：min Σ [ 电价·buy − 上网价·sell + 碳价·碳因子·(buy−sell) ]
            ≡ min Σ [ (上网价 + 碳价·碳因子)·g + (买电价 − 上网价)·pos ]
    碳排口径（范围2 市场法）：净购电 (buy − sell) 计碳排，绿电上网冲减，与 MILP 一致。
    """
    try:
        from scipy.optimize import linprog
    except Exception:
        return solve_dispatch(pv_arr, load_arr, price_arr, carbon_arr, soc_init, robust_load_margin)

    pv = np.asarray(pv_arr, float).clip(min=0)
    load = np.asarray(load_arr, float).clip(min=0) * (1 + robust_load_margin)
    price = np.asarray(price_arr, float)                      # 买电价(逐时)
    sell_price = cfg.GRID_SELL_PRICE_YUAN_PER_KWH
    carbon = np.asarray(carbon_arr, float)
    n = len(pv)

    cap = cfg.BATTERY_CAPACITY_KWH
    power = cfg.BATTERY_POWER_KW
    eff = cfg.BATTERY_EFF
    soc0 = (soc_init if soc_init is not None else cfg.BATTERY_SOC_INIT) * cap
    soc_min = cfg.BATTERY_SOC_MIN * cap
    soc_max = cfg.BATTERY_SOC_MAX * cap

    # 变量顺序: charge(0..n-1), discharge(n..2n-1), soc(2n..3n-1), g(3n..4n-1), pos(4n..5n-1)
    N = 5 * n
    c = np.zeros(N)
    for h in range(n):
        # g 的系数 = 上网价 + 碳价·碳因子 ; pos 的系数 = 买电价 − 上网价 (>0)
        c[3 * n + h] = sell_price + cfg.CARBON_PRICE_YUAN_PER_KG * carbon[h]
        c[4 * n + h] = price[h] - sell_price

    A_eq = np.zeros((2 * n, N))
    b_eq = np.zeros(2 * n)
    for h in range(n):
        # 能量守恒: pv + discharge + g = load + charge   (g = buy − sell)
        A_eq[h, 1 * n + h] += 1.0
        A_eq[h, 3 * n + h] += 1.0
        A_eq[h, 0 * n + h] += -1.0
        b_eq[h] = load[h] - pv[h]
        # SOC 递推: soc_h = soc_{h-1} + eff·ch_h − dis_h/eff
        row = n + h
        A_eq[row, 2 * n + h] += 1.0
        if h == 0:
            A_eq[row, 0 * n + 0] += -eff
            A_eq[row, 1 * n + 0] += 1.0 / eff
            b_eq[row] = soc0
        else:
            A_eq[row, 2 * n + (h - 1)] += -1.0
            A_eq[row, 0 * n + h] += -eff
            A_eq[row, 1 * n + h] += 1.0 / eff
            b_eq[row] = 0.0

    # 不等式约束 pos ≥ g  ⇔  −pos + g ≤ 0
    A_ub = np.zeros((n, N))
    b_ub = np.zeros(n)
    for h in range(n):
        A_ub[h, 4 * n + h] = -1.0
        A_ub[h, 3 * n + h] = 1.0

    # 边界：charge,discharge∈[0,power]；soc∈[soc_min,soc_max]；
    #       g∈[−BIG, +BIG]；pos∈[0, BIG]（BIG 与 MILP 的买卖上界一致，作数值安全网）。
    #       当上网价≤最低零售价（config 已保证）时成本函数凸，LP 与 MILP 严格等价，
    #       pos 在非套利小时自然取到 max(g,0)，BIG 上界不会触发。
    BIG = float(cap + np.max(load) + 1e3)
    bounds = [(0.0, power)] * n + [(0.0, power)] * n + [(soc_min, soc_max)] * n + \
             [(-BIG, BIG)] * n + [(0.0, BIG)] * n

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not res.success:
        logger.warning('LP 求解失败（%s），退回 MILP。', res.message)
        return solve_dispatch(pv_arr, load_arr, price_arr, carbon_arr, soc_init, robust_load_margin)

    x = res.x
    charge_v = x[0 * n:1 * n]
    discharge_v = x[1 * n:2 * n]
    soc_v = x[2 * n:3 * n]
    g_v = x[3 * n:4 * n]
    # 由 g 还原买/卖（互斥，结构保证）
    buy_v = np.maximum(g_v, 0.0)
    sell_v = np.maximum(-g_v, 0.0)
    cost = float(np.sum(price * buy_v - sell_price * sell_v))
    carbon_total = float(np.sum(carbon * g_v))
    z = cost + cfg.CARBON_PRICE_YUAN_PER_KG * carbon_total
    return {
        'charge': charge_v, 'discharge': discharge_v, 'soc': soc_v,
        'buy': buy_v, 'sell': sell_v,
        'cost': cost, 'carbon': carbon_total, 'energy_carbon_index': float(z),
        'solver': 'linprog',
    }


# ---- 没装 PuLP 时的启发式兜底（规则：低价充、高价放、多余光伏存起来）----
def _heuristic_dispatch(pv, load, price, carbon, cap, power, eff, soc0, soc_min, soc_max):
    soc = soc0; charge = []; discharge = []; buy = []; sell = []
    mean_price = np.mean(price)
    for h in range(len(pv)):
        net = pv[h] - load[h]                      # >0 光伏用不完，<0 不够
        if net >= 0:                              # 光伏过剩：先充电，再卖电
            c = min(power, (soc_max - soc) / eff, net)
            c = max(0.0, c)
            charge.append(c); discharge.append(0.0)
            sell.append(net - c); buy.append(0.0)
            soc += eff * c
        else:                                     # 光伏不足：先看电价高不高
            need = -net
            if price[h] > mean_price:              # 电价高：放电顶上
                d = min(power, (soc - soc_min) * eff, need)
                d = max(0.0, d)
                discharge.append(d); charge.append(0.0)
                buy.append(need - d); sell.append(0.0)
                soc -= d / eff
            else:                                 # 电价低：少放电，多买电
                discharge.append(0.0); charge.append(0.0)
                buy.append(need); sell.append(0.0)
    charge = np.array(charge); discharge = np.array(discharge)
    soc_arr = np.cumsum(eff * charge - discharge / eff) + soc0
    buy = np.array(buy); sell = np.array(sell)
    cost = np.sum(price * buy - cfg.GRID_SELL_PRICE_YUAN_PER_KWH * sell)
    carbon_total = np.sum(carbon * (buy - sell))
    z = cost + cfg.CARBON_PRICE_YUAN_PER_KG * carbon_total
    return {'charge': charge, 'discharge': discharge, 'soc': soc_arr,
            'buy': buy, 'sell': sell, 'cost': float(cost),
            'carbon': float(carbon_total), 'energy_carbon_index': float(z),
            'solver': 'heuristic'}


# -----------------------------------------------------------------------------
# 无储能基线：没有电池，光伏不够就买、用不完就卖
# -----------------------------------------------------------------------------
def baseline_no_storage(pv_arr, load_arr, price_arr, carbon_arr):
    pv = np.asarray(pv_arr, float).clip(min=0)
    load = np.asarray(load_arr, float).clip(min=0)
    price = np.asarray(price_arr, float)
    carbon = np.asarray(carbon_arr, float)
    buy = np.maximum(load - pv, 0)          # 缺多少买多少
    sell = np.maximum(pv - load, 0)         # 多多少卖多少
    cost = np.sum(price * buy - cfg.GRID_SELL_PRICE_YUAN_PER_KWH * sell)
    carbon_total = np.sum(carbon * (buy - sell))
    z = cost + cfg.CARBON_PRICE_YUAN_PER_KG * carbon_total
    return {'buy': buy, 'sell': sell, 'cost': float(cost),
            'carbon': float(carbon_total), 'energy_carbon_index': float(z)}


# -----------------------------------------------------------------------------
# 日前随机规划：看多种场景，算期望成本与波动
# -----------------------------------------------------------------------------
def run_day_ahead_stochastic(pv_nom, load_nom, price_24, carbon_24, pv_std=0.0, solver='milp'):
    """
    用名义预测 + 几个扰动场景，做“随机规划”示意：
    每个场景单独求最优，再取平均（期望），并报告场景间差异（不确定性代价）。

    solver : 'milp'（默认，PuLP/CBC 权威）或 'lp'（scipy 连续松弛，用于大规模扫描）。
    """
    pv_nom = np.asarray(pv_nom, float); load_nom = np.asarray(load_nom, float)
    solver_fn = solve_dispatch_lp if solver == 'lp' else solve_dispatch
    scenarios = {
        'nominal': (pv_nom, load_nom),
        'pv+10%': (pv_nom * 1.10, load_nom),
        'pv-10%': (pv_nom * 0.90, load_nom),
        'load+5%': (pv_nom, load_nom * 1.05),
        'load-5%': (pv_nom, load_nom * 0.95),
    }
    results = {}
    for name, (p, l) in scenarios.items():
        results[name] = solver_fn(p, l, price_24, carbon_24)
    costs = np.array([r['cost'] for r in results.values()])
    carbons = np.array([r['carbon'] for r in results.values()])
    return {
        'scenarios': results,
        'expected_cost': float(np.mean(costs)),
        'cost_std': float(np.std(costs)),
        'expected_carbon': float(np.mean(carbons)),
        'nominal': results['nominal'],
    }


# -----------------------------------------------------------------------------
# 实时鲁棒：滚动 MPC（边上边改），并与“开环日前计划”对比
# -----------------------------------------------------------------------------
def run_real_time_robust(pv_nom, load_nom, price_24, carbon_24, soc_init=None,
                         noise=0.03, robust_margin=0.05):
    """
    模拟“实时”运行：每小时用名义预测+随机噪声当作“真实值”，
    然后从当前时刻起把剩下 24 小时重新排一遍（MPC）；
    鲁棒版额外给未来负荷加 safety margin。
    返回开环计划、确定性MPC执行、鲁棒MPC执行 三套结果，便于对比。
    """
    n = len(pv_nom)
    rng = np.random.default_rng(0)
    # 模拟真实值（名义 + 噪声）
    pv_real = np.maximum(pv_nom * (1 + noise * rng.standard_normal(n)), 0)
    load_real = np.maximum(load_nom * (1 + noise * rng.standard_normal(n)), 0)

    # 开环：一开始就按名义预测定死一整天
    open_loop = solve_dispatch(pv_nom, load_nom, price_24, carbon_24, soc_init=soc_init)

    # MPC 滚动：每到一个小时，用“真实值走过的部分 + 名义预测剩下的部分”重算
    def mpc(robust):
        soc = (soc_init if soc_init is not None else cfg.BATTERY_SOC_INIT) * cfg.BATTERY_CAPACITY_KWH
        charge_t, discharge_t, buy_t, sell_t = [], [], [], []
        for t in range(n):
            # 过去/当前用真实值；未来用名义预测。
            # ⚠️ 鲁棒余量只能加在【未来时段】：已发生的真实小时负荷是确定的，
            #    若给它加 margin，执行动作会与真实负荷失衡（幻影购电虚增鲁棒成本，
            #    audit8 实测幻影折价~1374 元占成本差 80%）。故 margin 直接写入 ld_fc
            #    的未来段，给 solve_dispatch 传 robust_load_margin=0（避免其再对全 24h
            #    一律加 margin）。
            pv_fc = np.concatenate([pv_real[:t + 1], pv_nom[t + 1:]])
            if robust:
                ld_fc = np.concatenate([load_real[:t + 1],
                                        load_nom[t + 1:] * (1 + robust_margin)])
            else:
                ld_fc = np.concatenate([load_real[:t + 1], load_nom[t + 1:]])
            r = solve_dispatch(pv_fc, ld_fc, price_24, carbon_24,
                               soc_init=soc / cfg.BATTERY_CAPACITY_KWH,
                               robust_load_margin=0.0)
            # 把“这一步”的决策当作实时实际执行的动作，逐步累积成一条执行轨迹；
            # 这样鲁棒版（未来负荷被放大）与确定版会走出不同轨迹 → 成本不同，
            # 对比才有意义（鲁棒版通常更保守、成本略高）。
            charge_t.append(r['charge'][t]); discharge_t.append(r['discharge'][t])
            buy_t.append(r['buy'][t]); sell_t.append(r['sell'][t])
            soc = r['soc'][t]            # 取这一步的 SOC 作为下一步起点
        charge_t = np.array(charge_t); discharge_t = np.array(discharge_t)
        buy_t = np.array(buy_t); sell_t = np.array(sell_t)
        cost = np.sum(price_24 * buy_t - cfg.GRID_SELL_PRICE_YUAN_PER_KWH * sell_t)
        carbon_total = np.sum(carbon_24 * (buy_t - sell_t))
        z = cost + cfg.CARBON_PRICE_YUAN_PER_KG * carbon_total
        return {'charge': charge_t, 'discharge': discharge_t, 'buy': buy_t, 'sell': sell_t,
                'cost': float(cost), 'carbon': float(carbon_total),
                'energy_carbon_index': float(z), 'solver': r['solver']}

    mpc_det = mpc(robust=False)
    mpc_rob = mpc(robust=True)
    return {'open_loop': open_loop, 'mpc_det': mpc_det, 'mpc_rob': mpc_rob,
            'pv_real': pv_real, 'load_real': load_real}


# -----------------------------------------------------------------------------
# 汇总打印
# -----------------------------------------------------------------------------
def print_optimize_summary(da, rt, baseline):
    print('\n================ 方面2：鲁棒优化 + 随机规划 能-碳调控 ================')
    print(f'调度求解器：{da["nominal"]["solver"]}（日前随机规划 + 实时滚动MPC）')
    print('—— 日前随机规划（多场景期望）——')
    print(f'  期望总成本 ≈ {da["expected_cost"]:.1f} 元，场景间波动(标准差) ≈ {da["cost_std"]:.1f} 元')
    print(f'  期望总碳排 ≈ {da["expected_carbon"]:.1f} kgCO2')

    print('—— 与“无储能”基线对比（越低越好）——')
    base_c, base_co = baseline['cost'], baseline['carbon']
    new_c, new_co = da['nominal']['cost'], da['nominal']['carbon']
    e_pct = (base_c - new_c) / base_c * 100
    c_pct = (base_co - new_co) / base_co * 100
    print(f'  电费：基线 {base_c:.1f} 元  →  优化后 {new_c:.1f} 元  '
          f'({"省" if e_pct >= 0 else "增"} {abs(e_pct):.1f}%)')
    print(f'  碳排：基线 {base_co:.1f} kg →  优化后 {new_co:.1f} kg '
          f'({"降" if c_pct >= 0 else "升"} {abs(c_pct):.1f}%)')

    print('—— 实时滚动(MPC) vs 开环日前计划 ——')
    print(f'  开环计划成本：{rt["open_loop"]["cost"]:.1f} 元')
    print(f'  确定性MPC执行：{rt["mpc_det"]["cost"]:.1f} 元')
    print(f'  鲁棒MPC执行  ：{rt["mpc_rob"]["cost"]:.1f} 元（对未来负荷留安全余量）')
    # 鲁棒余量只加在未来时段（已发生小时负荷确定，不加 margin → 不产生幻影购电）。
    # 储能充裕时鲁棒与确定版成本可相同（余量未触顶）；预测偏乐观的最坏情况下鲁棒版更稳。
    print('  （鲁棒版对未来负荷留余量；储能充裕时与确定版成本可相同，'
          '预测偏乐观的最坏情况下鲁棒版更稳）')
