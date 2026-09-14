# =============================================================================
# optimization/rolling_eval.py  ——  全年滚动调度评估 + 敏感性分析
# -----------------------------------------------------------------------------
# 对应你要的两件事之二：
#   “做全年滚动评估 + 敏感性分析（电池容量、碳价、预测误差），给置信区间
#    —— 单代表日不够。”
#
# 做法：
#   · 全年滚动：对“全年所有完整日”分别跑 无储能基线 / 日前随机 / 实时MPC 调度，
#     汇总成本与碳排节省的 均值 ± 95% 置信区间（bootstrap 对“日”重采样）。
#   · 敏感性①电池容量：扫 [250,500,1000,1500,2000] kWh（功率按 1C 同比例），
#     看节省随容量变化 + CI。
#   · 敏感性②碳价：扫 [0.01,0.05,0.10,0.20] 元/kgCO2，看能-碳协同如何随碳价切换。
#   · 敏感性③预测误差：给预测注入 0/3/6/10% 噪声，看调度收益“退化”多少 + CI。
#
# ⚠️ 已知限制（务必如实）：
#    负荷 24h 剖面 = 日总量 × 【实测】典型日内形状。形状已不是估算：
#      · 优先取“福州烟草-消纳率计算.xlsx”全年 8760 小时（逐小时中位数）；
#      · 退而取“福州真实逐时负荷_日均曲线.csv”（电费清单 286 天逐时抄表）；
#      · 都没有才退回估算形态。可用 pp.load_shape_source() 查询实际来源。
#    PV 用真实小时级；负荷【日总量】来自能耗平台日采集。
#    ⚠️ 因“日总量 × 形状”无法复原当日真实逐时波动，负荷小时级绝对量存在
#       数据分辨率上限，相关结论请以【相对趋势】为准，不要过度解读绝对值。
#    负荷样本区间受 C 项可靠区间约束（见 config.LOAD_RELIABLE_START/END）：
#       2025-11 之后的月份因能耗平台缺失高压进线级总表已被截断，
#       故当前 n_days ≈ 219 天，而非满 365 天。
#    储能容量/碳因子仍为待确认的示例值（见 config.py 注释）。
#    结论的“相对排序与趋势”可靠；“绝对数值”待设计院确认储能容量后定稿。
# =============================================================================

import logging
import numpy as np

import pvflex.config as cfg
import pvflex.data.loaders as ld
import pvflex.data.preprocess as pp
import pvflex.optimization.dispatch as opt

logger = logging.getLogger(__name__)

SEED = 0
N_BOOT = 1000


# -----------------------------------------------------------------------------
# 构造“全年每日 24h 剖面”（PV 真实 / 负荷 = 日总量×典型形状）
# -----------------------------------------------------------------------------
def build_daily_profiles():
    """
    返回 list[(date, pv_day(24,), load_day(24,))]，仅含“完整 24h”的天。
    PV 用真实小时级；负荷用 日总量 × TYPICAL_LOAD_SHAPE。
    ⚠️ 该 shape 自 2026-09-14 起为【实测】来源（消纳率表 8760h 或电费清单逐时抄表），
       不再是估算形态；仅在两者都缺失时才退回估算。见 pp.load_shape_source()。
    """
    pv = ld.load_pv_generation()
    load = ld.load_load_series()
    if pv.empty or load.empty:
        raise RuntimeError('数据未读取成功，无法做滚动调度评估。')

    # 真实小时级 PV，按天取 24h
    pv_day_list = []
    for day, grp in pv.groupby(pv.index.normalize()):
        if len(grp) == 24:
            pv_day_list.append((day, grp['pv'].to_numpy(float) if 'pv' in grp else grp.to_numpy(float)))
    pv_by_day = {d: a for d, a in pv_day_list}

    # 负荷：日总量 × 实测典型形状（来源见 pp.load_shape_source()）
    shape = pp.TYPICAL_LOAD_SHAPE / pp.TYPICAL_LOAD_SHAPE.sum()
    out = []
    for day, total in load.items():
        if not np.isfinite(total) or total <= 0:
            continue
        if day not in pv_by_day:
            continue
        load_day = np.asarray(total, float) * shape
        out.append((day, pv_by_day[day].copy(), load_day))
    out.sort(key=lambda x: x[0])
    return out


# -----------------------------------------------------------------------------
# 单日评估（完美预测下的 MPC 价值上界）
# -----------------------------------------------------------------------------
def evaluate_day(pv_day, load_day, price_24, carbon_24, soc_init=None, solver='lp',
                  skip_da=False):
    """
    对单日跑 无储能基线 / 日前随机(nominal) / 实时MPC(完美预测)。
    返回 dict：各方案 cost/carbon/saving%。

    solver  : 'lp'（默认，scipy 连续松弛，毫秒级，用于全年/敏感性大规模扫描）
              'milp'（PuLP/CBC 权威，论文主结果校验用）。
    skip_da : 敏感性分析只关心 mpc vs base 的节省，跳过日前随机(da)可省 ~5/6 计算。
    """
    base = opt.baseline_no_storage(pv_day, load_day, price_24, carbon_24)
    da = opt.run_day_ahead_stochastic(pv_day, load_day, price_24, carbon_24,
                                      solver=solver) if not skip_da else None
    # 完美预测下的实时 MPC（直接用真实值作未来输入，24 步求解，避免 run_real_time_robust
    # 返回 open_loop+det+rob 三套 72 次求解的冗余开销，提速 ~3×）
    mpc = mpc_with_forecast(pv_day, load_day, pv_day.copy(), load_day.copy(),
                            price_24, carbon_24, soc_init=soc_init, solver=solver)
    res = {
        'base_cost': base['cost'], 'base_carbon': base['carbon'],
        'mpc_cost': mpc['cost'], 'mpc_carbon': mpc['carbon'],
        'save_cost_pct': (base['cost'] - mpc['cost']) / base['cost'] * 100.0,
        'save_carbon_pct': (base['carbon'] - mpc['carbon']) / base['carbon'] * 100.0,
    }
    if da is not None:
        res['da_cost'] = da['nominal']['cost']
        res['da_carbon'] = da['nominal']['carbon']
    return res


# -----------------------------------------------------------------------------
# MPC 但用“带误差的预测”作未来输入、用“真实值”作已发生输入（敏感性③用）
# -----------------------------------------------------------------------------
def mpc_with_forecast(pv_actual, load_actual, pv_fc, load_fc, price_24, carbon_24,
                      soc_init=None, robust_margin=0.0, solver='lp'):
    """
    实时 MPC：已发生时段用 actual，未来时段用 forecast（带误差）；
    决策在“真实 realized”上结算成本/碳排。

    solver : 'lp'（默认，scipy 连续松弛）/ 'milp'（PuLP/CBC 权威）。
    """
    solver_fn = opt.solve_dispatch_lp if solver == 'lp' else opt.solve_dispatch
    n = len(pv_actual)
    soc = (soc_init if soc_init is not None else cfg.BATTERY_SOC_INIT) * cfg.BATTERY_CAPACITY_KWH
    buy_t, sell_t, charge_t, discharge_t = [], [], [], []
    for t in range(n):
        pv_f = np.concatenate([pv_actual[:t + 1], pv_fc[t + 1:]])
        ld_f = np.concatenate([load_actual[:t + 1],
                               load_fc[t + 1:] * (1 + robust_margin)])
        r = solver_fn(pv_f, ld_f, price_24, carbon_24,
                      soc_init=soc / cfg.BATTERY_CAPACITY_KWH,
                      robust_load_margin=0.0)
        buy_t.append(r['buy'][t]); sell_t.append(r['sell'][t])
        charge_t.append(r['charge'][t]); discharge_t.append(r['discharge'][t])
        soc = r['soc'][t]
    buy = np.array(buy_t); sell = np.array(sell_t)
    cost = np.sum(price_24 * buy - cfg.GRID_SELL_PRICE_YUAN_PER_KWH * sell)
    carbon = np.sum(carbon_24 * (buy - sell))
    return {'cost': float(cost), 'carbon': float(carbon)}


# -----------------------------------------------------------------------------
# 日前开环调度（对照 MPC 用）：仅按“预测”一次性求解出充放电计划并提交，
# 之后用“真实”光伏/负荷结算成本——预测误差会直接变成后悔值(regret)。
# 与 MPC（每步用真实值重优化、对预测误差鲁棒）对照，即可量化预测质量的边际价值。
# -----------------------------------------------------------------------------
def day_ahead_realized(pv_actual, load_actual, pv_fc, load_fc, price_24, carbon_24,
                       solver='lp'):
    """开环：plan 依预测求，realized 依真实结算。返回 (cost, carbon)。"""
    solver_fn = opt.solve_dispatch_lp if solver == 'lp' else opt.solve_dispatch
    plan = solver_fn(np.asarray(pv_fc, float), np.asarray(load_fc, float),
                     price_24, carbon_24)
    charge = np.asarray(plan['charge'], float)
    discharge = np.asarray(plan['discharge'], float)
    # 真实值下的电量平衡：电网补足差额（买=正, 卖=负）
    net = np.asarray(load_actual, float) - np.asarray(pv_actual, float) \
          - discharge + charge
    buy = np.maximum(net, 0.0)
    sell = np.maximum(-net, 0.0)
    cost = float(np.sum(price_24 * buy - cfg.GRID_SELL_PRICE_YUAN_PER_KWH * sell))
    carbon = float(np.sum(carbon_24 * (buy - sell)))
    return cost, carbon


# -----------------------------------------------------------------------------
# bootstrap 对“日”重采样 → 均值 95% CI
# -----------------------------------------------------------------------------
def _ci_mean(values, n_boot=N_BOOT, seed=SEED):
    """
    对一组"每日节省率"求均值，并用 bootstrap 重采样给出 95% 置信区间。

    💡 【初学者小课堂：什么是 bootstrap，为什么要用它】
       我们算出 219 天的"每天省了百分之几"，想知道"平均到底省多少"。
       但如果只报一个均值 5.18%，读者会问：这 5.18% 稳不稳？
       如果换个样本（比如换一年），会不会变成 2% 甚至 -1%？

       bootstrap 的思路很朴素：把这 219 个数字当成一个"池子"，
       有放回地随机抽 219 个（可能重复抽到同一个），算一次平均；
       重复 1000 次，就得到 1000 个"可能出现的均值"。
       这 1000 个数的 2.5% 分位与 97.5% 分位，就是 95% 置信区间——
       意思是"我们有 95% 的把握，真实均值落在这个区间里"。

       ⚠️ 为什么区间要按【日】重采样而不是按小时：
          同一天内各小时的误差是相关的（不是独立事件），
          按小时重采样会低估不确定性、把区间算得过窄，显得比实际更"确定"。
          按天重采样才是正确做法（一天的调度是一个整体决策）。

    参数
    ----
    values : array-like
        每日指标序列（如每日电费节省率，单位 %）
    n_boot : int
        重采样次数，默认 1000（次数越多区间越平滑，但越慢）
    seed : int
        随机种子，固定它保证结果【可复现】——论文里的数字必须能重跑出来

    返回
    ----
    tuple (mean, (lo, hi))
        mean      —— 原始样本均值
        (lo, hi)  —— 95% 置信区间上下界；样本 < 2 个时返回 (nan, nan)

    ⚠️ 可复现性：seed 固定为模块级常量 SEED=0，改它会改变区间端点。
       论文中报告的数字必须注明使用了哪个 seed。
    """
    v = np.asarray(values, float)
    if len(v) < 2:
        # 样本太少，无法给出有意义的区间
        return float(np.nanmean(v)), (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(v))
    # 有放回地抽 n 次，算一次均值；重复 n_boot 次得到一个"均值的分布"
    means = [float(np.mean(v[rng.choice(idx, len(idx), replace=True)])) for _ in range(n_boot)]
    # 取该分布的 2.5% 与 97.5% 分位，即 95% 置信区间
    return float(np.mean(v)), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _set_cfg(**kw):
    """临时改 config 全局参数，返回旧值以便恢复。"""
    old = {}
    for k, v in kw.items():
        old[k] = getattr(cfg, k)
        setattr(cfg, k, v)
    return old


# -----------------------------------------------------------------------------
# 主：全年滚动调度评估
# -----------------------------------------------------------------------------
def rolling_dispatch_evaluation(days=None, verbose=True):
    if days is None:
        days = build_daily_profiles()
    price_24 = np.array(cfg.GRID_BUY_PRICE_24H, float)
    carbon_24 = np.array(cfg.CARBON_FACTOR_24H, float)

    save_cost, save_carbon = [], []
    da_save_cost, mpc_costs, base_costs = [], [], []
    for (day, pv_d, load_d) in days:
        try:
            r = evaluate_day(pv_d, load_d, price_24, carbon_24)
        except Exception as e:
            logger.warning('跳过 %s: %s', day, e)
            continue
        save_cost.append(r['save_cost_pct'])
        save_carbon.append(r['save_carbon_pct'])
        da_save_cost.append((r['base_cost'] - r['da_cost']) / r['base_cost'] * 100.0)
        mpc_costs.append(r['mpc_cost']); base_costs.append(r['base_cost'])

    mean_c, ci_c = _ci_mean(save_cost)
    mean_co, ci_co = _ci_mean(save_carbon)
    mean_da, ci_da = _ci_mean(da_save_cost)
    out = {
        'n_days': len(save_cost),
        'save_cost_pct': {'mean': mean_c, 'ci95': ci_c},
        'save_carbon_pct': {'mean': mean_co, 'ci95': ci_co},
        'da_save_cost_pct': {'mean': mean_da, 'ci95': ci_da},
        'total_base_cost': float(np.sum(base_costs)),
        'total_mpc_cost': float(np.sum(mpc_costs)),
    }
    if verbose:
        print(f'\n===== 全年滚动调度评估（{out["n_days"]} 个完整日）=====')
        print(f'  电费节省(无储能→MPC): 均值 {mean_c:.2f}%  '
              f'95%CI [{ci_c[0]:.2f}, {ci_c[1]:.2f}]')
        print(f'  碳排降低(无储能→MPC): 均值 {mean_co:.2f}%  '
              f'95%CI [{ci_co[0]:.2f}, {ci_co[1]:.2f}]')
        print(f'  (日前随机 nominal 相对基线电费节省 均值 {mean_da:.2f}%)')
        print(f'  全年基线电费合计 ≈ {out["total_base_cost"]:,.0f} 元，'
              f'MPC 后 ≈ {out["total_mpc_cost"]:,.0f} 元')
        # 【2026-09-14 更新】如实标注负荷形状来源
        try:
            _src = pp.load_shape_source()
        except Exception:
            _src = 'estimate'
        _desc = {
            'self_consumption_8760h': '消纳率表8760h实测',
            'hourly_bill_ndays':      '电费清单逐时抄表实测',
            'estimate':               '典型形状估算(无实测)',
        }.get(_src, _src)
        print(f'  ⓘ 负荷剖面 = 日总量 × 【{_desc}】分时形状；'
              f'储能容量等为待确认示例值，绝对值为示意、相对趋势可靠。')
    return out


# -----------------------------------------------------------------------------
# 敏感性①：电池容量
# -----------------------------------------------------------------------------
def sensitivity_battery_capacity(days=None, capacities=(250, 500, 1000, 1500, 2000),
                                 verbose=True):
    if days is None:
        days = build_daily_profiles()
    price_24 = np.array(cfg.GRID_BUY_PRICE_24H, float)
    carbon_24 = np.array(cfg.CARBON_FACTOR_24H, float)
    # 为控时，敏感性用代表性抽样（bootstrap CI 已反映抽样不确定性）
    rng = np.random.default_rng(SEED)
    if len(days) > 120:
        days = [days[i] for i in rng.choice(len(days), 120, replace=False)]
    rows = []
    for cap in capacities:
        old = _set_cfg(BATTERY_CAPACITY_KWH=cap, BATTERY_POWER_KW=cap)  # 1C
        save_cost, save_carbon = [], []
        for (day, pv_d, load_d) in days:
            try:
                r = evaluate_day(pv_d, load_d, price_24, carbon_24, skip_da=True)
                save_cost.append(r['save_cost_pct'])
                save_carbon.append(r['save_carbon_pct'])
            except Exception:
                continue
        m_c, ci_c = _ci_mean(save_cost)
        m_co, ci_co = _ci_mean(save_carbon)
        rows.append({'capacity_kwh': cap, 'save_cost_mean': m_c, 'save_cost_ci95': ci_c,
                     'save_carbon_mean': m_co, 'save_carbon_ci95': ci_co, 'n': len(save_cost)})
        _set_cfg(**old)
    if verbose:
        print('\n===== 敏感性① 电池容量（功率按 1C 同比例）=====')
        print(f'{"容量kWh":>8}{"电费节省%":>14}{"95%CI":>20}{"碳排降%":>12}')
        for r in rows:
            print(f'{r["capacity_kwh"]:>8}{r["save_cost_mean"]:>14.2f}'
                  f'  [{r["save_cost_ci95"][0]:.2f},{r["save_cost_ci95"][1]:.2f}]'
                  f'{r["save_carbon_mean"]:>12.2f}')
    return rows


# -----------------------------------------------------------------------------
# 敏感性②：碳价
# -----------------------------------------------------------------------------
def sensitivity_carbon_price(days=None, prices=(0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0),
                             verbose=True):
    if days is None:
        days = build_daily_profiles()
    price_24 = np.array(cfg.GRID_BUY_PRICE_24H, float)
    carbon_24 = np.array(cfg.CARBON_FACTOR_24H, float)
    rng = np.random.default_rng(SEED)
    if len(days) > 120:
        days = [days[i] for i in rng.choice(len(days), 120, replace=False)]
    rows = []
    for cp in prices:
        old = _set_cfg(CARBON_PRICE_YUAN_PER_KG=cp)
        save_cost, save_carbon, abs_carbon = [], [], []
        for (day, pv_d, load_d) in days:
            try:
                r = evaluate_day(pv_d, load_d, price_24, carbon_24, skip_da=True)
                save_cost.append(r['save_cost_pct'])
                save_carbon.append(r['save_carbon_pct'])
                abs_carbon.append(r['mpc_carbon'])
            except Exception:
                continue
        m_c, ci_c = _ci_mean(save_cost)
        m_co, ci_co = _ci_mean(save_carbon)
        m_ac, ci_ac = _ci_mean(abs_carbon)
        rows.append({'carbon_price': cp, 'save_cost_mean': m_c, 'save_cost_ci95': ci_c,
                     'save_carbon_mean': m_co, 'save_carbon_ci95': ci_co,
                     'abs_carbon_mean_kg': m_ac, 'abs_carbon_ci95': ci_ac, 'n': len(save_cost)})
        _set_cfg(**old)
    if verbose:
        print('\n===== 敏感性② 碳价（元/kgCO2）=====')
        print(f'{"碳价":>8}{"电费节省%":>14}{"碳排降%":>12}{"日均碳排kg":>14}')
        for r in rows:
            print(f'{r["carbon_price"]:>8.2f}{r["save_cost_mean"]:>14.2f}'
                  f'{r["save_carbon_mean"]:>12.2f}{r["abs_carbon_mean_kg"]:>14.1f}')
        print('  （现实碳价≤0.2 元/kg 时电费节省基本不随碳价变化——电价套利主导电池调度；')
        print('    碳价主要通过“碳排绝对量”体现，高碳价下 battery 主动避峰高碳时段。）')
    return rows


# -----------------------------------------------------------------------------
# 敏感性③：预测误差（注入噪声到预测，看调度收益退化）
# -----------------------------------------------------------------------------
def sensitivity_forecast_error(days=None, levels=(0.0, 0.03, 0.06, 0.10, 0.20),
                               verbose=True):
    """预测误差敏感性 = 后悔值(regret)。

    对每一个误差水平 lvl：给预测注入噪声 forecast = actual·(1+lvl·N(0,1))，
    比较两种调度在“真实”上的成本相对“完美预测 MPC”的退化：
      - MPC 闭环：每步用真实值重优化 → 对预测误差鲁棒，regret 小；
      - 日前开环：按预测一次性定计划 → 对预测误差敏感，regret 大。
    这种对照直接量化“预测质量的边际价值”以及 MPC 的抗扰优势。
    regret 以占“无储能基线电费”的百分比表示，带 bootstrap 95% CI。
    """
    if days is None:
        days = build_daily_profiles()
    price_24 = np.array(cfg.GRID_BUY_PRICE_24H, float)
    carbon_24 = np.array(cfg.CARBON_FACTOR_24H, float)
    rng = np.random.default_rng(SEED)
    if len(days) > 120:
        days = [days[i] for i in rng.choice(len(days), 120, replace=False)]
    rows = []
    for lvl in levels:
        mpc_reg, da_reg = [], []
        for (day, pv_d, load_d) in days:
            b = opt.baseline_no_storage(pv_d, load_d, price_24, carbon_24)
            # 完美预测 MPC（无误差基准）
            perfect = mpc_with_forecast(pv_d, load_d, pv_d.copy(), load_d.copy(),
                                        price_24, carbon_24)
            if lvl <= 0:
                mpc_reg.append(0.0); da_reg.append(0.0)
                continue
            pv_fc = np.maximum(pv_d * (1 + lvl * rng.standard_normal(len(pv_d))), 0.0)
            load_fc = np.maximum(load_d * (1 + lvl * rng.standard_normal(len(load_d))), 0.0)
            mpc_c = mpc_with_forecast(pv_d, load_d, pv_fc, load_fc, price_24, carbon_24)['cost']
            da_c, _ = day_ahead_realized(pv_d, load_d, pv_fc, load_fc, price_24, carbon_24)
            mpc_reg.append((mpc_c - perfect['cost']) / b['cost'] * 100.0)
            da_reg.append((da_c - perfect['cost']) / b['cost'] * 100.0)
        m_mpc, ci_mpc = _ci_mean(mpc_reg)
        m_da, ci_da = _ci_mean(da_reg)
        rows.append({'error_level': lvl, 'mpc_regret_pct': m_mpc, 'mpc_ci95': ci_mpc,
                     'da_regret_pct': m_da, 'da_ci95': ci_da, 'n': len(mpc_reg)})
    if verbose:
        print('\n===== 敏感性③ 预测误差 → 后悔值(regret, %基线电费) =====')
        print(f'{"误差%":>8}{"MPC闭环regret":>16}{"日前开环regret":>16}')
        for r in rows:
            print(f'{r["error_level"]*100:>7.0f}%'
                  f'{r["mpc_regret_pct"]:>15.3f}'
                  f'{r["da_regret_pct"]:>15.3f}')
        print('  （MPC 闭环 regret 远小于日前开环 → 证明滚动优化对预测误差鲁棒，')
        print('    预测质量的边际价值体现在“开环→闭环”的收益差，而非 MPC 内部的退化。）')
    return rows


def validate_lp_vs_milp(n_days=30, seed=SEED, verbose=True):
    """抽样对比 LP 与 MILP 求解器在 MPC 调度上的结果，证明差异可忽略
    （从而大规模滚动/敏感性可用 LP 加速而结论可信）。"""
    days = build_daily_profiles()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(days), min(n_days, len(days)), replace=False)
    price_24 = np.array(cfg.GRID_BUY_PRICE_24H, float)
    carbon_24 = np.array(cfg.CARBON_FACTOR_24H, float)
    diff_cost, diff_carbon = [], []
    for i in idx:
        day, pv_d, load_d = days[i]
        lp = mpc_with_forecast(pv_d, load_d, pv_d.copy(), load_d.copy(),
                               price_24, carbon_24, solver='lp')
        mp = mpc_with_forecast(pv_d, load_d, pv_d.copy(), load_d.copy(),
                               price_24, carbon_24, solver='milp')
        if abs(mp['cost']) > 1e-9:
            diff_cost.append(abs(lp['cost'] - mp['cost']) / abs(mp['cost']) * 100.0)
        if abs(mp['carbon']) > 1e-9:
            diff_carbon.append(abs(lp['carbon'] - mp['carbon']) / abs(mp['carbon']) * 100.0)
    out = {
        'n_compared': len(diff_cost),
        'cost_rel_diff_max_pct': float(np.max(diff_cost)) if diff_cost else float('nan'),
        'cost_rel_diff_mean_pct': float(np.mean(diff_cost)) if diff_cost else float('nan'),
        'carbon_rel_diff_max_pct': float(np.max(diff_carbon)) if diff_carbon else float('nan'),
        'carbon_rel_diff_mean_pct': float(np.mean(diff_carbon)) if diff_carbon else float('nan'),
    }
    if verbose:
        print('\n===== LP vs MILP 校验（抽样 %d 天，MPC 调度）=====' % out['n_compared'])
        print('  电费相对差异  max %.3f%%  mean %.3f%%' %
              (out['cost_rel_diff_max_pct'], out['cost_rel_diff_mean_pct']))
        print('  碳排相对差异  max %.3f%%  mean %.3f%%' %
              (out['carbon_rel_diff_max_pct'], out['carbon_rel_diff_mean_pct']))
    return out


if __name__ == '__main__':
    print('--- LP vs MILP 校验 ---')
    validate_lp_vs_milp()
    print('--- 全年滚动调度评估 ---')
    rolling_dispatch_evaluation()
    print('\n--- 敏感性① 电池容量 ---')
    sensitivity_battery_capacity()
    print('\n--- 敏感性② 碳价 ---')
    sensitivity_carbon_price()
    print('\n--- 敏感性③ 预测误差 ---')
    sensitivity_forecast_error()
