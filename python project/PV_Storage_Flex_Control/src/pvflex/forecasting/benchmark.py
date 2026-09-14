# =============================================================================
# forecasting/benchmark.py  ——  SOTA 预测基线对比 + 消融研究
# -----------------------------------------------------------------------------
# 对应你要的两件事之一：
#   "加 SOTA 预测基线 + 消融：LSTM/Transformer/N-BEATS 对比，证明你方法的增量"
#
# 【这个文件解决什么问题】
#   论文评审一定会问两件事，本文件就是用来回答它们的：
#     ① "你的方法和别人比怎么样？"  → 这就是【基线对比】
#     ② "你方法里的每个零件都有用吗？" → 这就是【消融实验(ablation)】
#
#   💡 消融是什么：像拆机器。把方法的某个零件拆掉再跑一遍，
#      如果成绩明显变差 → 说明这零件有用（贡献得到证明）；
#      如果没变化 → 说明它是多余的（应当从论文里删掉，别硬吹）。
#
# 【做法（与现有工程自洽、最公平）】
#   · 预测粒度 = 日总量（PV 日发电量 / 负荷日用电量），两者都是真实目标
#     （不用小时级的原因见 deep.py 文件头说明）
#   · 滚动起点(walk-forward)交叉验证：
#       训练窗口随时间【扩张】，每次预测下一天，每隔 REFIT_EVERY 天重训一次。
#       💡 为什么要滚动而不是简单切一刀：真实使用时，时间只能向前走、
#          数据只会越来越多。滚动验证模拟了这个过程，比"随机切分"更贴近实用，
#          也不会让模型"偷看未来数据"（数据泄漏）。
#   · 用 bootstrap 给出 RMSE/MAE/sMAPE 的 95% 置信区间
#     —— 不只是报一个数，而是报"这个数有多可信"（方法见 rolling_eval._ci_mean）
#   · 对比模型：季节朴素(SeasonalNaive) / XGBoost / 融合(本项目方法) /
#               LSTM / Transformer / N-BEATS 共 6 个
#   · 消融两组：
#       ① 本项目方法内部："只用朴素" vs "只用XGBoost" vs "两者融合"
#       ② 深度学习："带日历外生特征" vs "不带"（验证日历信息的价值）
#
# ⚠️ 系统级参数(光伏容量/储能容量/碳因子)仍为待确认示例值；
#    但本文件【只涉及预测部分】，不改那些参数，
#    所以这里的结论聚焦"预测精度"本身，不受储能容量未知的影响。
# =============================================================================

import os
import logging
import numpy as np

import pvflex.config as cfg
import pvflex.data.loaders as ld
import pvflex.forecasting.deep as deep

logger = logging.getLogger(__name__)

LOOKBACK = 14          # 回看窗口天数
REFIT_EVERY = 14       # 每隔多少天重训一次
TEST_RATIO = 0.30      # 最后 30% 当测试（覆盖全年后期，含季节波动）
N_BOOT = 1000          # bootstrap 次数（CI）
SEED = 0


# -----------------------------------------------------------------------------
# 数据：构造日总量 + 日历外生 + 滞后特征
# -----------------------------------------------------------------------------
def _cyclic(vals, period):
    v = np.asarray(vals, float)
    return np.sin(2 * np.pi * v / period), np.cos(2 * np.pi * v / period)


def build_daily_dataset(target='pv'):
    """
    返回对齐后的日总量序列及特征。

    返回 dict:
      dates : 可预测日(target 日)的日期数组（长度 K = n - LOOKBACK）
      target: 日总量数组（K,）
      exo   : 日历外生特征 (K, 7)  —— 给深度学习用
      feat : 完整特征 (K, 14) = exo + 7日滞后 —— 给 XGBoost 用
      months: 每月标签 (K,)
    """
    pv = ld.load_pv_generation()
    load = ld.load_load_series()
    if pv.empty or load.empty:
        raise RuntimeError('数据未读取成功，无法做预测基准。')

    pv_daily = pv.resample('D').sum()
    # 负荷本就是日合计；对齐到与 pv_daily 相同的日期
    dates = pv_daily.index
    if target == 'pv':
        y = pv_daily.reindex(dates).to_numpy(float)
    else:
        y = load.reindex(dates).to_numpy(float)

    y = np.asarray(y, float)
    # 丢弃含 NaN/非正的日（脏数据/缺口）—— 与现有清洗口径一致，仅保留有效日
    valid = np.isfinite(y) & (y > 0)
    dates = dates[valid]
    y = y[valid]

    months = np.array([d.month for d in dates])
    doy = np.array([d.timetuple().tm_yday for d in dates])

    m_sin, m_cos = _cyclic(np.array([(d.month - 1) for d in dates]), 12)
    dow = np.array([d.dayofweek for d in dates])
    dow_sin, dow_cos = _cyclic(dow, 7)
    is_weekend = (dow >= 5).astype(float)
    doy_sin, doy_cos = _cyclic(doy, 365)
    exo = np.stack([m_sin, m_cos, dow_sin, dow_cos, is_weekend, doy_sin, doy_cos],
                   axis=1)                                   # (K0, 7)

    # 7 日滞后（按日总量本身），用全局均值/标准差标准化
    lags = np.full((len(y), 7), np.nan)
    for i in range(1, 8):
        lags[i:, i - 1] = y[:-i]
    mu, sd = np.nanmean(y), np.nanstd(y)
    sd = sd if sd > 1e-9 else 1.0
    lags_norm = (lags - mu) / sd
    feat = np.concatenate([exo, lags_norm], axis=1)          # (K0, 14)

    # 构造“序列”(窗口)数组：seqs[k] 预测第 k+LOOKBACK 天的 target
    n = len(y)
    K = n - LOOKBACK
    seqs = np.stack([y[k:k + LOOKBACK] for k in range(K)], axis=0)   # (K, L)
    tgt = y[LOOKBACK:]                                            # (K,)
    exo_k = exo[LOOKBACK:]
    feat_k = feat[LOOKBACK:]
    months_k = months[LOOKBACK:]
    dates_k = dates[LOOKBACK:]

    return {'dates': dates_k, 'target': tgt, 'seqs': seqs,
            'exo': exo_k, 'feat': feat_k, 'months': months_k,
            'y_all_mean': float(mu), 'y_all_std': float(sd)}


# -----------------------------------------------------------------------------
# 评估指标 + bootstrap 置信区间
# -----------------------------------------------------------------------------
def _metrics(actual, pred, capacity_ref=None):
    a = np.asarray(actual, float); p = np.asarray(pred, float)
    err = a - p
    rmse = float(np.sqrt(np.nanmean(err ** 2)))
    mae = float(np.nanmean(np.abs(err)))
    denom = np.abs(a) + np.abs(p)
    smape = float(100 * np.nanmean(2 * np.abs(err) / np.where(denom < 1e-9, 1e-9, denom)))
    rmse_pct = (rmse / capacity_ref * 100.0) if (capacity_ref and capacity_ref > 0) else np.nan
    return {'RMSE': rmse, 'MAE': mae, 'sMAPE': smape, 'RMSE_PCT': rmse_pct}


def _bootstrap_ci(actual, pred, n_boot=N_BOOT, seed=SEED):
    a = np.asarray(actual, float); p = np.asarray(pred, float)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(a))
    store = {'RMSE': [], 'MAE': [], 'sMAPE': []}
    if len(a) < 2:
        return {k: (np.nan, np.nan) for k in store}
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        err = a[s] - p[s]
        store['RMSE'].append(np.sqrt(np.mean(err ** 2)))
        store['MAE'].append(np.mean(np.abs(err)))
        denom = np.abs(a[s]) + np.abs(p[s])
        store['sMAPE'].append(100 * np.mean(2 * np.abs(err) / np.where(denom < 1e-9, 1e-9, denom)))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in store.items()}


# -----------------------------------------------------------------------------
# 模型适配器（统一 fit/predict 接口，供 walk-forward 调用）
# -----------------------------------------------------------------------------
class _SeasonalNaive:
    def fit(self, seqs, exo, tgt, feat, months):
        import pandas as pd
        s = pd.Series(np.asarray(tgt, float), index=np.asarray(months))
        self.table = s.groupby(s.index).mean()
        self.gmean = float(np.mean(tgt)) if len(tgt) > 0 else 0.0

    def predict(self, seq, exo, feat, month):
        return float(self.table.get(int(month), self.gmean))


class _XGB:
    def fit(self, seqs, exo, tgt, feat, months):
        try:
            import xgboost as xgb
        except Exception:
            raise RuntimeError('XGBoost 未安装，无法跑 XGBoost 基线。')
        X = np.asarray(feat, float)
        y = np.asarray(tgt, float)
        mask = ~np.isnan(X).any(axis=1)
        self.model = xgb.XGBRegressor(n_estimators=120, max_depth=4,
                                      learning_rate=0.1, verbosity=0)
        self.model.fit(X[mask], y[mask])

    def predict(self, seq, exo, feat, month):
        X = np.asarray(feat, float).reshape(1, -1)
        # 测试期滞后特征均有效；极端情况下用 0 兜底避免 NaN
        X = np.nan_to_num(X, nan=0.0)
        return float(self.model.predict(X)[0])


class _Deep:
    def __init__(self, kind, use_exo=True, exo_dim=7):
        self.kind = kind; self.use_exo = use_exo; self.exo_dim = exo_dim

    def fit(self, seqs, exo, tgt, feat, months):
        exos = np.asarray(exo, float) if self.use_exo else None
        self.model = deep.DeepForecaster(self.kind, lookback=LOOKBACK,
                                         exo_dim=self.exo_dim, use_exo=self.use_exo,
                                         epochs=60, seed=SEED)
        self.model.fit(np.asarray(seqs, float), exos, np.asarray(tgt, float))

    def predict(self, seq, exo, feat, month):
        exos = np.asarray(exo, float).reshape(1, -1) if self.use_exo else None
        return float(self.model.predict(np.asarray(seq, float).reshape(1, -1), exos)[0])


# -----------------------------------------------------------------------------
# 滚动起点(walk-forward)交叉验证
# -----------------------------------------------------------------------------
def walk_forward(data, model, refit_every=REFIT_EVERY, test_ratio=TEST_RATIO,
                 verbose=False):
    """
    对单个模型做 walk-forward 评估，返回 (actuals, preds) 两个等长数组。
    """
    seqs = data['seqs']; exo = data['exo']; feat = data['feat']
    tgt = data['target']; months = data['months']
    K = len(tgt)
    k_test_start = max(LOOKBACK, int(round(K * (1 - test_ratio))))
    if k_test_start >= K:
        k_test_start = max(0, K - 50)

    actuals, preds = [], []
    last_fit_k = -10**9
    m = None
    for k in range(k_test_start, K):
        if k - last_fit_k >= refit_every or m is None:
            # 重新实例化（保证每次重训是干净模型，避免状态污染）
            m = _fresh(model)
            m.fit(seqs[:k], exo[:k], tgt[:k], feat[:k], months[:k])
            last_fit_k = k
        p = m.predict(seqs[k], exo[k], feat[k], months[k])
        actuals.append(tgt[k]); preds.append(p)
    return np.array(actuals), np.array(preds)


def _fresh(spec):
    """根据 spec 生成一个全新的模型对象。spec 可以是类实例（克隆同类）或构造器。"""
    if isinstance(spec, _SeasonalNaive):
        return _SeasonalNaive()
    if isinstance(spec, _XGB):
        return _XGB()
    if isinstance(spec, _Deep):
        return _Deep(spec.kind, spec.use_exo, spec.exo_dim)
    if callable(spec):
        return spec()
    return spec


# -----------------------------------------------------------------------------
# 主入口：跑全部模型 + 消融，汇总指标与 CI
# -----------------------------------------------------------------------------
def run_forecast_benchmark(target='pv', verbose=True, save_fig=True):
    """
    对 PV 或负荷 做完整预测基准 + 消融。

    返回 dict：每个模型 {实际值, 预测值, 指标, 95%CI}，以及消融表。
    """
    data = build_daily_dataset(target)
    # ⚠️ 归一参考口径：申报书“月均RMSE<20%”针对“小时级功率(kW)”/额定容量(kW)。
    #    本基准在“日总量(kWh)”粒度，故 PV 用 额定×24h（单日最大可能发电量）作分母，
    #    保持“占装机”语义一致；负荷无额定容量，用日均量作归一参考。
    if target == 'pv':
        cap_ref = cfg.PV_RATED_KW * 24.0
    else:
        cap_ref = data['y_all_mean']

    specs = {
        'SeasonalNaive': _SeasonalNaive(),
        'XGBoost': _XGB(),
        'Ours(fusion)': 'fusion',                 # 特殊：0.5*naive+0.5*xgb
        'LSTM': _Deep('lstm', use_exo=True),
        'Transformer': _Deep('transformer', use_exo=True),
        'N-BEATS': _Deep('nbeats', use_exo=True),
        'LSTM(no-exo)': _Deep('lstm', use_exo=False, exo_dim=0),
        'Transformer(no-exo)': _Deep('transformer', use_exo=False, exo_dim=0),
        'N-BEATS(no-exo)': _Deep('nbeats', use_exo=False, exo_dim=0),
    }

    results = {}
    # 先跑朴素与 XGBoost（融合要用）
    preds_cache = {}
    for name, spec in specs.items():
        if name == 'Ours(fusion)':
            continue
        if verbose:
            print(f'  · 训练/评估 {name} ...')
        act, prd = walk_forward(data, spec, verbose=False)
        preds_cache[name] = prd
        met = _metrics(act, prd, capacity_ref=cap_ref)
        ci = _bootstrap_ci(act, prd)
        results[name] = {'actual': act, 'pred': prd, 'metrics': met, 'ci': ci,
                         'n_test': len(act)}

    # 融合 = 0.5*naive + 0.5*xgb（即现有“方面1”方法的日总量等价形式）
    sn = preds_cache['SeasonalNaive']; xg = preds_cache['XGBoost']
    fusion_pred = 0.5 * sn + 0.5 * xg
    act = results['SeasonalNaive']['actual']
    met = _metrics(act, fusion_pred, capacity_ref=cap_ref)
    ci = _bootstrap_ci(act, fusion_pred)
    results['Ours(fusion)'] = {'actual': act, 'pred': fusion_pred,
                               'metrics': met, 'ci': ci, 'n_test': len(act)}

    if verbose:
        _print_benchmark(results, target, cap_ref)

    if save_fig:
        _save_benchmark_fig(results, data, target)

    return {'target': target, 'results': results, 'cap_ref': cap_ref,
            'n_test': len(act)}


def _print_benchmark(results, target, cap_ref):
    ref_name = f'参考容量={cap_ref:.0f}kWh' if cap_ref else '参考=日均量'
    print(f'\n===== 预测基准对比（目标={target}，{ref_name}）=====')
    print(f'{"模型":<20}{"RMSE":>10}{"MAE":>10}{"sMAPE%":>10}{"RMSE%":>10}')
    order = ['SeasonalNaive', 'XGBoost', 'Ours(fusion)', 'LSTM', 'Transformer',
             'N-BEATS', 'LSTM(no-exo)', 'Transformer(no-exo)', 'N-BEATS(no-exo)']
    for name in order:
        if name not in results:
            continue
        m = results[name]['metrics']
        print(f'{name:<20}{m["RMSE"]:>10.2f}{m["MAE"]:>10.2f}'
              f'{m["sMAPE"]:>10.2f}{("%.2f" % m["RMSE_PCT"]) if not np.isnan(m["RMSE_PCT"]) else "  -":>10}')
    print('  （RMSE%/sMAPE 越小越好；PV 的 RMSE% 对应申报书“月均RMSE<20%”硬指标）')


def _save_benchmark_fig(results, data, target):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # 尽量用中文字体；没有就退回英文标签，避免缺字形警告
        try:
            import os as _os
            from matplotlib import font_manager
            _f = None
            for cand in ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']:
                if _os.path.exists(cand):
                    font_manager.fontManager.addfont(cand)
                    _f = _os.path.basename(cand).split('.')[0]
                    break
            if _f:
                plt.rcParams['font.sans-serif'] = [_f]
                plt.rcParams['axes.unicode_minus'] = False
                use_cn = True
            else:
                use_cn = False
        except Exception:
            use_cn = False
        out = os.path.join(cfg.FIG_OUT_DIR, 'benchmark')
        os.makedirs(out, exist_ok=True)
        dates = data['dates']
        for name in ['Ours(fusion)', 'LSTM', 'Transformer', 'N-BEATS']:
            if name not in results:
                continue
            r = results[name]
            n = len(r['actual'])
            take = min(60, n)
            a = r['actual'][-take:]; p = r['pred'][-take:]
            dd = dates[-take:]
            plt.figure(figsize=(10, 4))
            plt.plot(dd, a, 'k-', label=('实际' if use_cn else 'Actual'))
            plt.plot(dd, p, '--', label=name)
            plt.title(f'{target} forecast vs actual (last 60 test days): {name}')
            plt.xlabel('Date'); plt.ylabel(f'{target} daily total (kWh)')
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out, f'bench_{target}_{name}.png'), dpi=110)
            plt.close()
        logger.info('已存预测对比图：%s', out)
    except Exception as e:
        logger.warning('跳过存图：%s', e)


if __name__ == '__main__':
    for tgt in ['pv', 'load']:
        run_forecast_benchmark(tgt, verbose=True)
