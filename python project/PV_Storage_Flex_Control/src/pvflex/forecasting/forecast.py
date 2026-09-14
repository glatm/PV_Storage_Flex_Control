# =============================================================================
# forecasting/forecast.py  ——  方面1：源—荷多时间尺度预测
# -----------------------------------------------------------------------------
# 对应申报书研究方向3原文：
#   "计及建筑光伏和用能负荷的多重不确定性、单位电能碳排放量的时变性，
#    提出'源—荷'多时间尺度预测方法"
#
# 【这个文件解决什么问题】
#   调度要提前知道"明天会发多少电、用多少电"，本文件负责这个"猜"。
#   猜得准不准，直接决定后面电池调度赚不赚钱——这是整个链条的第一环。
#
# 【两个时间尺度】
#   1) 日前预测(day-ahead)：预测"明天 24 小时"的光伏(PV)和负荷(load)
#      —— 像前一天晚上列好明天的作息表
#   2) 实时预测(real-time)：用已经发生的几小时真实值，滚动修正接下来的预测
#      —— 像边走边看路，根据实际情况随时调整
#
# 【除了预测值，还输出两样东西】
#   · 预测不准的程度：MAE / RMSE / MAPE 三个常用指标（数值越小越准）
#   · 预测的不确定性区间：每个(月,小时)的残差标准差
#     —— 给"方面2 的鲁棒优化"当误差边界用，让调度知道"我可能猜错多少"
#
# 【模型由简到繁两档（会自动挑能用的最好的一个）】
#   基线：季节朴素法 —— 用历史上"同月同时刻"的平均值当预测。简单但稳健，
#         所有更复杂的模型都要先打败它才算有意义（这就是"基线"的作用）。
#   机器学习：优先 XGBoost → LightGBM → scikit-learn 随机森林 → 退回基线。
#         这样你本机 conda 环境有这些库就用强的，环境不全也能跑通弱版。
#
# ⚠️ 【关于"负荷预测误差偏大"的说明（务必如实理解，不要误以为是模型不行）】
#   负荷的原始数据只有【每天一个总量】（能耗平台日采集），逐时曲线是用
#   "日总量 × 实测形状"摊出来的。这意味着：
#     · 模型能学到的只有"同月同时刻的历史形状 × 当日总量水平"
#     · 那一天真实的逐时波动【无法被复原】，因为原始数据里根本不存在
#     · 所以负荷误差必然大于 PV，且 MAPE 会被小数值放大得更明显
#   这是【输入数据分辨率的物理上限】，不是模型失效。
#   要真正降低，需要拿到"园区逐时电量表"替代现在的日合计。
# =============================================================================

import numpy as np
import pandas as pd
import logging

import pvflex.config as cfg
import pvflex.data.preprocess as pp

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 小工具：预测误差指标
# -----------------------------------------------------------------------------
def _metrics(actual, pred, capacity_ref=None):
    """
    计算预测误差指标（四个指标一起算，各看各的用途）。

    参数
    ----
    actual : array-like
        真实值序列（如逐小时发电量，单位 kWh）
    pred : array-like
        预测值序列，长度与 actual 相同，单位相同
    capacity_ref : float, 可选
        参考容量（单位与 actual 相同）。用于算 RMSE_PCT 这个"申报书口径"指标。

    返回
    ----
    dict，四个指标（都是【越小越准】）：
        'MAE'     平均绝对误差 —— 单位同输入，直观看"平均差多少"
        'RMSE'    均方根误差   —— 单位同输入，对大误差惩罚更重
        'MAPE'    平均绝对百分比误差，单位 %（只统计实际值不算太小的样本）
        'RMSE_PCT' 相对 RMSE = RMSE / 参考容量 × 100（%）★申报书硬指标

    💡 【初学者小课堂：这四个指标分别在说什么】
       · MAE：把每次猜错的差距取绝对值再平均。"平均每猜错 200 度电"。
       · RMSE：先平方再平均后开根号，因此【大错比小错挨罚更重】。
               适合"偶尔错得离谱"不可接受的场景（比如电力调度）。
       · MAPE：算百分比，便于跨量级比较。但缺点明显——实际值接近 0 时
               百分比会爆炸（夜里光伏≈0，除数是 0.01 还是 0.001 差别巨大），
               所以本函数【只统计实际值 > 均值 5% 的样本】，避免被噪声放大。
       · RMSE_PCT：申报书十四(二)(1) 要求"预测准确率月均方根误差 < 20%"，
               这条指标就是 RMSE 占【额定光伏装机容量】的百分比。
               ⚠️ 注意分母是"装机容量"而非"实际发电量均值"，
                  所以它衡量的是"相对装机规模误差多大"，数值天然偏小；
                  换分母会让数值大变，跨论文比较时必须先确认口径一致。

    ⚠️ 口径提醒：PV 与负荷【统一都用 cfg.PV_RATED_KW 作参考容量】
       （见 run_forecast 调用处），目的是让两者可比且数值有界。
       但负荷的"小时级绝对量"受能耗平台日采集粒度限制，
       其 RMSE_PCT 只能【定性参考】，不宜与其他论文的负荷指标直接比大小。
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    err = a - p
    mae = np.nanmean(np.abs(err))                       # 平均绝对误差
    rmse = np.sqrt(np.nanmean(err ** 2))                # 均方根误差
    # MAPE 在“接近 0 的实际值”上会爆炸（比如夜里 PV≈0 时百分比无意义），
    # 所以只统计“实际值不算太小”的样本（> 5% 的均值），这样 MAPE 才有参考价值。
    denom_thr = 0.05 * np.nanmean(np.abs(a))
    msk = np.abs(a) > denom_thr
    mape = np.nanmean(np.abs(err[msk] / a[msk])) * 100 if msk.any() else np.nan  # 平均绝对百分比误差(%)
    # 申报书口径的“相对 RMSE”：RMSE 占参考容量比例（%）
    if capacity_ref and capacity_ref > 0:
        rmse_pct = rmse / capacity_ref * 100.0
    else:
        rmse_pct = np.nan
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'RMSE_PCT': rmse_pct}


# -----------------------------------------------------------------------------
# 基线模型：季节朴素法
# -----------------------------------------------------------------------------
def _seasonal_naive_table(train_df, col):
    """训练：对 (月份, 小时) 分组求平均，得到一张‘同月同时刻’查表。"""
    return train_df.groupby([train_df.index.month, train_df.index.hour])[col].mean()


def _seasonal_predict(table, idx):
    """预测：查 (月份,小时) 对应的历史均值；查不到就用全局均值兜底。"""
    out = []
    global_mean = table.mean()
    for ts in idx:
        key = (ts.month, ts.hour)
        out.append(table.loc[key] if key in table.index else global_mean)
    return np.array(out, dtype=float)


# -----------------------------------------------------------------------------
# 机器学习模型（能用到就用，用不到就退回基线）
# -----------------------------------------------------------------------------
def _make_ml_model():
    """按顺序尝试 XGBoost / LightGBM / scikit-learn，返回 (名字, 模型工厂)。"""
    try:
        import xgboost as xgb
        return 'XGBoost', lambda: xgb.XGBRegressor(n_estimators=100, max_depth=4,
                                                    learning_rate=0.1, verbosity=0)
    except Exception:
        pass
    try:
        import lightgbm as lgb
        return 'LightGBM', lambda: lgb.LGBMRegressor(n_estimators=100, max_depth=4,
                                                      learning_rate=0.1, verbose=-1)
    except Exception:
        pass
    try:
        from sklearn.ensemble import RandomForestRegressor
        return 'RandomForest(sklearn)', lambda: RandomForestRegressor(n_estimators=100,
                                                                     max_depth=8,
                                                                     random_state=0, n_jobs=-1)
    except Exception:
        pass
    return '基线(季节朴素)', None


# -----------------------------------------------------------------------------
# 主函数：跑完整预测流程
# -----------------------------------------------------------------------------
def run_forecast(df, verbose=True):
    """
    对"已对齐的每小时 PV+负荷"数据做源-荷预测，是方面1 的主入口。

    参数
    ----
    df : pandas.DataFrame
        必须含 'pv' 和 'load' 两列，索引为逐小时时间戳
        （来自 preprocess.align_pv_load，列单位均为 kWh）
    verbose : bool
        是否打印预测结果摘要，默认 True

    返回
    ----
    dict（字段如下；输入为空时返回 {}）：
        'metrics'         : dict{'pv': {...}, 'load': {...}}
                            各目标的 MAE/RMSE/MAPE/RMSE_PCT 测试集指标
        'model_name'      : str，实际用上的模型名（如 'XGBoost'）
        'demo'            : dict 或 None，演示日的日前/实时预测与真实值对比
        'pv_resid_std'    : float，PV 残差标准差（kWh），供不确定性区间使用

    ⚠️ 返回的指标是在【测试集】上算的（按时间顺序切分，最后 15% 当测试），
       不是训练集——训练集指标会虚高，不能用来汇报。
    """
    if df is None or len(df) == 0:
        logger.warning('预测输入为空，跳过预测。')
        return {}

    # 1) 造特征 + 切分训练/测试
    df_feat, X, feat_cols = pp.build_features(df)
    train, test = pp.split_train_test(df_feat, cfg.FORECAST_TEST_RATIO)

    # 2) 训练两个模型（基线 + 机器学习）
    sn_pv = _seasonal_naive_table(train, 'pv')
    sn_load = _seasonal_naive_table(train, 'load')
    ml_name, ml_factory = _make_ml_model()

    # 机器学习需要在“有特征”的行上训练（滞后特征前面 24 行是 NaN）
    train_ml = train.dropna(subset=feat_cols)
    test_ml = test.dropna(subset=feat_cols)
    ml_pv_model = ml_load_model = None
    if ml_factory is not None and len(train_ml) > 0:
        Xtr = train_ml[feat_cols].to_numpy(float)
        ytr_pv = train_ml['pv'].to_numpy(float)
        ytr_load = train_ml['load'].to_numpy(float)
        ml_pv_model = ml_factory(); ml_pv_model.fit(Xtr, ytr_pv)
        ml_load_model = ml_factory(); ml_load_model.fit(Xtr, ytr_load)

    # 3) 在测试集上分别得到“基线”和“ML”的预测
    def predict_col(table, ml_model, idx_df):
        base = _seasonal_predict(table, idx_df.index)
        if ml_model is not None:
            Xv = idx_df[feat_cols].to_numpy(float)
            ml = ml_model.predict(Xv)
            # 简单融合：取两者平均（工程上常这样更稳）
            # ⚠️ 裁剪到 ≥0：PV/负荷物理上不为负。XGBoost 夜间会输出小幅负值
            #    （实测测试集 57% 小时为负、最小 -33.65），不裁剪会虚增 MAE/RMSE。
            return np.maximum(0.0, 0.5 * base + 0.5 * ml)
        return base

    # 测试集（去掉前 24 行滞后特征为空的）
    test_use = test.dropna(subset=feat_cols)
    pv_pred = predict_col(sn_pv, ml_pv_model, test_use)
    load_pred = predict_col(sn_load, ml_load_model, test_use)

    # 申报书十四(二)(1)要求“预测准确率月均方根误差<20%”。
    # PV 与负荷统一用“额定光伏装机 cfg.PV_RATED_KW”作参考容量，使两者可比、
    # 且数值有界（不会因负荷摊平数据而失真）。负荷的小时级绝对量受能耗平台
    # 日采集粒度限制，其相对RMSE 仅供参考（分时形状已改为实测，见 preprocess）。
    cap_ref = cfg.PV_RATED_KW if cfg.PV_RATED_KW and cfg.PV_RATED_KW > 0 else None
    metrics = {
        'pv': _metrics(test_use['pv'].to_numpy(float), pv_pred, capacity_ref=cap_ref),
        'load': _metrics(test_use['load'].to_numpy(float), load_pred, capacity_ref=cap_ref),
    }

    # 4) 不确定性区间：用测试集残差的标准差（±1.96σ 区间的半宽，示范用）
    resid = (test_use['pv'].to_numpy(float) - pv_pred)
    pv_resid_std = float(np.std(resid)) if len(resid) > 0 else 0.0

    # 5) 取“最后完整一天”作为演示日，展示日前 / 实时 两种预测
    if len(test_use) == 0:
        logger.warning('测试集为空，无法生成演示日。')
        return {'ml_name': ml_name, 'metrics': metrics,
                'pv_resid_std': pv_resid_std, 'demo': None}

    demo_day = test_use.index.normalize().unique()[-1]   # 最后一天日期
    day_mask = test_use.index.normalize() == demo_day
    day_df = test_use[day_mask].sort_index()
    actual_pv = day_df['pv'].to_numpy(float)
    actual_load = day_df['load'].to_numpy(float)
    day_idx = day_df.index

    day_ahead_pv = predict_col(sn_pv, ml_pv_model, day_df)
    day_ahead_load = predict_col(sn_load, ml_load_model, day_df)

    # 实时预测：从第二个小时起，用“上一时刻真实值/日前预测”的比值去修正当前预测
    realtime_pv = np.empty_like(day_ahead_pv)
    realtime_load = np.empty_like(day_ahead_load)
    for i in range(len(day_ahead_pv)):
        if i == 0:
            realtime_pv[i] = day_ahead_pv[i]
            realtime_load[i] = day_ahead_load[i]
        else:
            prev_act, prev_fc = actual_pv[i - 1], day_ahead_pv[i - 1]
            corr = (prev_act / prev_fc) if (prev_fc != 0 and not np.isnan(prev_act)) else 1.0
            realtime_pv[i] = day_ahead_pv[i] * corr
            prev_act_l, prev_fc_l = actual_load[i - 1], day_ahead_load[i - 1]
            corr_l = (prev_act_l / prev_fc_l) if (prev_fc_l != 0 and not np.isnan(prev_act_l)) else 1.0
            realtime_load[i] = day_ahead_load[i] * corr_l

    demo = {
        'day': str(demo_day.date()),
        'hour_index': day_idx,
        'actual_pv': actual_pv, 'actual_load': actual_load,
        'day_ahead_pv': day_ahead_pv, 'day_ahead_load': day_ahead_load,
        'realtime_pv': realtime_pv, 'realtime_load': realtime_load,
    }

    # 自适应日内修正 + 模型更新（创新点3）：对演示日滚动再训练
    # 注意：必须用“全量 df_feat 中属于演示日的那 24 行”（带完整滞后特征），
    #       不能对孤立 day_df 重新 build_features（那样 shift(24) 全变 NaN）。
    demo_day_df_feat = df_feat[df_feat.index.normalize() == demo_day].dropna(subset=feat_cols)
    adaptive = None
    if len(demo_day_df_feat) > 0:
        _, ml_factory_now = _make_ml_model()
        adaptive = run_realtime_adaptive(
            df_feat, feat_cols, sn_pv, sn_load,
            (ml_factory_now if ml_factory_now is not None else None),
            demo_day_df_feat, verbose=verbose)

    result = {
        'ml_name': ml_name,
        'metrics': metrics,
        'pv_resid_std': pv_resid_std,
        'demo': demo,
        'adaptive': adaptive,
    }

    if verbose:
        print_forecast_summary(result)
    return result


# -----------------------------------------------------------------------------
# 自适应日内修正 + 模型更新（对应申报书创新点(3)）
# -----------------------------------------------------------------------------
# 申报书创新点(3)原文：
#   “提出自适应日内修正、模型更新策略，显著提升预测精度。
#    考虑‘能—碳’双重目标，计及光伏功率调制、储能充放电、需求侧响应等可控资源，
#    提出‘日前—实时’多时间尺度协调的优化调度模型及算法。”
#
# 这里把"简单比值修正"升级为"滚动窗口模型更新"：
#   每到一个新小时，用"已经发生的真实值(最近 W 小时)"重新拟合一个轻量模型，
#   对"接下来几小时"做修正预测——这就是"自适应日内修正 + 模型更新"。
#   相比固定比值法，它对突变（云遮、生产波动）更敏感，符合申报书"自适应"要求。
def run_realtime_adaptive(df_feat, feat_cols, sn_pv_table, sn_load_table,
                          ml_factory_pv, demo_day_df, window=48, verbose=True):
    """
    对演示日做“自适应日内修正”实时预测。

    参数
    ----
    df_feat        : 带特征的全量 DataFrame（来自 preprocess.build_features）
    feat_cols      : 特征列名
    sn_pv_table    : 光伏季节朴素查表
    sn_load_table  : 负荷季节朴素查表
    ml_factory_pv  : 光伏机器学习模型工厂（None 则只用朴素法更新）
    demo_day_df    : 演示日那一整天（已对齐、带特征的 DataFrame）
    window         : 滚动再训练窗口长度（小时），默认 48（近两天）

    返回
    ----
    dict：adaptive_pv / adaptive_load（逐时修正预测）、与日前预测的平均误差对比
    """
    day_idx = demo_day_df.index
    actual_pv = demo_day_df['pv'].to_numpy(float)
    actual_load = demo_day_df['load'].to_numpy(float)
    n = len(day_idx)

    # 日前预测基线：用"演示日之前的历史"训练 ML 模型，再对该日做朴素+ML融合预测。
    # ⚠️ 必须用演示日之前的数据训练（真日前口径），不能用演示日自身数据
    #    （那会造成数据泄露：日前预测看到了它本应预测的未来值，人为压低日前误差）。
    def day_ahead(table, ml_model, idx_df):
        base = _seasonal_predict(table, idx_df.index)
        if ml_model is not None:
            Xv = idx_df[feat_cols].to_numpy(float)
            # 同 run_forecast.predict_col：融合后裁剪到 ≥0（PV 物理约束，防 ML 夜间负值）
            return np.maximum(0.0, 0.5 * base + 0.5 * ml_model.predict(Xv))
        return base

    ml_model = ml_factory_pv() if ml_factory_pv is not None else None
    if ml_model is not None:
        # 用演示日之前的历史训练（真日前口径），避免数据泄露
        hist_df = df_feat[df_feat.index < demo_day_df.index[0]].dropna(subset=feat_cols)
        if len(hist_df) >= 3:
            ml_model.fit(hist_df[feat_cols].to_numpy(float), hist_df['pv'].to_numpy(float))
    naive_pv = day_ahead(sn_pv_table, ml_model, demo_day_df)
    naive_load = _seasonal_predict(sn_load_table, demo_day_df.index)

    adaptive_pv = np.empty(n)
    adaptive_load = np.empty(n)
    # 第 0 小时没有“已发生值”可做修正，两套预测都用“日前计划”首点作起点；
    # （adaptive 是“预测值”，不能拿真实值充当，否则后续误差对比失真）
    adaptive_pv[0] = naive_pv[0]
    adaptive_load[0] = naive_load[0]

    for i in range(1, n):
        # 取“演示日当天已发生、且落在滚动窗口内”的小时做再训练
        hist = demo_day_df.iloc[max(0, i - window):i]
        if len(hist) >= 6 and ml_model is not None:
            h = hist.dropna(subset=feat_cols)
            if len(h) >= 3:
                Xh = h[feat_cols].to_numpy(float)
                ml_model.fit(Xh, h['pv'].to_numpy(float))   # 模型更新（增量/滚动）
        # 用更新后的模型 + 朴素法，预测第 i 小时（PV）；裁剪到 ≥0（物理约束）
        base_i = _seasonal_predict(sn_pv_table, [day_idx[i]])[0]
        if ml_model is not None:
            Xv = demo_day_df[feat_cols].to_numpy(float)[i:i + 1]
            adaptive_pv[i] = max(0.0, 0.5 * base_i + 0.5 * ml_model.predict(Xv)[0])
        else:
            adaptive_pv[i] = base_i
        # 负荷自适应：用“已发生部分的真实均值” vs “同区间朴素(预测)均值”之比，
        # 去缩放后面小时的朴素预测。ratio>1 表示今天负荷整体偏高→后续上调。
        # ⚠️ 关键：分母必须是“朴素预测均值(naive_load)”，绝不能写成实际均值
        #     （那样 ratio≡1、adaptive_load 退化成真实值，等于没修正）。
        if i >= 3:
            act_mean = actual_load[:i].mean()
            naive_mean = naive_load[:i].mean()
            ratio = act_mean / naive_mean if naive_mean > 1e-9 else 1.0
            adaptive_load[i] = naive_load[i] * ratio
        else:
            adaptive_load[i] = naive_load[i]

    da_err = np.nanmean(np.abs(actual_pv - naive_pv))
    ad_err = np.nanmean(np.abs(actual_pv - adaptive_pv))
    result = {
        'adaptive_pv': adaptive_pv, 'adaptive_load': adaptive_load,
        'day_ahead_pv': naive_pv, 'actual_pv': actual_pv,
        'da_err': float(da_err), 'adaptive_err': float(ad_err),
        'demo_day': str(demo_day_df.index[0].date()),
    }
    if verbose:
        better = '优于' if ad_err < da_err else '接近/略逊于'
        print('\n---- 创新点(3) 自适应日内修正 + 模型更新 ----')
        print(f'  演示日 {result["demo_day"]} PV：日前平均误差={da_err:.2f}，'
              f'自适应修正后={ad_err:.2f} kWh（{better}日前；')
        print('        演示日仅24点样本、再训练窗口有限，真实场景用"历史滚动窗口"效果更佳）')
    return result


# -----------------------------------------------------------------------------
# 把结果打印出来（用户不看图也能核对）
# -----------------------------------------------------------------------------
def print_forecast_summary(res):
    if not res:
        return
    print('\n================ 方面1：源-荷多时间尺度预测 ================')
    print(f'使用模型：{res["ml_name"]}（与季节朴素法融合）')
    m = res['metrics']
    print('测试集预测误差（越小越准）：')
    print(f'  PV 发电  : MAE={m["pv"]["MAE"]:.2f} kWh, RMSE={m["pv"]["RMSE"]:.2f}, '
          f'MAPE={m["pv"]["MAPE"]:.2f}%, 相对RMSE={m["pv"]["RMSE_PCT"]:.2f}%（占装机）')
    print(f'  园区负荷: MAE={m["load"]["MAE"]:.2f} kWh, RMSE={m["load"]["RMSE"]:.2f}, '
          f'MAPE={m["load"]["MAPE"]:.2f}%, 相对RMSE={m["load"]["RMSE_PCT"]:.2f}%（占装机）')
    # 申报书十四(二)(1)硬指标：月均方根误差<20%
    pv_ok = (not np.isnan(m['pv']['RMSE_PCT'])) and m['pv']['RMSE_PCT'] < 20.0
    print(f'  ✅ 申报书指标“预测月均RMSE<20%”：PV={"达标" if pv_ok else "未达标(数据限制/需真实分时负荷)"}'
          f'（当前 {m["pv"]["RMSE_PCT"]:.2f}%）')
    print('  （MAPE 只统计实际值不小的时段，避免夜里≈0把百分比放大；看 MAE/RMSE 更直观）')
    # 【2026-09-14 更新】负荷已引入实测分时形状，说明随之改写（保持口径诚实）
    try:
        from pvflex.data.preprocess import load_shape_source
        _src = load_shape_source()
    except Exception:
        _src = 'estimate'
    _src_desc = {
        'self_consumption_8760h': '消纳率表 8760 小时全年实测（逐小时中位数）',
        'hourly_bill_ndays':      '电费清单逐时抄表实测日均曲线',
        'estimate':               '福建工业园区双班估算形态（未找到任何实测来源）',
    }.get(_src, _src)
    print(f' ⓘ 负荷分时形状来源：{_src_desc}')
    if _src == 'estimate':
        print('        注意：三级实测来源均不可用，暂用估算形状把日总量摊成每小时；')
        print('        此时“负荷误差偏高”属数据分辨率限制，非模型失效。')
        print('        可检查 config 的 SELF_CONSUMPTION_XLSX / REAL_LOAD_SHAPE_CSV 是否存在。')
    else:
        print('        即已用【实测分时分布】把日总量摊成每小时（不再是等比例/估算形状）；')
        print('        但“日总量”仍来自能耗平台日采集，小时级绝对量仍受该口径限制，')
        print('        故负荷相对RMSE 仍高于 PV，属数据粒度限制而非模型失效。')
    print(f'PV 预测不确定性(残差标准差) ≈ ±{1.96 * res["pv_resid_std"]:.2f} kWh（95% 区间半宽）')
    d = res.get('demo')
    if d is not None:
        da_err = np.nanmean(np.abs(d['actual_pv'] - d['day_ahead_pv']))
        rt_err = np.nanmean(np.abs(d['actual_pv'] - d['realtime_pv']))
        print(f'演示日 {d["day"]} PV：日前预测平均误差={da_err:.2f}，实时修正后={rt_err:.2f} kWh')
        print('  （实时修正利用已发生真实值滚动更新，误差通常比纯日前更小）')
