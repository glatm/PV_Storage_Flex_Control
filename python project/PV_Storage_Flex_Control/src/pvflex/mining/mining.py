# =============================================================================
# mining/mining.py  ——  方面3：大数据挖掘 + 云边协同“光储直柔”智能调控框架
# -----------------------------------------------------------------------------
# 对应申报书研究方向3原文：
#   “提出基于云计算的‘光储直柔’智能调控系统框架，
#    实现基于大数据挖掘的调控策略优化”
#
# 【这个文件解决什么问题】
#   本文件做两件事：
#   A) 大数据挖掘：从历史数据里“挖”出有规律的东西，给优化/预测当养料——
#        · 典型日聚类：把 365 天压成 K 个“代表日”（大幅减少优化计算量）；
#        · 源-荷相关性：看光伏和负荷在不同时刻是“同向”还是“反向”；
#        · 异常检测：找出用电量反常的异常日（可能是设备异常或抄表错误）。
#   B) 云边框架仿真：演一遍“云(离线训练/挖掘/日前计划) → 边(实时执行/修正)”的闭环，
#       说明这套架构是怎么把前面的预测、优化串起来的。
#
# 【为什么"大数据挖掘"这件事在工程上真的有用？】
#   最直接的理由是省算力：一年 365 天，如果每天都做一次 24 小时优化，
#   要解 365 个优化问题；但全年天气其实就那么几种模式（晴天、多云、阴雨……），
#   归纳成 8 个典型日之后，解 8 次就能覆盖绝大多数情况。
#   这不是"为了论文好看"，而是云边架构里"云侧做重活"的真实做法 ——
#   把历史数据离线挖完，只把几个代表性策略下发到边缘设备执行。
#
# 【本文件在整个工程里的位置】
#   数据流的最后一站：前面 forecasting 给预测、optimization 给调度，
#   这里负责"归纳规律"+"串成一套可讲的架构"。
#   被 cli.py 的 run_direction3() 调用。
# =============================================================================

import numpy as np
import pandas as pd
import logging

import pvflex.config as cfg
import pvflex.optimization.dispatch as opt

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# A-1) 典型日聚类：把每天 24 小时曲线压成 K 个代表日
# -----------------------------------------------------------------------------
def mine_typical_days(pv_series, K=None):
    """对“每小时光伏”做 KMeans 聚类，得到 K 个典型日曲线。

    参数
    ----
    pv_series : pandas.Series
        每小时光伏发电量(kWh)，索引为时间戳。
    K : int 或 None
        要聚成几类（几个典型日）。填 None 则取 config.TYPICAL_DAY_K（默认 8）。

    返回
    ----
    dict，字段含义：
        'K'         : int，实际使用的聚类个数
        'centroids' : numpy 数组，形状 (K, 24)，K 条典型日曲线（每行 24 小时）
        'labels'    : numpy 数组，长度 = 天数，每个元素表示"那天属于第几类"
        'dates'     : DatetimeIndex，每行对应的日期（与 labels 一一对应）
        'sizes'     : dict，{簇编号: 该簇有多少天}
        'pivot'     : DataFrame，形状 (天数, 24)，整理好的"日期×小时"矩阵

    【KMeans 聚类是什么？用大白话解释】
        假设你手里有 365 张"一天 24 小时的发电曲线"图。KMeans 做的事就是：
        把它们按"长得像不像"分成 K 堆，每一堆选出一个"典型的代表"。
        算法思路很朴素，反复做两件事直到稳定：
          ① 分堆：每张图归到"离它最近的那个代表"那一堆；
          ② 更新代表：把那堆里所有图取平均，作为新的代表。
        "离得近"在数学上就是"24 个点逐个相减、平方、加起来"（欧氏距离）。

    【⚠️ 为什么第 45 行要 dropna()？】
        KMeans 要求每个样本的维度一样（都是 24 个数）。
        但有些天数据不全（比如只抄到 20 小时），不丢掉的话程序会报错。
        所以这里只保留"整天 24 小时齐全"的日子。
        副作用：可用天数会少于 365，日志里会提示。
    """
    K = K or cfg.TYPICAL_DAY_K
    s = pv_series.copy()                    # 复制一份，避免修改调用者传进来的数据
    s.index = pd.to_datetime(s.index)       # 保证索引是时间类型（防止传进来是字符串）
    # 透视：行=日期，列=0~23 点，值=该小时发电量
    hour = s.index.hour                     # 取出每个时间戳的"小时"部分，得到一列 0~23
    date = s.index.normalize()              # 把时间戳抹成当天 0 点，得到"日期"
    pivot = (s.to_frame('v')                # Series → DataFrame，列名叫 'v'
             .assign(hour=hour, date=date)  # 新增两列
             .pivot_table(index='date', columns='hour', values='v', aggfunc='mean'))
    # pivot_table 是"透视表"：把长条形的数据（一行一小时）摊成表格
    # （一行一天、一列一小时）。aggfunc='mean' 处理极少数同小时重复记录的情况。
    pivot = pivot.dropna()                      # 只保留完整 24 小时的日子
    if pivot.shape[0] < K:
        logger.warning('典型日聚类：有效天数(%d) < K(%d)，聚类可能不稳。', pivot.shape[0], K)
    # 上面这个警告很重要：如果只有 5 天数据却要分成 8 类，
    # 那聚类结果没有统计意义（纯粹是把 5 个点硬拆成 8 堆）。
    mat = pivot.to_numpy(dtype=float)       # 转成 numpy 矩阵，喂给 KMeans
    dates = pivot.index

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=10, random_state=0)
        # n_init=10：用 10 组不同的随机初始点各跑一次，取最好的结果
        #            （KMeans 对初始点敏感，多跑几次能避免掉进差的局部解）
        # random_state=0：固定随机种子，保证【每次运行结果完全一样】。
        #            做科研必须这样，否则论文里的数字每次重跑都变，没法复现。
        labels = km.fit_predict(mat)        # 每个样本属于第几类
        centroids = km.cluster_centers_     # 每一类的中心（就是"典型日曲线"）
    except Exception:
        # 没有 sklearn 时的极简兜底：按“日均发电量”分桶成 K 组
        # 为什么要有兜底？因为 sklearn 不是必装项（它体积大）。
        # 这里用"按日均发电量大小均分成 K 档"做个粗糙替代，
        # 精度差很多但保证程序不会因为缺库就整个跑不动。
        daily_mean = mat.mean(axis=1)       # 每天的平均发电量
        edges = np.percentile(daily_mean, np.linspace(0, 100, K + 1)[1:-1])
        # percentile 求分位点，例如把数据切成 K 等份需要的 K-1 个分界值。
        # np.linspace(0,100,K+1) 生成 K+1 个均匀的百分位（如 K=8 → 0,12.5,25...100），
        # [1:-1] 去掉头尾的 0 和 100，剩下 K-1 个分界值。
        labels = np.digitize(daily_mean, edges)   # 按分界值把每天归入对应的档
        centroids = np.array([mat[labels == k].mean(axis=0) for k in range(K)])
        # 上面这行：对每一档，把属于该档的所有天取平均，作为该档的"代表曲线"。

    # 统计每一类里有多少天。int() 是为了后面能安全地写入 JSON。
    sizes = {int(k): int(np.sum(labels == k)) for k in range(K)}
    return {'K': K, 'centroids': centroids, 'labels': labels,
            'dates': dates, 'sizes': sizes, 'pivot': pivot}


# -----------------------------------------------------------------------------
# A-2) 源-荷相关性：光伏和负荷在不同时刻是“同涨同跌”还是“此消彼长”
# -----------------------------------------------------------------------------
def analyze_correlation(df):
    """计算光伏与负荷的整体相关系数，以及“逐小时”相关系数。

    参数
    ----
    df : pandas.DataFrame
        必须含两列：'pv'（光伏发电量）和 'load'（园区用电量），索引为时间戳。

    返回
    ----
    dict，字段含义：
        'overall' : float，全时段的整体相关系数，范围 -1 ~ +1
        'by_hour' : list of (小时, 相关系数)，只包含能算出结果的小时

    【相关系数怎么读？】
        相关系数 r 在 -1 到 +1 之间，衡量两个量"一起变动的程度"：
          r ≈ +1 → 一个涨另一个也涨（同向）
          r ≈  0 → 没什么关系
          r ≈ -1 → 一个涨另一个反而跌（反向）
        直觉：若白天光伏高时负荷也高，相关性为正；
              若光伏高时负荷低（比如光伏高峰恰逢午休），则为负。
        这个指标在论文里的作用是：如果源（光伏）和荷（用电）天生同步，
        那么"就地消纳"容易做；如果反向，就必须靠储能来"搬运"电量。

    ⚠️ 本函数对夜里的小时会自动跳过，原因见下面的行内注释。
    """
    overall = float(np.corrcoef(df['pv'].to_numpy(float), df['load'].to_numpy(float))[0, 1])
    # np.corrcoef 返回一个 2×2 的相关系数矩阵（自己跟自己算就是 1）。
    # [0, 1] 取右上角那个数 = pv 和 load 的相关系数。
    # 逐小时：把每天同一小时的光伏和负荷排成两列，算该小时的相关系数。
    # 注意：若某小时某侧数据几乎不变（方差≈0，例如夜里 PV 恒为 0），
    # np.corrcoef 会除零得 nan；这种小时本就无相关可谈，直接跳过。
    #   —— 这个细节很容易被忽略：如果你不检查，得到一堆 nan 会让后续统计出错。
    pv = df['pv']; load = df['load']
    by_hour = []
    for h in range(24):                       # 0 点到 23 点逐个处理
        mask = pv.index.hour == h             # mask 是一串 True/False，标记"哪些行是这个小时"
        pv_h = pv[mask].to_numpy(float)       # 取出所有天在这个小时的光伏值
        load_h = load[mask].to_numpy(float)
        # 三重检查缺一不可：
        #   mask.sum() > 2  → 样本太少算相关系数没意义（至少 3 个点）
        #   np.std(..) > 1e-9 → 数据不能是常数（否则相关系数会除零）
        if mask.sum() > 2 and np.std(pv_h) > 1e-9 and np.std(load_h) > 1e-9:
            c = float(np.corrcoef(pv_h, load_h)[0, 1])
            by_hour.append((h, c))
    return {'overall': overall, 'by_hour': by_hour}


# -----------------------------------------------------------------------------
# A-3) 异常检测：日总用电量偏离“正常水平”太多的日子
# -----------------------------------------------------------------------------
def detect_anomalies(daily_load, z_thresh=3.0):
    """用“z-score”（偏离均值几个标准差）挑出用电量异常的日子。

    参数
    ----
    daily_load : pandas.Series
        每天园区总用电量(kWh)，索引为日期。
    z_thresh : float
        判定阈值，默认 3.0。数字越小抓出来的异常日越多（越敏感）。

    返回
    ----
    list of (日期字符串, z 值)
        例如 [('2025-06-15', 3.42), ...]；无异常时返回空列表 []。

    【z-score 是什么？】
        z = (今天的用电 - 平均用电) ÷ 标准差
        标准差可以理解成"日常波动的典型幅度"。
        所以 z = 3 的意思是"今天比平时多/少了 3 个'典型波动'那么多"，
        按正态分布经验，这种事发生的概率不到 1%，值得查一查。

    ⚠️ 这里有个已知的局限：均值和标准差本身会被异常值"带偏"
       （如果某天用电是平时的 10 倍，它会把均值拉高、标准差撑大，
        结果反而显得它自己不那么异常）。更稳健的做法是用中位数和 MAD，
       但本函数在 219 天的尺度上表现够用，暂不引入额外复杂度。
       如果以后异常日明显抓不准，可以考虑换成中位数法。
    """
    s = daily_load.astype(float)
    if len(s) < 3:
        return []          # 数据太少（少于 3 天）无法判断什么叫"异常"
    mu, sigma = s.mean(), s.std()
    if sigma == 0:
        return []          # 标准差为 0 说明每天一模一样，没有"异常"的概念
    z = (s - mu) / sigma
    anom = s.index[z.abs() > z_thresh]   # abs() 取绝对值 → 偏高偏低都算异常
    return [(str(d.date()), float(z.loc[d])) for d in anom]


# -----------------------------------------------------------------------------
# B) 云边协同调控仿真：演一遍“云→边”闭环
# -----------------------------------------------------------------------------
def simulate_cloud_edge(pv_series, load_series, price_24, carbon_24, K=None, prefetch=None):
    """模拟云边架构的一次完整运行（用一段有代表性的“典型日”来演示）。

    参数
    ----
    pv_series : pandas.Series
        每小时光伏发电量(kWh)，用来做典型日聚类。
    load_series : pandas.Series
        每天园区用电量(kWh)，用来推算演示日的负荷水平。
    price_24 : array-like
        长度 24 的电价(元/kWh)，对应 0~23 点。
    carbon_24 : array-like
        长度 24 的碳因子(kgCO2/kWh)。
    K : int 或 None
        典型日个数，None 则取 config.TYPICAL_DAY_K。
    prefetch : tuple 或 None
        可选。若调用者已经算好了 (da, rt, baseline, demo_date)，
        直接传进来复用，避免本函数再算一遍同样的优化（省时间）。

    返回
    ----
    dict，字段含义：
        'typical_day_K' : int，典型日个数
        'cluster_sizes' : dict，各簇天数
        'day_ahead'     : dict，日前随机规划的结果
        'real_time'     : dict，实时鲁棒调度的结果
        'baseline'      : dict，不装储能的对照结果
        'demo_date'     : str，演示日的日期

    ------ 云边架构是怎么分工的 ------
    云端（离线、算力强）：
      1) 大数据挖掘：把全年聚成 K 个典型日；
      2) 预测：用历史训练出源-荷预测模型（见 forecasting 模块）；
      3) 日前计划：对典型日做随机规划，得到“明天怎么调度电池”的计划。

    边端（现场、实时）：
      4) 实时执行：用真实值滚动修正（MPC），并对最坏情况留余量（鲁棒）。

    为什么这么分工？因为两类活的性质完全不同：
      云端能等（可以跑几小时），但边缘设备算力小、要秒级响应。
      所以把"需要大量历史数据和算力的活"放云端离线做，
      边缘只做"拿现成模型 + 快速求解"的轻活。这也是"云边协同"的核心思想。
    """
    # 云端挖掘
    mined = mine_typical_days(pv_series, K)

    # 若外部已算好 (da, rt, baseline, demo_date)，直接复用，避免重复计算
    if prefetch is not None:
        da, rt, baseline, demo_date = prefetch
        # 这是 Python 的"元组解包"：把一个四元素的元组一次性拆成四个变量。
    else:
        # 从聚类里挑一个“最接近某簇中心”的典型日，当作演示日
        daily_mean = mined['pivot'].mean(axis=1)
        demo_date = daily_mean.sort_values().index[len(daily_mean) // 2]
        pv_demo = mined['pivot'].loc[demo_date].to_numpy(float)
        # 负荷演示曲线：用 preprocess 的典型日内形状，从“该日总量”摊出来（演示用）。
        # ⚠️ 该形状自 2026-09-14 起为【实测】来源（消纳率表8760h / 电费清单逐时抄表），
        #    仅在实测数据全缺时才退回估算形态。见 pp.load_shape_source()。
        # ▲ 这里用的仍是"日总量 × 分时形状"的构造方式，不是真正的逐时实测负荷。
        #   这是数据分辨率决定的：能耗平台只给日总量。
        #   它能保证"日总量真实"，但不能反映"某天特别反常的日内波动"。
        from pvflex.data.preprocess import TYPICAL_LOAD_SHAPE
        shape = np.asarray(TYPICAL_LOAD_SHAPE, float)
        shape = shape / shape.sum()   # 与 to_hourly_load 保持一致：先归一化，保证日总量=均值
        # pv_demo * 0.0 是个"生成同形状全零数组"的小技巧（长度跟 pv_demo 一样），
        # 加上后面那一项后就得到"24 小时负荷曲线"，长度与 pv_demo 对齐。
        load_demo = pv_demo * 0.0 + np.mean(load_series) * shape

        # 云端：日前随机规划
        da = opt.run_day_ahead_stochastic(pv_demo, load_demo, price_24, carbon_24)
        # 边端：实时滚动（鲁棒）
        rt = opt.run_real_time_robust(pv_demo, load_demo, price_24, carbon_24)
        baseline = opt.baseline_no_storage(pv_demo, load_demo, price_24, carbon_24)

    return {
        'typical_day_K': mined['K'],
        'cluster_sizes': mined['sizes'],
        'day_ahead': da,
        'real_time': rt,
        'baseline': baseline,
        'demo_date': str(demo_date),
    }


def print_framework_summary(mined, corr, anomalies, ce):
    """把方面3 的三块挖掘结果 + 云边仿真结果打印成人能读的一段文字。

    参数
    ----
    mined : dict       mine_typical_days() 的结果
    corr : dict        analyze_correlation() 的结果
    anomalies : list   detect_anomalies() 的结果
    ce : dict          simulate_cloud_edge() 的结果

    返回
    ----
    None（只往屏幕打印）
    """
    print('\n================ 方面3：大数据挖掘 + 云边协同框架 ================')
    print('—— 大数据挖掘 ——')
    print(f'  典型日聚类：把历史压成 {mined["K"]} 个代表日，各簇天数={mined["sizes"]}')
    print(f'  源-荷相关性(整体)：r = {corr["overall"]:.3f}'
          f'（正=同涨同跌，负=此消彼长）')
    by_h = corr['by_hour']
    if by_h:
        # 列表推导式：把 [(小时, 系数), ...] 里的系数单独抽出来成为一个列表。
        rs = [c for _, c in by_h]
        print(f'  逐小时相关性范围：[{min(rs):.2f}, {max(rs):.2f}]，平均 {np.mean(rs):.2f}')
    print(f'  异常日检测：发现 {len(anomalies)} 个用电量反常日'
          + (f'，例如 {anomalies[0]}' if anomalies else '（无）'))
    # 上面这个 + 号拼接技巧：三元表达式 (A if 条件 else B) 生成一段文字再拼上去，
    # 这样"有异常就举例、没异常就说无"，比写 if-else 分支简洁。

    print('—— 云边协同调控仿真（演示日） ——')
    print(f'  云端挖掘{mined["K"]}典型日 → 日前随机规划 → 边端实时滚动(MPC+鲁棒)')
    print(f'  演示日 {ce["demo_date"]}：')
    base_c = ce['baseline']['cost']             # 不装储能的电费
    new_c = ce['day_ahead']['nominal']['cost']  # 优化后的电费
    e_pct = (base_c - new_c) / base_c * 100     # 省钱百分比
    print(f'    电费：基线 {base_c:.1f} 元 → 优化 {new_c:.1f} 元'
          f'（{"省" if e_pct >= 0 else "增"} {abs(e_pct):.1f}%）')
    # 注意这里用 abs(e_pct) 取绝对值：因为文字已经写了"省"或"增"，
    # 数字只用来表示幅度，再带个负号反而重复又别扭。
    print('  架构要点：云做“重活”(训练/挖掘/日前计划)，边做“快活”(秒级执行/修正)，')
    print('            两者通过“日前计划下发 + 实时反馈”闭环，实现能-碳主动调控。')
