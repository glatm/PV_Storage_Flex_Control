# =============================================================================
# cli.py  ——  命令行入口（整个工程的“总开关”）
# -----------------------------------------------------------------------------
# 【这个文件解决什么问题】
#   前面的 data / forecasting / optimization / mining 都是"零件"，
#   每个零件单独跑不出完整结论。这个文件负责把这些零件按正确顺序拼起来，
#   一口气跑完并打印结论。
#
# 【怎么用】在命令行进入本工程目录后运行下面任一命令：
#     python -m pvflex.cli                     # 跑默认的"全部"（方案B统计 + 方向3三方面）
#     python -m pvflex.cli --mode stats        # 只跑方案B：发电/负荷/消纳率/电费
#     python -m pvflex.cli --mode direction3   # 只跑方向3：预测/优化/云边框架
#     python -m pvflex.cli --mode netload      # 只跑净购电小时曲线还原（方案C）
#     python -m pvflex.cli --mode benchmark    # 只跑预测基线对比+消融（方面1扩展）
#     python -m pvflex.cli --mode rolling      # 只跑全年滚动调度+敏感性（方面2扩展）
#
# 【⚠️ 新手最容易踩的坑】
#   "全部"模式【不包含】netload / benchmark / rolling 三个重活。
#   因为它们分别要读几百个 PDF、训好几个深度学习模型、跑上千次优化，
#   跑一次要几分钟甚至十几分钟。所以设计成"想跑才手动指定"。
#   如果你只敲 `python -m pvflex.cli`，却奇怪"怎么没看到论文里那些结果"，
#   原因就在这里 —— 要另外加 --mode 参数。
#
# 【设计原则】
#   每个"做什么"都拆成独立的小函数（run_stats / run_direction3 ...），
#   你以后想改某一部分只动对应函数，不会互相干扰。
#
# 【本文件在整个工程里的位置】
#   最上层"调度员"：自己不读数据、不算模型，只负责调用别的模块 + 打印进度。
# =============================================================================

import argparse
import logging

import numpy as np
import pandas as pd

import pvflex
import pvflex.config as cfg
import pvflex.data.loaders as ld
import pvflex.data.preprocess as pp
import pvflex.analysis.stats as st
import pvflex.forecasting.forecast as fc
import pvflex.forecasting.benchmark as bm
import pvflex.optimization.dispatch as opt
import pvflex.optimization.rolling_eval as re
import pvflex.mining.mining as mng

# 上面这批 import 是 Python 的"导入语句"：
#   写法 `import a.b.c as x` 的意思是"把 a/b/c.py 这个文件拿来，起个短名字 x"。
#   之后代码里写 x.某个函数() 就等于调用那个文件里的函数。
#   这里用的短名字是社区习惯（cfg=配置、ld=loaders、pp=preprocess、
#   st=stats、fc=forecast、bm=benchmark、opt=optimization、re=rolling_eval、
#   mng=mining），换别人写的代码也常是这套缩写，看懂它们能省很多力气。
#   ⚠️ 注意 `re` 在这里指的是本项目的 rolling_eval，
#      会覆盖 Python 自带的同名标准库 re（正则表达式）；
#      本文件不用正则，所以无副作用，但你自己写新代码时别在同一个文件里
#      又 `import re` 又 `import rolling_eval as re`。


# 配置 logging：普通信息打印到屏幕，方便你看到进度
#   level=INFO    → 只显示"信息及以上"级别（DEBUG 这类啰嗦内容不显示）
#   format=...    → 每行开头打印时间、级别、来自哪个模块
#   datefmt='%H:%M:%S' → 时间只显示"时:分:秒"，不显示日期（看进度够用了）
# 这样你会看到类似： 10:23:45 [INFO] pvflex.analysis.stats: 已保存图片：...
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    datefmt='%H:%M:%S')
# 给本文件单独建一个"日志记录器"，名字叫 pvflex.cli。
# 之后用 logger.warning('...') 打印的内容会带上这个来源标签，方便定位。
logger = logging.getLogger('pvflex.cli')


# -----------------------------------------------------------------------------
# 流程 A：方案 B —— 项目数据统计分析
# -----------------------------------------------------------------------------
# 做什么：只做"算账"，不做预测也不做优化。
#          算清楚一年发多少电、用多少电、光伏自己用掉多少、电费大约多少。
# 输出：屏幕打印 4 段结论 + 5 张图（存到 figures/ 目录）。
def run_stats():
    # 打印一条分隔线（'=' 重复 60 次），让屏幕输出看起来分段清楚。
    print('=' * 60)
    print('  本园区 · 项目数据统计分析工具（方案 B）')
    print('=' * 60)

    # 第 1 步：加载数据
    #   \n 表示"换行"，让输出前面空一行，读起来不那么挤。
    print('\n[1/4] 正在读取数据...')
    pv = ld.load_pv_generation()      # 返回 Series：每小时光伏发电量(kWh)
    load = ld.load_load_series()      # 返回 Series：每天园区总用电量(kWh)
    print(f'  发电量数据点数 : {len(pv)}（应约 8760 小时）')
    print(f'  负荷数据天数   : {len(load)} 天')
    # f'...' 是 Python 的"格式化字符串"：花括号 { } 里的东西会被算出来后填进文字里。
    # 例如 {len(pv)} 会变成实际的数据点数。这是本项目打印信息的主力写法。
    #
    # 覆盖度提示：若负荷月覆盖不全（如被清洗排除了故障月），统计口径会偏短，显式提醒。
    if not load.empty:
        # resample('ME').count() = 按月切块，数每块里有几天有数据。
        months = load.resample('ME').count()
        months = months[months > 0]           # 丢掉"一天数据都没有"的月份
        span_days = (load.index.max() - load.index.min()).days + 1
        print(f'  负荷覆盖       : {len(months)} 个有数据月（'
              f'{load.index.min().date()} ~ {load.index.max().date()}，约 {span_days} 天跨度）')

    # 数据读不出来就提前收工，并提示用户去查路径（而不是继续跑出一堆 0）。
    if pv.empty or load.empty:
        print('\n⚠️ 数据未成功读取，请检查 config.py 里的路径是否正确。')
        return

    # 第 2 步：各项分析（每个分析都返回一个 dict，里面装着算出来的数字）
    print('\n[2/4] 正在分析...')
    pv_res = st.analyze_pv(pv)                 # 光伏：年发电、日均、月发电、容量系数
    load_res = st.analyze_load(load)           # 负荷：年用电、日均、峰谷、日负荷率
    pv_daily = pv.resample('D').sum()          # 把 PV 从"每小时"汇总成"每天"
    cons_res = st.calc_consumption_rate(pv_daily, load)   # 消纳率
    # ⚠️ 口径一致：电费基数用"消纳同口径的期内总用电"，而非全年 load。
    #    否则"全年电费"配"部分期自消纳量"→ 节省占电费比 ≠ 消纳率（被低估）。
    #    同口径后 节省/电费 ≡ 自消纳率（审计验证：0.186 = 0.186）。
    #
    # 什么是"口径"？用大白话说就是"你比的两个数是不是在同一个范围内算的"。
    # 举例：光伏只有 219 天有数据，但用电量有 365 天。
    #   若拿"219 天的光伏节省"去除"365 天的全年电费"，比例自然被摊薄、看着很小，
    #   但这不是因为光伏效果差，而是两个数的天数对不上（口径不一致）。
    #   正确做法：分子分母都用"219 天"这个共同范围算。
    cost_res = st.calc_cost(cons_res['期内总用电量_kWh'], cons_res['期内被消纳发电量_kWh'])

    # 第 3 步：打印结论
    print('\n[3/4] 分析结论：')
    st.print_pv_summary(pv_res)                 # 光伏段
    st.print_load_summary(load_res)             # 负荷段
    st.print_consumption_summary(cons_res)      # 消纳率段
    st.print_cost_summary(cost_res)             # 电费段

    # 第 4 步：画图
    print('\n[4/4] 正在生成图表...')
    st.make_figures(pv, load, pv_res, load_res, cons_res)
    print('\n✅ 全部完成。图表已保存到：', cfg.FIG_OUT_DIR)


# -----------------------------------------------------------------------------
# 流程 B：研究方向3 —— 多时间尺度能-碳指标智能调控（三个方面）
# -----------------------------------------------------------------------------
# 做什么：预测（方面1）→ 优化调度（方面2）→ 数据挖掘+云边框架（方面3）。
# 输出：三个方面各自的数值结论 + 3 张示意图（存到 figures/direction3/）。
def _pick_representative_day(pv_hourly):
    """挑"日均发电量处于中位数"的那一天，作为优化/调控的演示日。

    参数
    ----
    pv_hourly : Series
        每小时光伏发电量(kWh)，索引为时间戳。

    返回
    ----
    Timestamp
        被选中的那一天的日期（0 点）。

    为什么叫"代表日"？
        一年 365 天不可能每天都单独做一遍优化演示。挑一天"不好不坏、
        最接近全年中位水平"的日子来演示，结论最不容易被极端天气带偏。
        这比随手挑今天更有说服力。
    """
    # 只在"完整 24 小时"的日中选代表日，避免不完整日导致优化器维度不匹配。
    # （优化器要求恰好 24 个数，代表一天 24 小时；缺小时的天会少几项，
    #   直接喂进去会报错或算错。）
    daily_full = pv_hourly.resample('D').sum()      # 每天总发电量
    counts = pv_hourly.resample('D').count()        # 每天有几个小时的数据
    daily = daily_full[counts == 24].dropna()       # 只留"整天 24 小时齐全"的
    if len(daily) == 0:
        daily = daily_full  # 极端退化：无完整日时退回原始逻辑（宁可不准也别崩）
    # sort_values() 从小到大排队，取正中间那个 = 中位数所在的那天。
    rep = daily.sort_values().index[len(daily) // 2]
    return rep


def run_direction3():
    print('#' * 60)
    print('# 研究方向3：多时间尺度“能—碳”指标智能调控 —— 总运行')
    print('#' * 60)

    # 0) 加载并预处理数据
    print('\n[0] 加载项目数据 ...')
    pv = ld.load_pv_generation()            # 每小时光伏发电量
    load_daily = ld.load_load_series()      # 每天园区用电量
    if pv.empty or load_daily.empty:
        print('\n⚠️ 数据未成功读取，无法运行方向3。')
        return
    # 把"日总量"摊成"每小时"，再把光伏和负荷按时间对齐。
    #   摊分用的 24 小时形状来自 config/preprocess（实测优先，见 preprocess.py 说明）。
    load_hourly = pp.to_hourly_load(load_daily)
    df = pp.align_pv_load(pv, load_hourly)
    print(f'    对齐后共有 {len(df)} 个小时（'
          f'{df.index.min().date()} ~ {df.index.max().date()}）')

    # 转成 numpy 数组，因为下面传给优化器的是"纯数字"，不需要时间标签。
    price_24 = np.array(cfg.GRID_BUY_PRICE_24H, float)      # 24 个电价(元/kWh)
    carbon_24 = np.array(cfg.CARBON_FACTOR_24H, float)      # 24 个碳因子(kgCO2/kWh)

    rep_day = _pick_representative_day(df['pv'])
    # 选出代表日那一天的 24 行数据（normalize() 把时间戳抹成当天 0 点再比较）。
    mask = df.index.normalize() == rep_day
    pv_day = df['pv'][mask].to_numpy(float)       # 代表日 24 小时光伏(kWh)
    load_day = df['load'][mask].to_numpy(float)   # 代表日 24 小时负荷(kWh)
    print(f'    代表演示日：{rep_day.date()}')

    # 申报书十一(一)2：直流电压等级序列（48V/700V 双母线，与 Simulink 架构一致）
    print(f'    直流电压等级序列：{cfg.DC_VOLTAGE_LEVELS_V} V '
          f'（高压母线 {cfg.DC_BUS_VOLTAGE_V}V / 低压母线 {cfg.DC_LOW_VOLTAGE_V}V）')

    # 消纳率（十一(一)1 源荷储一体化）：从方案B的统计模块取，纳入方向3汇报
    pv_daily_all = pv.resample('D').sum()
    cons = st.calc_consumption_rate(pv_daily_all, load_daily)
    print(f'    光伏消纳率（全期）：{cons["消纳率"]*100:.2f}%，'
          f'自消纳量 {cons["期内被消纳发电量_kWh"]:,.0f} kWh / 用电 {cons["期内总用电量_kWh"]:,.0f} kWh')
    # 上面 f 字符串里用了双引号 {cons["消纳率"]}：
    #   因为外层字符串是单引号 '...'，里面再用单引号会打架，
    #   所以字典取值改用双引号。这是 Python 里很常见的小技巧。
    # 数字格式 {x:,.0f} 表示"千位加逗号、保留 0 位小数"，例如 8519419 → 8,519,420。

    # 方面1：源-荷预测（保存返回结果，存图时还要用）
    fc_res = fc.run_forecast(df)

    # 方面2：鲁棒优化 + 随机规划
    #   da = day-ahead 日前计划（提前一天安排）
    #   rt = real-time 实时修正（真数据来了以后滚动调整）
    #   baseline = 不装储能时的对照，用来算"加储能省了多少"
    print('\n[2] 方面2：在代表日上做日前随机规划 + 实时滚动(MPC) ...')
    da = opt.run_day_ahead_stochastic(pv_day, load_day, price_24, carbon_24)
    rt = opt.run_real_time_robust(pv_day, load_day, price_24, carbon_24)
    baseline = opt.baseline_no_storage(pv_day, load_day, price_24, carbon_24)
    opt.print_optimize_summary(da, rt, baseline)

    # 方面3：大数据挖掘 + 云边仿真
    print('\n[3] 方面3：大数据挖掘 + 云边协同仿真 ...')
    mined = mng.mine_typical_days(df['pv'])       # 典型日聚类
    corr = mng.analyze_correlation(df)            # 源-荷相关性
    anomalies = mng.detect_anomalies(load_daily)  # 异常日检测
    # prefetch 的作用：把上面已经算好的 da/rt/baseline 直接传进去"复用"，
    # 否则云边仿真内部会再算一遍同样的优化（白跑一次，浪费几秒）。
    ce = mng.simulate_cloud_edge(df['pv'], load_daily, price_24, carbon_24,
                                 prefetch=(da, rt, baseline, str(rep_day.date())))
    mng.print_framework_summary(mined, corr, anomalies, ce)

    # 存图（你不便看图，但保留下来写报告/汇报可用）
    _save_direction3_figures(fc_res, da, mined)

    print('\n========== 研究方向3 三个方面全部跑完 ==========')
    print('提示：方面1/2/3 各自的数值结论已打印在上方；')
    print('      图已存到', __import__('os').path.join(cfg.FIG_OUT_DIR, 'direction3'), '（可选查看）。')
    print('      想改参数（电池容量、电价、碳因子…）只改 config.py 一处即可。')


# -----------------------------------------------------------------------------
# 流程 D：SOTA 预测基线对比 + 消融（方面1 扩展）
# -----------------------------------------------------------------------------
# 做什么：回答评审老师一定会问的两件事——
#   ① 你的预测方法比别人的好多少？（对比 SOTA 基线）
#   ② 你方法里的每个零件都真的有用吗？（消融实验）
# 注意：这项比较耗时（要训练多个深度学习模型），所以不在 "all" 模式里自动跑。
def run_benchmark():
    print('#' * 60)
    print('# 预测基准对比 + 消融：季节朴素 / XGBoost / 融合(现有方法) / '
          'LSTM / Transformer / N-BEATS')
    print('#' * 60)
    # pv（光伏）和 load（负荷）两个预测目标各跑一遍。
    for tgt in ['pv', 'load']:
        print(f'\n########## 目标 = {tgt} ##########')
        bm.run_forecast_benchmark(tgt, verbose=True, save_fig=True)
    print('\n✅ 预测基准对比完成（含消融与末60日预测对比图，存于 figures/benchmark）。')


# -----------------------------------------------------------------------------
# 流程 E：全年滚动调度评估 + 敏感性分析（方面2 扩展）
# -----------------------------------------------------------------------------
# 做什么：把代表日的一天实验，放大到全年 200 多天；再逐个改参数看结果稳不稳。
# 注意：这项最耗时（上千次优化求解），所以也不在 "all" 模式里自动跑。
def run_rolling():
    print('#' * 60)
    print('# 全年滚动调度评估 + 敏感性分析（电池容量 / 碳价 / 预测误差）')
    print('#' * 60)
    print('\n[1/4] 全年滚动调度评估（无储能 vs 日前随机 vs 实时MPC）...')
    re.rolling_dispatch_evaluation()      # 全年逐日跑，给出省钱/降碳的置信区间
    print('\n[2/4] 敏感性①：电池容量...')
    re.sensitivity_battery_capacity()     # 电池造大一点，收益能多多少？
    print('\n[3/4] 敏感性②：碳价...')
    re.sensitivity_carbon_price()         # 碳价高一点，结论会不会翻？
    print('\n[4/4] 敏感性③：预测误差...')
    re.sensitivity_forecast_error()       # 预测不准时，方案会不会崩？
    print('\n✅ 全年滚动 + 敏感性分析完成。')


# -----------------------------------------------------------------------------
# 流程 C：净购电小时曲线（账单锚定还原，方案C）
# -----------------------------------------------------------------------------
# 做什么：把"电能表每小时从电网买了多少电"还原成一条逐时曲线。
# 为什么要"锚定"：电能表抄数偶尔有缺漏，靠自己算对不上账单；
#   而电费账单上的"每月总购电量"是权威数字。
#   所以做法是——用小时数据算出形状，再按账单月总量成比例缩放，
#   让每月合计严丝合缝等于账单。这叫"锚定缩放"。
def run_netload():
    print('=' * 60)
    print('  净购电小时曲线还原（电费清单/每小时电量 · 账单逐月锚定）')
    print('=' * 60)

    print('\n[1/3] 读取小时表 + 账单锚定还原 ...')
    net = ld.load_net_purchase_hourly()
    if net.empty:
        print('\n⚠️ 净购电曲线还原失败，请检查每小时电量 xls 与总账单 pdf。')
        return
    print(f'    曲线点数 : {len(net)} 小时（{net.index.min()} ~ {net.index.max()}）')

    # 与账单月总量交叉验证
    print('\n[2/3] 与电费账单月净购电量交叉验证 ...')
    print('    （这一步验证“锚定缩放”是否成功：还原后的月合计应≈账单月合计）')
    bill = ld.bill_monthly_net_purchase()     # 从 PDF 账单读出的每月净购电量
    net_month = net.resample('ME').sum()      # 把还原曲线按月汇总
    print('    月        还原净购(万kWh)  账单净购(万kWh)  偏差%')
    max_dev = 0.0                             # 记录最大偏差（可能为负，取绝对值比较）
    for ym, v in net_month.items():
        key = ym.strftime('%Y-%m')            # 把时间戳变成 '2025-09' 这种字符串当键
        if key in bill and bill[key] > 0:     # 账单里可能没有这个月，先判断存在且非零
            b = bill[key]
            dev = (v - b) / b * 100           # 偏差百分比（正=还原多了，负=还原少了）
            max_dev = max(max_dev, abs(dev))  # abs() 取绝对值，只关心偏多少不看方向
            print(f'    {key}      {v/1e4:8.2f}      {b/1e4:8.2f}      {dev:5.1f}')
            # v/1e4 是把 kWh 换成"万 kWh"（1e4 就是 10000），数字小了好读。

    # 与账单交叉验证后，显式列出“账单有、但小时表未还原/无锚定”的缺失月份，
    # 避免“最大偏差 0.0%”给人“已完整还原”的误导（缺失月根本没进曲线，不会体现在偏差里）。
    # 这是本项目一直坚持的"诚实性"原则：数据缺了就要说出来，不能装作没有缺口。
    restored = {ym.strftime('%Y-%m') for ym in net_month.index}   # { } 是集合，便于快速查"在不在"
    missing = sorted(k for k in bill if k not in restored)
    if missing:
        print('    ⚠️ 以下账单月份未能还原（小时表缺失或无对应锚定，曲线中存在缺口）：')
        for k in missing:
            print(f'       {k}  账单净购 {bill[k]/1e4:8.2f} 万kWh（未还原）')

    # 与能耗平台“总负荷”对比，说明口径差异
    print('\n[3/3] 口径说明：净购电 vs 园区总用电 ...')
    load_total = ld.load_load_series()
    if not load_total.empty:
        # 对齐到相同月份对比（取两个时间轴的交集，只比双方都有的月份）
        net_y = net.resample('ME').sum()
        tot_y = load_total.resample('ME').sum()
        common = net_y.index.intersection(tot_y.index)
        if len(common) > 0:
            ratio = net_y.loc[common].sum() / tot_y.loc[common].sum()
            print(f'    重叠期净购电合计 / 总用电合计 = {ratio*100:.1f}%')
            print('    （差额=光伏就地消纳量，不进电网表；所以净购电<总用电）')
    # 峰谷比（净购电曲线自身的日内形状）
    daily = net.resample('D').sum()
    daily = daily[daily > 0]                       # 剔除异常零日（如9月仅7天缩放失真）
    if len(daily) > 0:
        pk = daily.max(); tr = daily.min()
        print(f'    日内峰谷比（日总量）：峰 {pk/1e3:.1f} MWh / 谷 {tr/1e3:.1f} MWh = {pk/tr:.2f}')
        # pk/1e3 是把 kWh 换成 MWh（1 MWh = 1000 kWh）。
    span = f'{net.index.min().date()} ~ {net.index.max().date()}'
    if missing:
        print(f'\n⚠️ 还原完成（含缺口）：曲线覆盖 {span}，已还原月份与账单最大偏差 '
              f'{max_dev:.1f}%；另有 {len(missing)} 个账单月份无小时表数据未还原（见上方清单）。')
    else:
        print(f'\n✅ 还原完成：曲线覆盖 {span}，已还原月份与账单最大偏差 '
              f'{max_dev:.1f}%（锚定法下应≈0）。若开头/结尾月份缺失，见上方“曲线存在缺口”WARNING。')
    print('    说明：本曲线是“从电网净购入”的电量（不含光伏就地消纳），')
    print('          用于购电优化/能-碳调控；园区总负荷请用 stats 模式的能耗平台数据。')


def _save_direction3_figures(fc_res, da, mined):
    """存方向3的三张示意图（matplotlib 不可用就跳过）。

    参数
    ----
    fc_res : dict
        run_forecast() 的返回值，其中 'demo' 键里是演示日的实际/预测曲线。
    da : dict
        run_day_ahead_stochastic() 的返回值，其中 'nominal' 键里是电池调度结果。
    mined : dict
        mine_typical_days() 的返回值，其中 'centroids' 是典型日曲线矩阵。

    返回
    ----
    None（副作用：把 3 张 png 存到 figures/direction3/）

    ⚠️ 为什么整个函数包在 try 里？
        matplotlib（画图库）不一定装得上，而且它依赖系统字体。
        画图只是"锦上添花"，不能因为画不了图就让整个方向3 运行崩掉，
        所以一旦环境有问题就打印警告后直接返回（这叫"优雅降级"）。
    """
    try:
        import os
        import matplotlib
        matplotlib.use('Agg')     # 'Agg' = 不用弹窗显示，直接存成文件（服务器/无人环境也能用）
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        # 找系统里的中文字体。不设置的话，图里的中文会变成一个个方框（□□□）。
        for f in ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']:
            if os.path.exists(f):
                font_manager.fontManager.addfont(f)
                # 从文件名里取出字体名（去掉扩展名），例如 simhei.ttf → 'simhei'
                plt.rcParams['font.sans-serif'] = [os.path.basename(f).split('.')[0]]
                break     # 找到第一个能用的就停，后面的不用再找
        plt.rcParams['axes.unicode_minus'] = False   # 让"负号"正常显示而不是方块
    except Exception as e:
        logger.warning('跳过存图：matplotlib 不可用（%s）', e)
        return

    out = os.path.join(cfg.FIG_OUT_DIR, 'direction3')
    os.makedirs(out, exist_ok=True)      # exist_ok=True：目录已存在也不报错

    # ---- 图1：方面1 源-荷预测（演示日实际值 vs 两种预测）----
    d = fc_res.get('demo')
    if d is not None:                    # 有可能没有 demo，那样就跳过这张图
        plt.figure(figsize=(10, 4))      # 新建一张 10×4 英寸的画布
        plt.plot(d['hour_index'].hour, d['actual_pv'], 'k-', label='实际')
        plt.plot(d['hour_index'].hour, d['day_ahead_pv'], 'b--', label='日前预测')
        plt.plot(d['hour_index'].hour, d['realtime_pv'], 'g-.', label='实时修正')
        # 'k-' = 黑色实线，'b--' = 蓝色虚线，'g-.' = 绿色点划线（k/b/g 是颜色首字母）
        plt.xlabel('小时'); plt.ylabel('PV (kWh)'); plt.title('方面1 源-荷预测（演示日）')
        plt.legend(); plt.tight_layout()  # legend 显示图例，tight_layout 自动排整齐不重叠
        plt.savefig(os.path.join(out, 'dir3_forecast.png'), dpi=120); plt.close()
        # dpi=120 是清晰度，plt.close() 释放内存（不关的话画多了会卡）

    # ---- 图2：方面2 代表日电池调度（SOC 曲线 + 充放电柱）----
    nom = da['nominal']
    plt.figure(figsize=(10, 4))
    plt.plot(nom['soc'], 'b-', label='电池SOC (kWh)')
    # 柱子稍微左右错开 0.2，否则充电柱和放电柱会叠在一起看不清。
    # 放电画成负值，这样"往上=充电、往下=放电"一眼能看出来。
    plt.bar(np.arange(24) - 0.2, nom['charge'], width=0.4, color='g', label='充电')
    plt.bar(np.arange(24) + 0.2, -nom['discharge'], width=0.4, color='r', label='放电')
    plt.xlabel('小时'); plt.ylabel('功率/电量'); plt.title('方面2 代表日电池调度')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out, 'dir3_dispatch.png'), dpi=120); plt.close()

    # ---- 图3：方面3 典型日聚类中心（K 条曲线，每条代表一类天气）----
    cen = mined['centroids']
    plt.figure(figsize=(10, 4))
    for k in range(cen.shape[0]):        # cen.shape[0] = 有几行 = 有几个簇（K 个）
        plt.plot(cen[k], label=f'簇{k}')
    plt.xlabel('小时'); plt.ylabel('PV (kWh)'); plt.title('方面3 典型日聚类中心')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out, 'dir3_typical_days.png'), dpi=120); plt.close()

    logger.info('已存图：%s', out)


# -----------------------------------------------------------------------------
# 主入口
# -----------------------------------------------------------------------------
def main():
    # argparse 是 Python 自带的"命令行参数解析"工具。
    # 它的作用是：让程序能接收你在命令行敲的参数，例如 --mode stats。
    parser = argparse.ArgumentParser(description='本园区 能-碳调控工具包')
    parser.add_argument('--mode', choices=['all', 'stats', 'direction3', 'netload',
                                            'benchmark', 'rolling'],
                        default='all', help='运行模式：all=全部，stats=方案B统计，'
                                            'direction3=方向3，netload=净购电小时曲线，'
                                            'benchmark=预测基线对比+消融，rolling=全年滚动+敏感性')
    # choices=[...] 限定只能填这几项，填别的会直接报错并提示。
    # default='all' 表示"你什么都不加时，默认跑 all"。
    args = parser.parse_args()    # 把命令行里的实际内容读进来

    # 按 --mode 决定跑哪几个流程。
    # 注意 'all' 用的是 in（属于），意思是"all 的时候顺便把 stats 和 direction3 也跑了"；
    # 而后三个用 ==（等于），因为它们是重活，只在明确指定时才跑（见文件头说明）。
    if args.mode in ('all', 'stats'):
        run_stats()
    if args.mode in ('all', 'direction3'):
        run_direction3()
    if args.mode == 'netload':
        run_netload()
    if args.mode == 'benchmark':
        run_benchmark()
    if args.mode == 'rolling':
        run_rolling()


if __name__ == '__main__':
    # 这句是 Python 惯例：只有“直接运行 python -m pvflex.cli”时才执行 main()；
    # 被别的文件 import 时不会自动跑。
    #
    # 为什么要有这句？因为你的代码可能被别人 import 去复用。
    # 如果没这个判断，别人一 import 你的文件，整个分析流程就自己跑起来了，
    # 既浪费时间又可能覆盖文件 —— 加上这句就只在本文件被直接运行时才执行。
    main()
