# pvflex/analysis —— 项目数据统计分析子包
# -----------------------------------------------------------------------------
# 【新手看这里】这个包负责"把真实数据算成直观结论"：
#   一年发了多少电、每天平均用多少电、太阳能自己用掉多少（消纳率）、
#   一年电费大概多少。它不追求复杂算法，就是把数据汇总成能写进报告的数字。
#
# 【为什么这个"最简单"的包反而最该先看？】
#   因为汇报时最先被问到的就是这些数（"你们园区一年用多少电？消纳率多少？"）。
#   它们算错了，后面再花哨的优化也站不住。所以建议读代码的顺序是：
#   先看 analysis/stats.py 搞懂"数是怎么来的"，再看别的模块。
#
# 【核心文件】
#   stats.py  ——  每个分析都拆成 analyze_算数 和 print_打印 两个函数。
#                 这样拆的原因：算出的 dict 能被别的模块复用
#                 （比如 cli.py 把消纳率的结果再传给电费计算），
#                 而 print 只管给人看。两者混在一起就没法复用了。
# -----------------------------------------------------------------------------
# 下面这一大段 import 是把 stats.py 里的函数"转发"出来，
# 使得别处可以写 `from pvflex.analysis import analyze_pv`（短路径），
# 而不用写 `from pvflex.analysis.stats import analyze_pv`（长路径）。
# 纯粹是图方便的写法，不影响功能。
from pvflex.analysis.stats import (
    analyze_pv, print_pv_summary,
    analyze_load, print_load_summary,
    calc_consumption_rate, print_consumption_summary,
    calc_cost, print_cost_summary,
    make_figures, set_chinese_font, save_fig,
)
