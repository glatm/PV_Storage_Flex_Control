# =============================================================================
# pvflex/__init__.py  ——  顶层包初始化
# -----------------------------------------------------------------------------
# 【这个文件解决什么问题】
#   这是整个 Python 包的"门牌"文件。它的作用是：
#     ① 告诉 Python "pvflex 是一个包（可以被 import 的文件夹）"
#     ② 让 `import pvflex` 时能顺手拿到常用子模块
#     ③ 用一段文字写清这个包到底是干什么的（这段文字叫 docstring）
#   你平时写代码一般用不到动这里，知道"整个工程叫 pvflex"即可。
#
# 【⚠️ 新手最容易困惑的一点：为什么文件夹里都有个 __init__.py？】
#   Python 规定：一个文件夹要能被 import，里面必须有 __init__.py。
#   所以每个子目录（data / forecasting / optimization ...）里都有一份，
#   内容大同小异 —— 就是"打招呼 + 把对外常用的函数列出来"。
#   你不用去记它，只要记住"看到 __init__.py 说明这是个可导入的包"就够了。
#
# 【本文件里的 __version__ 有什么用？】
#   记录代码版本号。论文/报告里写"结果由 pvflex v1.0.0 产生"，
#   以后代码改了、结果变了，靠版本号能对上号，避免"这数字是哪版跑出来的"说不清。
# =============================================================================

"""pvflex —— 本园区 光储直流微电网 能-碳智能调控工具包。

【这个包包含什么：六个功能模块】
  - data        数据读取与预处理（PV 发电量、园区负荷）
  - forecasting 方面1 源-荷多时间尺度预测
  - optimization 方面2 鲁棒优化 + 随机规划 能-碳调控
  - mining      方面3 大数据挖掘 + 云边协同框架
  - carbon      碳排放计算
  - simulation  700V 模型背景参数
  - analysis    项目数据统计分析（发电/负荷/消纳率/电费）

【模块之间的数据流（从哪里进、到哪里出）】
    数据资料/ (原始文件：CSV、XLSX、PDF)
        │
        ▼
      data  ──→  干净的逐时数据（Series / DataFrame）
        │
        ├──→  analysis   → 年发电/年用电/消纳率/电费（方案B统计结论）
        │
        ├──→  forecasting → 明天/未来几小时发多少电、用多少电
        │                      │
        │                      ▼
        ├──→  optimization → 电池该充还是该放？省多少钱、减多少碳（核心）
        │                      │
        │                      ▼
        ├──→  carbon     →  单独核算碳排放量（写碳报表用）
        │
        └──→  mining     → 典型日归纳 + 云边架构演示（把上面几块串起来讲）

    最上层 cli.py 负责按顺序调用上面这些模块，也就是"总开关"。

【记住三个时间尺度，就不容易乱】
  - 日前（day-ahead）：提前一天算好计划（用预测数据）
  - 实时（real-time / MPC）：真数据来了以后滚动修正（每几小时调一次）
  - 长期评估（rolling）：把上面两件事在全年 200 多天上重复做，看平均效果
"""

__version__ = '1.0.0'

# 这几行 import 的作用：让 `import pvflex` 之后，可以直接写
#   pvflex.data.loaders.xxx()
# 而不用先 `import pvflex.data.loaders`。
# 属于"方便调用"的写法，不影响功能正确性。
from pvflex import config
from pvflex.data import loaders, preprocess
from pvflex import forecasting, optimization, mining, carbon, simulation, analysis
