# pvflex/data —— 数据读取与预处理子包
# 【新手看这里】整个程序的"进货口"。它负责把放在"数据资料"文件夹里的
#   原始文件(CSV表格、PDF账单)读进来，整理成整齐的时间序列。
#   没有它，后面的预测和调度就没有原料。核心文件：loaders.py(读取)、
#   preprocess.py(整理对齐)、pdf_readers.py(从账单抠真实电价)。
from pvflex.data.loaders import (
    load_pv_generation,
    load_load_series,
    load_self_consumption_hourly,   # 【2026-09-14 新增】真实逐时源-荷-消纳(8760h)
    load_real_load_shape,           # 【2026-09-14 新增】真实24h归一化负荷形状
)
from pvflex.data.preprocess import (
    to_hourly_load, align_pv_load, build_features, split_train_test,
    TYPICAL_LOAD_SHAPE,             # 24h 形状，恒为【占比、sum=1】（与来源无关）
    TYPICAL_LOAD_SHAPE_RAW,         # 同上但【未归一】（kWh 量纲），需要真实量纲时用
    load_shape_source,              # 【2026-09-14 新增】查询形状来源（实测/估算）
    _load_real_load_shape as _load_real_load_shape_csv,  # 【2026-09-14 新增】CSV实测形状
)
from pvflex.data import pdf_readers  # PDF 账单/结算单读取（校准电价用）
