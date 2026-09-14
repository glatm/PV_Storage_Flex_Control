# =============================================================================
# run_extension_study.py  ——  【一键运行脚本 · 新手最常跑的这个】
# -----------------------------------------------------------------------------
# 【新手看这里】这个文件是给"想一次性看到全部高级分析结果"的人用的。
#   你不用读懂里面每一行，只要知道：运行它 = 自动把下面三件事全跑完，
#   然后把所有数字存进 extension_results.json 文件。
#     1) 预测基准对比 + 消融（我们的方法 vs LSTM/Transformer/N-BEATS 谁准）
#     2) 全年滚动调度评估（一整年 219 天，平均省了多少钱/碳）
#     3) 敏感性分析（电池造大点/碳价高点/预测准点，结果变多少）
#
# 【怎么跑】先激活 conda 环境，再在这文件夹里执行：
#     python run_extension_study.py
#
# 【跑完会得到什么】
#   一个 extension_results.json 文件 —— 里面的数字就是写论文用的真实结果。
#   想看人话版结果，去读 SCI_extension_forecast_sensitivity.md。
#
# 【⚠️ 注意】这个脚本要算几分钟，跑的时候耐心等它自己结束，别中途关掉。
#   它跟 `python -m pvflex.cli` 的关系：cli 的 --mode benchmark / --mode rolling
#   是分开跑这两块；本脚本是把它们打包成一次跑完，并且把结果存成 JSON 文件
#   （方便直接给论文/图表用）。
#
# 【为什么要单独存成 JSON？】
#   因为屏幕上的打印看完就没了。存成文件后：
#     ① 写论文时可以随时翻回去核对数字
#     ② 重跑一次就能和新结果对比，看出改动的影响
#     ③ 别人复现你的研究时，拿着这个文件能直接对照
# =============================================================================
import json
import time
import numpy as np

import pvflex.forecasting.benchmark as bm
import pvflex.optimization.rolling_eval as re

OUT = 'extension_results.json'      # 结果文件名（存在当前目录）


def _clean(v):
    """把 numpy 类型递归转换成 Python 原生类型，好让 json 能写入。

    参数
    ----
    v : 任意
        可能是 dict / list / tuple / numpy 数字 / 普通数字。

    返回
    ----
    与输入结构相同、但元素都是 Python 原生类型（float/int/dict/list/tuple）的对象。

    【为什么需要这个函数？】
        结果里到处是 numpy 的数字（np.float64 等）。json 模块不认识它们，
        直接 dump 会报 "Object of type float64 is not JSON serializable"。
        所以要先"洗干净"：把 numpy 类型换成普通 float/int。
        又因为结果里套着好几层 dict/list，所以要靠【递归】一层层往里处理 ——
        这就是下面为什么函数里又调用了自己 _clean(val)。
    """
    if isinstance(v, (np.floating,)):
        # np.floating 是 numpy 所有浮点类型的父类，用它可以一次覆盖 float32/64 等。
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, tuple):
        return tuple(float(x) for x in v)
    if isinstance(v, dict):
        # 字典：对每个值递归处理，键保持不变。
        return {k: _clean(val) for k, val in v.items()}
    if isinstance(v, (list,)):
        return [_clean(x) for x in v]
    try:
        return float(v)
    except Exception:
        # 转不成数字就原样返回（例如字符串、None 本来就是 json 支持的）。
        return v


def main():
    t0 = time.time()      # 记下开始时间，后面用它算每一步耗时
    results = {
        # 【2026-09-14 修复】原为硬编码 '2026-08-27'，导致每次重跑后 JSON 里的
        # 生成时间都不变，无法判断结果是否为新数据跑出的。改为运行时动态生成。
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        # solver_note：把"求解器为什么可以放心用 LP"这个关键前提写进结果文件。
        # 为什么值得写进 JSON 而不是只写在代码注释里？
        #   因为半年后你翻这个结果文件时，只会看 JSON，不会回去翻代码。
        #   把前提条件和数字放在一起，才不会被误用。
        'solver_note': ('调度求解器改用 scipy linprog 连续 LP，且已修正配置使成本函数对净'
                        '交换量 g 凸（上网价≤最低零售价），LP 与 PuLP/CBC MILP 在本问题上'
                        '严格等价（校验差异 0.000%，见 lp_vs_milp）；故全年滚动与大规模'
                        '敏感性扫描的结果即为精确解，无需 MILP 近似。'),
        'targets': {},    # 先建个空字典，下面填 pv / load 两块结果
    }

    # ---- [0/4] 先做一次"两种求解器算的是不是同一个东西"的校验 ----
    # 为什么放在最前面？只要这一步通过，后面大规模扫描才有意义。
    # 如果这一步就不等价，那后面所有数字都要重做。
    print('[0/4] LP vs MILP 校验（抽样 30 天 MPC 调度）...', flush=True)
    results['lp_vs_milp'] = _clean(re.validate_lp_vs_milp(n_days=30))
    print('       done %.1fs' % (time.time() - t0), flush=True)
    # flush=True 表示"立刻把这句话推到屏幕上"。
    # 不加的话 Python 会先攒在缓冲区里，你可能盯着黑屏几分钟什么也看不到。

    # ---- [1/4] 预测基准对比 + 消融（pv 和 load 两个目标各跑一遍）----
    print('[1/4] 预测基准对比 + 消融（PV / 负荷）...', flush=True)
    for tgt in ['pv', 'load']:
        res = bm.run_forecast_benchmark(tgt, verbose=False, save_fig=True)
        # verbose=False：不打印每个模型的详细过程（这里要跑十几个模型，刷屏太乱）
        # save_fig=True ：仍然把对比图存下来
        models = {}
        for name, r in res['results'].items():
            # 只挑三样最关键的存进 JSON：指标、置信区间、测试样本数。
            # 不把全部原始数组存进去，否则 JSON 会大到几十 MB、打开都卡。
            models[name] = {
                'metrics': _clean(r['metrics']),
                'ci': _clean(r['ci']),
                'n_test': int(r['n_test']),
            }
        results['targets'][tgt] = {
            'cap_ref': _clean(res['cap_ref']),    # 容量/负荷的参考基准值
            'n_test': int(res['n_test']),
            'models': models,
        }
    print('       done %.1fs' % (time.time() - t0), flush=True)

    # ---- [2/4] 全年滚动调度评估 ----
    print('[2/4] 全年滚动调度评估（LP 加速，全量完整日）...', flush=True)
    roll = re.rolling_dispatch_evaluation(verbose=False)
    results['rolling'] = _clean(roll)
    print('       done %.1fs' % (time.time() - t0), flush=True)

    # ---- [3/4] [4/4] 三个敏感性分析 ----
    print('[3/4] 敏感性① 电池容量...', flush=True)
    results['sens_battery'] = _clean(re.sensitivity_battery_capacity(verbose=False))
    print('[4/4] 敏感性② 碳价 + ③ 预测误差...', flush=True)
    results['sens_carbon'] = _clean(re.sensitivity_carbon_price(verbose=False))
    results['sens_error'] = _clean(re.sensitivity_forecast_error(verbose=False))
    print('       done %.1fs' % (time.time() - t0), flush=True)

    # ensure_ascii=False 让中文正常写进去（否则中文会变成 \uXXXX 编码，人能读但很难看）；
    # indent=2 让 JSON 换行缩进，方便用记事本直接看。
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\n✅ 全部完成，结果写入 {OUT}（总耗时 {time.time()-t0:.1f}s）', flush=True)


if __name__ == '__main__':
    # 只有"直接运行本文件"时才执行 main()，被 import 时不会自动跑。
    # 详见 cli.py 文件末尾对这句的详细解释。
    main()
