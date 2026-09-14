# =============================================================================
# data/pdf_readers.py  ——  从真实账单/结算单 PDF 提取电价与上网电量
# -----------------------------------------------------------------------------
# 【这个文件解决什么问题】
#   这个文件用来把“电费清单/总账单”和“光伏结算单”里的真实数字读出来，
#   方便你核对、校准 config.py 里的电价参数（不再靠拍脑袋填示例值）。
#
# 【为什么值得专门写个文件去抠 PDF？】
#   因为"电价"这个参数对结论影响非常大，但它没法从国内公开渠道查到园区级精确值。
#   唯一权威来源就是你手里的真实账单。而账单是 PDF、表格排版还经常错位，
#   手抄容易出错也没法复现。写代码读虽然麻烦，但：
#     ① 数字有出处、可追溯（论文里能写"取自 XX 期真实账单"）
#     ② 换新账单重新跑一遍就行，不用重抄
#     ③ 万一有人质疑电价，你能当场复现
#
# 【背景小课堂】
#   - 电费账单 pdf（国网福建）：每期给出“本期电量、本期电费、峰谷分时比例”，
#     可以算出真实加权均价，用来校准 TARIFF_AVG / GRID_BUY_PRICE_24H。
#   - 光伏结算单 pdf：给出各发电表计“反相有功（上网）”的尖/峰/平/谷电量与结算单价，
#     可以算出真实上网结算价，用来校准 GRID_SELL_PRICE。
#
# 【怎么用】
#   from pvflex.data import pdf_readers as pr
#   avg = pr.weighted_buy_price()      # 读全部电费账单，返回加权买电均价(元/kWh)
#   sell = pr.weighted_sell_price()    # 读全部光伏结算单，返回加权上网价(元/kWh)
#   print(avg, sell)
#
# 【依赖】pypdf（已在 requirements.txt 里）。如果没装，函数会返回 None 并提示。
#
# 【⚠️ 关于"PDF 表格抽取"的一个重要提醒】
#   PDF 里其实没有"表格"这个东西，只有一堆散落在页面上的文字块。
#   程序读出来是"文本流"，要自己用正则表达式去猜"哪几个数字是一组的"。
#   好处是偶尔错位，所以本文件所有函数都做了"抓不到就返回 None"的处理，
#   而不是硬猜一个数出来 —— 宁可让你手动看，也不要给你一个错的数据。
# =============================================================================

import os
import glob
import logging

import numpy as np
import pandas as pd

import pvflex.config as cfg

logger = logging.getLogger(__name__)


def _read_pdf_text(path):
    """用 pypdf 把 pdf 所有页文本拼成一个字符串；读不到返回空串。

    参数
    ----
    path : str
        PDF 文件的完整路径。

    返回
    ----
    str
        全部页面的文字拼在一起（页与页之间用换行连接）。
        读失败时返回空字符串 ''，而不是抛异常 —— 这样上层函数
        只要检查"是不是空串"就能判断成功与否，不用写 try/except。

    【为什么要做成"失败不报错"？】
        因为要批量读几十上百个 PDF。如果某个文件损坏就整个流程崩掉，
        那么你跑了 10 分钟的结果全白费。返回空串则相当于"跳过这个文件"，
        其他文件照样处理，只是那份数据缺失（日志里会有警告）。
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        # 库没装的话只提示一次、返回空串，让程序继续跑完其余逻辑。
        logger.warning('未安装 pypdf，无法读取 PDF（pip install pypdf）')
        return ''
    try:
        r = PdfReader(path)
        # 逐页调用 extract_text() 取文字，'\\n'.join(...) 把它们用换行拼成一个大字符串。
        # 有些页可能没有文字（纯图片扫描件），extract_text() 会返回 None，
        # 所以写 (p.extract_text() or '') 兜一下，防止 None 参与拼接时报错。
        return '\n'.join((p.extract_text() or '') for p in r.pages)
    except Exception as e:
        logger.warning('读取 PDF 失败 %s：%s', os.path.basename(path), e)
        return ''


def weighted_buy_price():
    """读“电费清单/总账单”下所有 pdf，按“总电费 / 总电量”算加权买电均价。

    返回
    ----
    float 或 None
        加权买电均价（元/kWh）；一个账单都没读到、或总电量为 0 时返回 None。

    【"加权均价"和"算术平均"的区别】
        假设两期账单：第 1 期用了 100 度花了 50 元，第 2 期用了 10000 度花了 6000 元。
        算术平均：(0.50 + 0.60) / 2 = 0.55 元/度  ← 错！把 100 度和 10000 度当同等重要
        加权平均：(50+6000) / (100+10000) = 0.599 元/度  ← 对，按电量加权
        用电量差异很大时两者能差很多，所以必须用加权。
    """
    folder = os.path.join(cfg.REPO_DATA_DIR, '电费清单', '总账单')
    # glob 是"按通配符找文件"。'**' 配合 recursive=True 表示"连子文件夹一起找"。
    files = sorted(glob.glob(os.path.join(folder, '**', '*.pdf'), recursive=True))
    if not files:
        logger.warning('未找到电费账单 PDF：%s', folder)
        return None
    total_kwh = 0.0
    total_fee = 0.0
    import re
    # re 是 Python 的正则表达式库，可以在大段文字里按"模式"找内容。
    # 延迟到函数内部 import 是为了让"不用这个函数的人"不必加载它（省启动时间）。
    for fp in files:
        txt = _read_pdf_text(fp)
        # 下面两个模式解释：
        #   r'本期电量\s*([\d,.]+)\s*千瓦时'
        #     r'...'      → 原样字符串（反斜杠不当转义符）
        #     本期电量     → 字面匹配这四个字
        #     \s*        → 中间允许有任意多个空白/换行（PDF 抽取后常有奇怪空格）
        #     ([\d,.]+)  → 【捕获组】抓一串"数字、点、逗号"（就是要提取的数值）
        #     千瓦时      → 后面跟着的单位，用来确认抓对了地方
        m1 = re.search(r'本期电量\s*([\d,.]+)\s*千瓦时', txt)
        m2 = re.search(r'本期电费\s*([\d,.]+)\s*元', txt)
        if m1 and m2:
            # group(1) 取第 1 个捕获组的内容；账单里数字带千位逗号，要先去逗号再转数字。
            total_kwh += float(m1.group(1).replace(',', ''))
            total_fee += float(m2.group(1).replace(',', ''))
    if total_kwh <= 0:
        return None     # 分母保护
    avg = total_fee / total_kwh
    logger.info('买电加权均价：%.4f 元/kWh（电量 %.0f，电费 %.0f）',
                avg, total_kwh, total_fee)
    return avg


def weighted_sell_price():
    """读“光伏结算单”下所有 pdf，提取明确写出的上网结算单价。

    参数
    ----
    无

    返回
    ----
    float 或 None
        上网结算参考价（元/kWh）；提取不到返回 None。

    做法：如“尖峰电价 3270.47 0.84789131 2773.00”中的 0.84789131。
          取“尖峰/峰”档单价的中位数作为“加权上网价（参考）”。

    注：pdf 表格抽取偶尔错位，这里只抓“明确标出的电价数字”，
        比用“总量÷电量”稳健；正式值以结算单系统导出为准。

    【⚠️ 本函数的结果是"参考值"，不是本项目实际使用的参数】
        原因见 config.py 里 GRID_SELL_PRICE_YUAN_PER_KWH 的长注释：
        真实结算价 0.8479 高于谷段零售价 0.420，会构成"谷时买、卖高价"的
        非法套利，让优化模型算出发散的假解。
        所以本项目取 0.39（燃煤基准价）作为建模值，
        这个函数存在的意义是"记录真实值"，好在论文里讨论差异。
    """
    folder = os.path.join(cfg.REPO_DATA_DIR, '光伏结算单')
    files = sorted(glob.glob(os.path.join(folder, '*.pdf')))
    if not files:
        logger.warning('未找到光伏结算单 PDF：%s', folder)
        return None
    import re
    pairs = []   # (档位名, 单价)
    for fp in files:
        txt = _read_pdf_text(fp)
        # 抓 “尖峰电价 X.XX 0.84789131 Y.YY” / “峰电价 ...” 这类行里的单价
        # 单价特征：0. 开头、4~8 位小数
        #   正则解释： (尖峰电价|峰电价|平电价|谷电价)  → 四选一，括号里用 | 分隔
        #              \s*[\d.]+\s*                     → 中间夹着一个数字（电量）
        #              (0\.\d{4,8})                     → 捕获"0.开头+4~8位小数"的单价
        #   {4,8} 表示"重复 4 到 8 次"，用来避免把 0.5 这种不完整的数误抓进来。
        for m in re.finditer(r'(尖峰电价|峰电价|平电价|谷电价)\s*[\d.]+\s*(0\.\d{4,8})', txt):
            # finditer 找【所有】匹配（search 只找第一个），因为一份结算单有多档电价。
            try:
                pairs.append((m.group(1), float(m.group(2))))
            except ValueError:
                pass     # 万一转数字失败就跳过这一条，不影响其他
    if not pairs:
        logger.warning('结算单未提取到明确上网单价')
        return None
    # 上网结算单价应取“尖峰/峰”档（与 config.GRID_SELL_PRICE 对齐）；
    # 平/谷档价低，若一起取中位会把 0.85 与 0.35 混算成 ~0.747（失真）。
    # 故只取尖峰/峰档做中位数；某文件缺这俩档时再退回全档中位数兜底。
    # 用中位数而不是平均数的原因：中位数不怕个别错位抓来的离谱数字。
    export = [p for tier, p in pairs if tier in ('尖峰电价', '峰电价')]
    if export:
        avg = float(np.median(export))
    else:
        avg = float(np.median([p for _, p in pairs]))
    logger.info('上网结算价（参考，取尖峰/峰档中位数）：%.4f 元/kWh（命中 %d 处）',
                avg, len(pairs))
    return avg


def peak_valley_ratio():
    """从电费账单读峰谷分时比例（尖峰/峰/平/谷 四段）。

    参数
    ----
    无

    返回
    ----
    dict 或 None
        形如 {'尖峰': 10.51, '峰': 26.82, '平': 41.82, '谷': 20.84}（单位 %）；
        读不到返回 None。

    ⚠️ 仅作展示/校验，不参与计算。
       它的用途是"拿真实账单的峰谷电量占比，去检查你的电价曲线设置是否合理"：
       如果账单显示 80% 电量都在谷段，而你的模型假设用电均匀分布，
       那电费估算就会明显偏大。只信自己设的参数、不跟账单对账，是很容易踩的坑。
    """
    folder = os.path.join(cfg.REPO_DATA_DIR, '电费清单', '总账单')
    files = sorted(glob.glob(os.path.join(folder, '**', '*.pdf'), recursive=True))
    if not files:
        return None
    txt = _read_pdf_text(files[0])  # 取首个账单看比例（峰谷比例是政策性的，各期一致）
    import re
    # 账单原文类似 “峰谷分时比例为10.51%、26.82%、41.82%、20.84%”
    # （pdf 抽取偶尔把全角标点替成近似字符，用宽松写法：只抓“比例”后前 4 个浮点数）
    #   关键点：用 \D+ 表示"中间夹着任意非数字字符"，这样无论原文是 %、、
    #   还是别的什么符号，都能正确跳过，不会因为标点被替换而抓不到。
    flat = txt.replace('\n', '')     # 先把换行去掉，否则数字可能被换行拆开
    nums = re.findall(r'峰谷分时比例为\s*([\d.]+)\D+([\d.]+)\D+([\d.]+)\D+([\d.]+)', flat)
    if nums:
        a, b, c, d = nums[0]         # 元组解包，把 4 个捕获值分别赋给 4 个变量
        return {'尖峰': float(a), '峰': float(b), '平': float(c), '谷': float(d)}
    # 退化：只抓到前两段也返回（有些账单排版只给了两段）
    nums2 = re.findall(r'峰谷分时比例为\s*([\d.]+)\D+([\d.]+)', flat)
    if nums2:
        a, b = nums2[0]
        return {'尖峰': float(a), '峰': float(b)}
    return None


if __name__ == '__main__':
    # 直接运行本文件可以打印校准参考值：
    #     python -m pvflex.data.pdf_readers
    # 这是 Python 的惯例写法 —— 只有"直接跑这个文件"时才执行下面的代码，
    # 被别的文件 import 时不会执行（跟 cli.py 结尾那段的道理一样）。
    logging.basicConfig(level=logging.INFO)
    print('买电加权均价:', weighted_buy_price())
    print('上网加权价  :', weighted_sell_price())
    print('峰谷比例    :', peak_valley_ratio())
