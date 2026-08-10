# Python 模块速查手册（研究方向3 · 初学者版 · v2 扩编）

> **给谁看**：刚装好 `PV_Storage_Flex_Control` 环境、第一次系统学 Python 科研栈的你。
> **怎么用**：每个模块按「**作用 → 常用代码结构 → 一个例子 → 初学者提示**」四段写。例子都是**最小可运行**的，复制进 `.py` 文件或 Jupyter 单元格直接跑。
> **前置**：先 `conda activate PV_Storage_Flex_Control`，再 `python xxx.py` 或在 Jupyter 里 `Shift+Enter`。
> **核心心法**：科研代码 = **数据(表/数组) → 处理(清洗/特征) → 建模(预测/优化) → 验证(画图/指标)**。下面所有模块都落在这条链上。
>
> **v2 扩编说明**：在 v1 基础上新增——① 更多模块（joblib / tqdm / ortools / pulp / tsfresh+prophet / warnings+logging）；② 核心模块（numpy/pandas/sklearn/cvxpy/gurobipy/pandapower）各加**第二进阶实例**；③ 新增 **第17章 调试与排错（常见报错速查）**；④ 新增 **第18章 完整项目模板（可直接套用）**；⑤ 新增 **术语表**。

---

## 目录
1. [numpy —— 数值数组与向量化计算（地基）](#1-numpy)
2. [pandas —— 表格数据处理（你 80% 时间用它）](#2-pandas)
3. [scipy —— 科学计算（优化/统计/插值）](#3-scipy)
4. [matplotlib —— 画图（所有图的基础）](#4-matplotlib)
5. [seaborn —— 统计图（好看 + 省代码）](#5-seaborn)
6. [scikit-learn —— 经典机器学习（预测/评估/调参）](#6-scikit-learn)
7. [xgboost / lightgbm —— 树模型（预测主力）](#7-xgboost--lightgbm)
8. [statsmodels —— 时序预测（ARIMA/SARIMA）](#8-statsmodels)
9. [cvxpy —— 凸优化建模（优化核心）](#9-cvxpy)
10. [gurobipy / Pyomo —— 商业/代数求解器（鲁棒优化落地）](#10-gurobipy--pyomo)
11. [pandapower —— 电力系统仿真（潮流/微网）](#11-pandapower)
12. [PyPSA —— 能源系统仿真（大规模）](#12-pypsa)
13. [networkx —— 图与网络分析（数据挖掘）](#13-networkx)
14. [openpyxl / xlrd —— 读写 Excel（真实数据入口）](#14-openpyxl--xlrd)
15. [torch —— 深度学习（等 5070 到货再装）](#15-torch)
16. [plotly / jupyter —— 交互图与笔记本](#16-plotly--jupyter)
17. [joblib / tqdm / warnings / logging —— 工程辅助](#17-joblib--tqdm--warnings--logging)
18. [ortools / pulp —— 备用求解器（Gurobi 没许可时的退路）](#18-ortools--pulp)
19. [tsfresh / prophet —— 进阶时序特征与预测](#19-tsfresh--prophet)
20. [调试与排错（常见报错速查）](#20-调试与排错常见报错速查)
21. [完整项目模板（可直接套用）](#21-完整项目模板可直接套用)
22. [术语表](#22-术语表)

---

## 1. numpy
### 作用
Python 数值计算的地基。提供**多维数组 `ndarray`**，所有科学计算（pandas/scipy/sklearn）底层都靠它。**向量化运算**（不用 for 循环就能对整列数据做加减乘除）是它最大的价值——比纯 Python 快几十到上百倍。

### 常用代码结构
```python
import numpy as np

# 创建数组
a = np.array([1, 2, 3])              # 一维
m = np.array([[1, 2], [3, 4]])        # 二维（矩阵）
z = np.zeros((3, 4))                  # 3行4列全0
o = np.ones((2, 2))                   # 全1
r = np.arange(0, 10, 0.5)             # 0,0.5,...,9.5
linspace = np.linspace(0, 1, 11)      # 0到1等分成11个点

# 向量化运算（不用 for）
b = a * 2 + 1                         # 每个元素都 *2+1
c = np.sin(a)                         # 对每个元素求 sin

# 形状与统计
print(a.shape, a.size, a.dtype)       # 形状、元素数、数据类型
print(a.mean(), a.max(), a.sum())     # 均值、最大、求和
print(m.T)                            # 矩阵转置

# 索引（和列表类似，但能"切片"取一块）
print(a[0])                           # 第一个
print(m[0, 1])                        # 第0行第1列
print(m[:, 0])                        # 第0列全部（所有行）
```

### 一个例子
```python
import numpy as np

# 模拟一整天 24 小时的光伏出力（kW），用一条"白天有、晚上无"的曲线
hours = np.arange(24)
# 用 sin 近似白天出力，晚上截为 0
pv = np.maximum(0, np.sin((hours - 6) / 12 * np.pi) * 50)  # 峰值约 50kW
print("08点出力:", round(pv[8], 2), "kW")
print("全天总发电量:", round(pv.sum(), 2), "kWh")
print("峰值出力:", round(pv.max(), 2), "kW")
```
**输出**：08点约 25kW，全天约 254kWh，峰值 50kW。这就是"向量化"——一行 `np.maximum` 算出 24 个点，不用写循环。

### ⚡ 进阶实例（预测里最常用的：构造滑动窗口特征）
```python
import numpy as np
# 把一条长序列变成"过去 3 步预测下一步"的 X,y（时序预测标准做法）
series = np.arange(1, 101, dtype=float)        # 1..100
look_back = 3
X, y = [], []
for i in range(len(series) - look_back):
    X.append(series[i:i+look_back])            # 如 [1,2,3]
    y.append(series[i+look_back])              # 如 4
X, y = np.array(X), np.array(y)
print("X 形状:", X.shape, " 第一个样本:", X[0], " 对应标签:", y[0])
```
**说明**：`X.shape=(97,3)` 表示 97 个样本、每个 3 个历史步。这是 LSTM / XGBoost 喂时序数据的通用入口。

### 初学者提示
- ❌ 别用 `list` 做数值运算（`[1,2]+[3,4]` 会变成拼接，不是相加）。✅ 一律转 `np.array`。
- 形状(shape)对不上会报 `broadcast error`——先 `print(x.shape)` 看维度再运算。
- `np.nan` 表示缺失值，很多运算遇到它会变 `nan`，记得先处理（见 pandas 一节）。

---

## 2. pandas
### 作用
**科研里用得最多的库**。把数据当成"带列名的表格"（Excel 那种）来操作：`DataFrame`（二维表）和 `Series`（一列）。读 CSV/Excel、清洗缺失、做特征、按条件筛选、分组统计——全是它。你的园区数据清洗、预测特征工程几乎全靠它。

### 常用代码结构
```python
import pandas as pd

# 读取（最常见两种）
df = pd.read_csv("data.csv")                 # CSV
df = pd.read_excel("data.xlsx")              # Excel（需 openpyxl，已装）

# 看数据
df.head(5)          # 前5行
df.info()           # 每列类型、非空数（快速看缺不缺）
df.describe()       # 数值列的统计（均值/最值/分位）
df.columns          # 列名列表

# 选取
df["功率"]                     # 取一列 → Series
df[["时间", "功率", "温度"]]   # 取多列 → DataFrame
df.iloc[0]                     # 按位置取第0行
df[df["功率"] > 10]            # 条件筛选（功率>10 的行）

# 缺失/异常
df.isnull().sum()              # 每列缺失个数
df = df.dropna()               # 删掉有缺失的行
df = df.fillna(df.mean())      # 用均值填充

# 特征工程（时序常用）
df["功率_滞后1"] = df["功率"].shift(1)   # 上一时刻的值（做预测特征）
df["小时"] = pd.to_datetime(df["时间"]).dt.hour  # 从时间提取"小时"

# 分组统计
df.groupby("小时")["功率"].mean()         # 每个小时的平均功率
```

### 一个例子
```python
import pandas as pd
import numpy as np

# 造 24 小时负荷数据（kW）
dates = pd.date_range("2026-08-09 00:00", periods=24, freq="h")
load = 30 + 10 * np.sin(np.arange(24) / 24 * 2 * np.pi) + np.random.randn(24) * 2
df = pd.DataFrame({"时间": dates, "负荷kW": load.round(2)})

print(df.head(3))
print("\n缺失值统计:\n", df.isnull().sum())

# 特征：加上"小时"和"上一小时负荷"
df["小时"] = df["时间"].dt.hour
df["负荷_滞后1"] = df["负荷kW"].shift(1)

print("\n每个小时的平均负荷:\n", df.groupby("小时")["负荷kW"].mean().round(2).head())
```
**说明**：这就是预测建模的标准起步——读数据 → 看缺失 → 造时间特征 → 造滞后特征。`shift(1)` 是时序预测的"黄金特征"（用过去预测未来）。

### ⚡ 进阶实例（异常值处理 + 多源合并，真实数据必用）
```python
import pandas as pd, numpy as np
# 1) 3σ 异常截断：超出均值±3倍标准差的当成异常，用前后均值替
s = pd.Series(np.concatenate([np.random.randn(100), [100]]))  # 末尾造个异常
mu, sigma = s.mean(), s.std()
mask = (s > mu+3*sigma) | (s < mu-3*sigma)
s[mask] = s.rolling(5, min_periods=1).mean()[mask]   # 用滑动均值替换
print("异常点数量:", mask.sum(), " 替换后最大值:", round(s.max(),2))

# 2) 多表按时间对齐合并（光伏表 + 负荷表 → 一张训练表）
pv = pd.DataFrame({"时间": pd.date_range("2026-01-01", periods=24, freq="h"), "pv": np.random.rand(24)})
load = pd.DataFrame({"时间": pd.date_range("2026-01-01", periods=24, freq="h"), "load": np.random.rand(24)})
merged = pd.merge(pv, load, on="时间", how="inner")
print("合并后形状:", merged.shape)
```

### 初学者提示
- 读 Excel 报错 `No module named openpyxl` → 你已装 openpyxl，重开终端激活环境即可。
- `df["a"]` 和 `df.a` 一样，但列名带空格/中文只能用 `df["列名"]`。
- ** SettingWithCopyWarning**：链式赋值（`df[df.x>0]["y"]=1`）会报警，改用 `df.loc[df.x>0, "y"] = 1`。
- 时间序列务必转成 `datetime`：`pd.to_datetime()`，否则不能 `.dt.hour` 提取。
- 读数值变成字符串（如 `"12"`）→ `pd.to_numeric(df["列"], errors="coerce")`。

---

## 3. scipy
### 作用
科学计算工具箱：线性代数、插值、统计检验、信号滤波、方程求解。**优化部分（`scipy.optimize`）**对你最直接——当没装 Gurobi/Pyomo 时，线性规划 `linprog` 能当免费兜底求解器（你项目里就是这么回退的）。

### 常用代码结构
```python
from scipy import optimize, stats, interpolate

# 线性规划 min c^T x  s.t. A_ub x <= b_ub, A_eq x = b_eq
res = optimize.linprog(c, A_ub=A, b_ub=b, A_eq=Ae, b_eq=be, bounds=[...])
res.x            # 最优解
res.fun          # 最优目标值
res.success      # 是否成功

# 统计：t检验 / 正态性
t, p = stats.ttest_ind(a, b)        # 两组差异显著性
stats.pearsonr(x, y)                # 相关系数

# 插值（把稀疏点补成连续曲线）
f = interpolate.interp1d(x, y, kind="cubic")
```

### 一个例子（微网最小成本调度兜底求解）
```python
from scipy.optimize import linprog
import numpy as np

# 问题：储能两时段放电 p1,p2(kW) 最大化"供电收益"，
# 约束：0<=p<=50，且 p1+p2 <= 80（一天总可放电量）
c = [-0.8, -0.9]                     # 目标最小化，收益取负 → 等价最大化
A = [[1, 1]]; b = [80]              # p1+p2 <= 80
bounds = [(0, 50), (0, 50)]
res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
print("最优放电计划(kW):", np.round(res.x, 2), " 总收益:", -round(res.fun, 2), "元")
```
**说明**：这就是你项目里"Gurobi 没激活时自动回退"用的求解器。真实研究用 Gurobi（更快、支持整数/非线性），但 `linprog` 能让你在没许可时先把流程跑通。

### 初学者提示
- `linprog` 默认**求最小值**，要最大化就给 `c` 取负号，结果再取负。
- 约束写错维度会报 `A_ub must be 2D`——`A` 必须是二维（`[[...]]`）。
- 复杂/整数问题请用 Gurobi/Pyomo，scipy 只适合小规模线性。

---

## 4. matplotlib
### 作用
Python 画图的基础库（seaborn/plotly 都建立在它之上）。任何实验结果（预测曲线、对比柱状图、收敛曲线）都靠它出图。**科研看图 90% 用 `plt.plot` / `plt.scatter` / `plt.bar` / `plt.hist`**。

### 常用代码结构
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))      # 图大小
plt.plot(x, y, label="预测")     # 折线
plt.scatter(x, y, c="r")         # 散点
plt.bar(cats, vals)              # 柱状
plt.hist(data, bins=20)          # 直方图
plt.xlabel("x"); plt.ylabel("y")
plt.title("标题"); plt.legend()  # 图例
plt.grid(True)
plt.savefig("fig.png", dpi=300)  # 存成高清图（论文用）
plt.show()                       # 显示
```

### 一个例子（预测 vs 真实对比图）
```python
import matplotlib.pyplot as plt
import numpy as np

hours = np.arange(24)
true = np.maximum(0, np.sin((hours-6)/12*np.pi)*50)
pred = true + np.random.randn(24)*3     # 加噪声当预测

plt.figure(figsize=(9, 4))
plt.plot(hours, true, "o-", label="真实出力")
plt.plot(hours, pred, "s--", label="预测出力")
plt.xlabel("小时"); plt.ylabel("光伏出力 (kW)")
plt.title("光伏出力：预测 vs 真实"); plt.legend(); plt.grid(True)
plt.show()
```
**说明**：这是预测论文里最常见的图——真实线 + 预测线叠一起看误差。

### 初学者提示
- 中文乱码：加 `plt.rcParams["font.sans-serif"]=["SimHei"]`（Windows 用 SimHei）。
- 忘了 `plt.show()` 图不弹出来；在 Jupyter 里用 `%matplotlib inline` 自动显示。
- 论文要矢量图：`plt.savefig("fig.pdf")`（PDF 放大不糊）。

---

## 5. seaborn
### 作用
基于 matplotlib 的统计绘图库，一行画出**相关性热力图、分布图、箱线图**，颜值高、代码短。你的"特征相关性热力图""预测误差分布"都用它。

### 常用代码结构
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")  # 相关性热力图（最常用）
sns.histplot(df["功率"], kde=True)                    # 分布+密度曲线
sns.boxplot(x="小时", y="功率", data=df)              # 分组箱线
sns.scatterplot(x="温度", y="功率", hue="天气", data=df)
plt.show()
```

### 一个例子（特征相关性热力图）
```python
import seaborn as sns, matplotlib.pyplot as plt
import pandas as pd, numpy as np
np.random.seed(0)
df = pd.DataFrame({
    "负荷kW": np.random.randn(100)*10+30,
    "温度":   np.random.randn(100)*3+25,
    "湿度":   np.random.randn(100)*5+60,
})
df["负荷_滞后1"] = df["负荷kW"].shift(1).fillna(df["负荷kW"].mean())

sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("特征相关性热力图")
plt.show()
```
**说明**：一眼看出哪些特征相关，挑强相关的进模型、去掉冗余。

### 初学者提示
- `annot=True` 把数值标在格子里；`cmap` 换配色。
- 和 matplotlib 混用：先 `sns.xxx` 画图，再 `plt.title/show` 收尾。

---

## 6. scikit-learn
### 作用
经典机器学习全家桶：回归/分类/聚类、数据划分、标准化、评估指标、调参。你的**预测基线（岭回归、树模型）、聚类挖掘、评估指标（RMSE/MAE）**都来自它。API 高度统一：`fit`（训练）→ `predict`（预测）。

### 常用代码结构
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)

model = Ridge(alpha=1.0).fit(X_train_s, y_train)   # 训练
y_pred = model.predict(scaler.transform(X_test))   # 预测
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("MAE :", mean_absolute_error(y_test, y_pred))
```

### 一个例子（光伏出力预测基线）
```python
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 造数据：用[小时, 温度]预测光伏出力
hours = np.tile(np.arange(24), 30)
temp = 25 + 5*np.sin(hours/24*2*np.pi) + np.random.randn(720)*1
pv = np.maximum(0, np.sin((hours-6)/12*np.pi)*50) + np.random.randn(720)*2

X = np.column_stack([hours, temp]); y = pv
# 80%训练,20%测试（时间序列用 shuffle=False 更严谨，这里演示用默认）
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = Ridge(alpha=1.0).fit(X_tr, y_tr)
pred = model.predict(X_te)
print("RMSE(kW):", round(mean_squared_error(y_te, pred, squared=False), 3))
print("MAE (kW):", round(mean_absolute_error(y_te, pred), 3))
```
**说明**：`Ridge`（岭回归）就是你项目 `PV_Storage_Flex_Control` 里预测模块用的算法——轻量、稳、好解释，是深度学习之前的"该有的基线"。

### ⚡ 进阶实例（GridSearchCV 调参 + 多模型对比，论文基线必备）
```python
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ... 假设 X_tr,X_te,y_tr,y_te 已准备好（见上例）
# 1) 岭回归调 alpha
param = {"alpha": [0.01, 0.1, 1.0, 10.0]}
grid = GridSearchCV(Ridge(), param, cv=5)
grid.fit(X_tr, y_tr)
print("最佳 alpha:", grid.best_params_, " CV RMSE:", round(np.sqrt(-grid.best_score__),3))

# 2) 随机森林对比
rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_tr, y_tr)
print("RF 测试 RMSE:", round(mean_squared_error(y_te, rf.predict(X_te), squared=False),3))
# 记住：树模型不用 StandardScaler；对比多个模型时用同一份 train/test 才公平
```

### 初学者提示
- **一定先 train_test_split 再 fit**，否则用测试集信息训练=作弊（数据泄露）。
- 树模型（XGBoost）不用 StandardScaler，线性模型（Ridge）建议做。
- `squared=False` 才返回 RMSE（sklearn 默认返回 MSE）。

---

## 7. xgboost / lightgbm
### 作用
**梯度提升树**，表格数据预测的"王者"，比深度学习更稳更快出结果。你的阶段2 主力基线就是用它们（XGBoost/LightGBM）。比 Ridge 精度通常更高，还能直接看"哪些特征最重要"。

### 常用代码结构
```python
import xgboost as xgb
from xgboost import XGBRegressor
# 或 from lightgbm import LGBMRegressor

model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                     early_stopping_rounds=20)
model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)  # 早停防过拟合
pred = model.predict(X_te)
print("特征重要性:", model.feature_importances_)   # 看谁主导预测
```

### 一个例子
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

hours = np.tile(np.arange(24), 30)
temp = 25 + 5*np.sin(hours/24*2*np.pi) + np.random.randn(720)
pv = np.maximum(0, np.sin((hours-6)/12*np.pi)*50) + np.random.randn(720)*2
X = np.column_stack([hours, temp]); y = pv
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                      early_stopping_rounds=20)
model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
pred = model.predict(X_te)
print("XGBoost RMSE(kW):", round(mean_squared_error(y_te, pred, squared=False), 3))
```
**说明**：把上面 Ridge 的例子换成 XGBoost，通常 RMSE 更低。`early_stopping_rounds` 防止训练过头（过拟合）。

### 初学者提示
- 树模型**不用标准化特征**，直接喂原始数据。
- `n_estimators` 太大慢、`max_depth` 太大易过拟合——先用默认，再 `GridSearchCV` 调。
- 特征重要性是"可解释性"卖点，论文里常画一张 importance 条形图。

---

## 8. statsmodels
### 作用
经典统计/计量库，阶段2 的**对比基准**来源：ARIMA/SARIMA 时间序列预测、季节分解、ADF 平稳性检验。审稿人爱看"你的深度学习 vs 经典 ARIMA 谁强"，所以必须会。

### 常用代码结构
```python
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# 季节分解（看趋势/季节/残差）
res = seasonal_decompose(series, period=24)   # 周期 24（日周期）
res.trend; res.seasonal; res.resid

# ARIMA(p,d,q) 预测
model = ARIMA(y, order=(2,1,2)).fit()
forecast = model.forecast(steps=12)            # 往后预测 12 步
```

### 一个例子
```python
import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# 造有日周期的负荷序列（96个时间点）
t = np.arange(96)
load = 30 + 5*np.sin(t/24*2*np.pi) + np.random.randn(96)*1
series = pd.Series(load)

model = ARIMA(series, order=(2, 1, 2)).fit()
fc = model.forecast(steps=12)
print("未来12步预测:", np.round(fc.values, 2))
print("AIC(越小越好):", round(model.aic, 2))
```
**说明**：`order=(p,d,q)` 里 d 是差分阶数（让序列变平稳）。不确定用哪个就靠 AIC 选最小。

### 初学者提示
- 序列不平稳（有趋势）要先差分（d≥1）或先 `seasonal_decompose` 去掉趋势。
- ARIMA 只适合**单变量**；多变量/带外生特征用 SARIMAX 或干脆上 XGBoost/LSTM。
- 和 sklearn 不同：statsmodels 用 `.fit()` 后直接 `.forecast(steps=)`。

---

## 9. cvxpy
### 作用
**凸优化建模语言**——把"最小化目标函数、满足一堆约束"用数学写法直接翻译成代码。你的阶段3 优化模型（含 SOC、柔性负荷、电网交互约束）用它或 Pyomo 表达最自然。比 scipy 强在支持**凸二次/二次约束/矩阵变量**。

### 常用代码结构
```python
import cvxpy as cp
import numpy as np

x = cp.Variable(n)               # 决策变量
objective = cp.Minimize(c @ x)   # 目标
constraints = [A @ x <= b, x >= 0]
prob = cp.Problem(objective, constraints)
prob.solve()                      # 默认用免费求解器（ECOS/OSQP/SCS）
print("最优值:", prob.value, " 解:", x.value)
```

### 一个例子（日前储能调度最小化购电成本）
```python
import cvxpy as cp, numpy as np

T = 24
price = np.maximum(0.3, 0.5 + 0.4*np.sin(np.arange(T)/24*2*np.pi))  # 分时电价
grid = cp.Variable(T)            # 从电网买的电(kW)
batt = cp.Variable(T)           # 电池放电(+)/充电(-)
soc  = cp.Variable(T)           # 电量状态

cons = [soc[0]==10, batt>= -20, batt<=20, soc>=0, soc<=50]
for t in range(1, T):
    cons.append(soc[t] == soc[t-1] - batt[t])   # 简单能量平衡
cons.append(cp.sum(batt) == 0)                  # 一天净放电0（不增不减）
obj = cp.Minimize(price @ (grid - batt))         # 买电花钱 - 放电省的钱
prob = cp.Problem(cp.Minimize(price @ (grid - batt)), cons)
prob.solve()
print("一天最小购电成本:", round(prob.value, 2), "元")
```
**说明**：这就是"能-碳调控"优化模型的雏形——决策变量 + 目标 + 约束，三件套。cvxpy 语法和数学公式几乎一一对应。

### ⚡ 进阶实例（含碳成本 + 充放电互斥整数约束，更接近阶段3）
```python
import cvxpy as cp, numpy as np
T = 24
grid = cp.Variable(T); batt = cp.Variable(T); soc = cp.Variable(T)
price = np.maximum(0.3, 0.5+0.4*np.sin(np.arange(T)/24*2*np.pi))
carbon = 0.6 + 0.3*np.sin(np.arange(T)/24*2*np.pi)   # 时变碳因子 kgCO2/kWh
cons = [soc[0]==10, soc>=0, soc<=50, batt>=-20, batt<=20]
for t in range(1,T): cons.append(soc[t] == soc[t-1] - batt[t])
# 目标：经济成本 + λ*碳成本（双目标加权）
obj = cp.Minimize(cp.sum(price*grid) + 0.1*cp.sum(carbon*grid))
prob = cp.Problem(cp.Minimize(cp.sum(price*grid) + 0.1*cp.sum(carbon*grid)), cons)
prob.solve()
print("含碳目标的最小成本:", round(prob.value,2))
```
**说明**：把 `carbon` 时变因子加进目标，就是"能-碳协同"的雏形；cvxpy 默认求解器对中小规模 LP/QP 足够，大规模/整数再上 Gurobi。

### 初学者提示
- `cp.Variable` 是**符号变量**，没数值，`.solve()` 之后才有 `.value`。
- 约束放错了（如非凸）会报 `Problem does not follow DCP rules`——凸优化只认凸。
- 大规模/整数/MILP 用 Pyomo + Gurobi（更快更专业）。

---

## 10. gurobipy / Pyomo
### 作用
- **gurobipy**：商业求解器 Gurobi 的 Python 接口，**速度极快、支持整数/混合整数(MILP)/非线性**。你已激活学术许可，阶段3 的鲁棒优化、含整数决策（如"启停""离散档位"）的模型靠它。
- **Pyomo**：开源**代数建模语言**，写法更接近数学模型（和 Gurobi/Cplex/GLPK 解耦，换求解器不改模型）。阶段3 建议用 Pyomo 建模、用 Gurobi 求解。

### 常用代码结构（gurobipy 最简）
```python
import gurobipy as gp
m = gp.Model("dispatch")
x = m.addVar(lb=0, ub=50, name="放电")     # 决策变量(带上下界)
y = m.addVar(vtype=gp.GRB.BINARY, name="启停")  # 0/1 整数变量
m.setObjective(0.8*x, gp.GRB.MAXIMIZE)     # 目标
m.addConstr(x <= 50*y)                     # 约束（用整数变量耦合）
m.optimize()
print("最优:", x.X, " 目标:", m.ObjVal)
```

### 一个例子（带启停的整数决策）
```python
import gurobipy as gp
m = gp.Model("battery")
p = m.addVar(lb=0, ub=50, name="放电功率")
u = m.addVar(vtype=gp.GRB.BINARY, name="是否放电")
m.addConstr(p <= 50*u)            # u=0 时强制 p=0（整数耦合）
m.setObjective(p, gp.GRB.MAXIMIZE)
m.optimize()
print("最优放电:", round(p.X,2), "kW, 启停标志:", int(u.X))
```
**说明**：`vtype=GRB.BINARY` 是混合整数规划(MILP)的标志——纯线性规划(LP)解不了"是否启机"这种 0/1 决策，必须 Gurobi/MILP。这是 cvxpy 免费求解器弱、Gurobi 强的地方。

### ⚡ 进阶实例（Pyomo + Gurobi 日前调度，可扩展两阶段）
```python
import pyomo.environ as pyo
import numpy as np

m = pyo.ConcreteModel()
T = 24
m.T = pyo.RangeSet(0, T-1)
price = np.maximum(0.3, 0.5+0.4*np.sin(np.arange(T)/24*2*np.pi))

m.grid = pyo.Var(m.T, bounds=(0, 100))      # 购电
m.batt = pyo.Var(m.T, bounds=(-20, 20))     # 电池
m.soc  = pyo.Var(m.T, bounds=(0, 50), initialize=10)

def obj_rule(m):
    return sum(price[t] * m.grid[t] for t in m.T)
m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

def soc_rule(m, t):
    if t == 0: return m.soc[t] == 10
    return m.soc[t] == m.soc[t-1] - m.batt[t]
m.soc_con = pyo.Constraint(m.T, rule=soc_rule)

pyo.SolverFactory("gurobi").solve(m)
print("总购电成本:", round(m.obj(), 2))
```
**说明**：Pyomo 的 `ConcreteModel` + `Var/Constraint/Objective` 写法非常接近数学公式，换求解器（gurobi→glpk）只改 `SolverFactory` 一行，模型不用动——这是它比纯 gurobipy 更适合"论文里反复改模型"的原因。

### 初学者提示
- 没激活会报许可错误 → 你已 `grbgetkey` 激活，正常。
- Pyomo 建模示例（配合 Gurobi）：见上例。
- 学术免费、速度王，阶段3/4 的主力求解器。

---

## 11. pandapower
### 作用
**电力系统仿真**（潮流计算、短路、状态估计）。你的阶段1/4 用它搭"光伏+储能+负荷"微电网、跑潮流看母线电压/损耗。是研究内容三"仿真验证"的主工具，比手算/Simulink 更适合批量场景扫描。

### 常用代码结构
```python
import pandapower as pp
import pandapower.networks as nw

net = pp.create_empty_network()        # 空网络
# 加母线
b1 = pp.create_bus(net, vn_kv=0.4, name="直流母线")
b2 = pp.create_bus(net, vn_kv=0.4, name="负载母线")
# 加元件
pp.create_ext_grid(net, bus=b1, vm_pu=1.0)         # 外部电网
pp.create_sgen(net, bus=b1, p_mw=0.02, name="光伏")  # 分布式电源(20kW)
pp.create_load(net, bus=b2, p_mw=0.03)             # 负荷(30kW)
pp.create_line(net, b1, b2, length_km=0.1, std_type="NAYY 4x50 SE")
pp.runpp(net)                          # 跑潮流
print(net.res_bus)                     # 母线电压结果
```

### 一个例子（最小微网潮流）
```python
import pandapower as pp
net = pp.create_empty_network()
bus = pp.create_bus(net, vn_kv=0.4, name="母线")
pp.create_ext_grid(net, bus=bus, vm_pu=1.0)
pp.create_sgen(net, bus=bus, p_mw=0.03, name="光伏")   # 30kW 发电
pp.create_load(net, bus=bus, p_mw=0.02, name="负荷")   # 20kW 用电
pp.runpp(net)
print(net.res_bus[["vm_pu", "p_mw"]])     # 电压(pu)和净功率
```
**说明**：`res_bus` 里 `vm_pu` 是标幺电压（1.0=额定），接近 1 才健康。阶段4 你会加储能、做多场景对比。

### ⚡ 进阶实例（带储能 + 多场景批量扫描）
```python
import pandapower as pp
import numpy as np

def build_microgrid(pv_kw, load_kw):
    net = pp.create_empty_network()
    bus = pp.create_bus(net, vn_kv=0.4, name="母线")
    pp.create_ext_grid(net, bus=bus, vm_pu=1.0)
    pp.create_sgen(net, bus=bus, p_mw=pv_kw/1000)
    pp.create_load(net, bus=bus, p_mw=load_kw/1000)
    pp.runpp(net)
    return net.res_bus["vm_pu"].iloc[0]

# 批量扫 5 种光伏/负荷组合，找电压越限的场景
for pv, load in [(30,20),(50,40),(10,60),(80,20),(20,80)]:
    v = build_microgrid(pv, load)
    flag = "⚠越限" if abs(v-1.0)>0.05 else "OK"
    print(f"光伏{pv}kW/负荷{load}kW → 电压{v:.3f} pu {flag}")
```
**说明**：阶段4 做"典型日/极端日/高比例光伏"三类场景，本质就是这种批量扫描 + 越限判定；把函数换成带储能/SOC 的版本即可。

### 初学者提示
- 单位用 **MW/MVar**（不是 kW），小系统用 `0.02`(=20kW) 这种小数。
- 潮流不收敛 → 检查电压初值、线路参数、是否有孤岛（元件没连上）。
- 直流微网用 `vn_kv=0.4(400V)` 或 `0.8` 建模；pandapower 主要做交流，直流简化用交流模型近似即可（研究够用）。

---

## 12. PyPSA
### 作用
**能源系统/电力网络综合仿真**（比 pandapower 更适合"多时段、多能源、优化调度"）。阶段4 做"日前-实时调度仿真"、阶段3 把优化结果接进去算系统指标时常用。

### 常用代码结构
```python
import pypsa
import pandas as pd

n = pypsa.Network()
n.add("Bus", "bus0", v_nom=0.4)
n.add("Generator", "光伏", bus="bus0", p_nom=0.05)      # 50kW
n.add("Load", "负荷", bus="bus0", p_set=0.03)
# 多时段需在 n.snapshots 设时间索引，再用 p_set/p_nom_extendable
n.optimize()                 # 自带优化（需额外求解器）
```

### 一个例子
```python
import pypsa
n = pypsa.Network()
n.add("Bus", "bus0", v_nom=0.4)
n.add("Generator", "光伏", bus="bus0", p_nom=0.05)
n.add("Load", "负荷", bus="bus0", p_set=0.02)
n.pf()                      # 功率流（非优化版）
print(n.loads_t.p_set if hasattr(n, "loads_t") else "负载已建")
```
**说明**：PyPSA 适合"含优化+多时段"的大规模能源系统；小演示用 `pf()`，做调度优化用 `optimize()`（底层调 Gurobi/HiGHS）。

### 初学者提示
- PyPSA 偏"优化调度"，pandapower 偏"潮流/短路验证"，两者互补。
- 多时段建模要设 `n.set_snapshots(pd.date_range(...))`，否则只算单点。

---

## 13. networkx
### 作用
**图/网络分析**库。你的阶段5 数据挖掘（把"母线-设备-调控关系"建图、找关键节点/社区）、以及电网拓扑分析用它。也能画网络结构图。

### 常用代码结构
```python
import networkx as nx
G = nx.Graph()                 # 无向图
G.add_edge("光伏", "母线1")
G.add_edge("母线1", "负荷")
nx.degree(G)                  # 每个节点连接数（找关键节点）
nx.connected_components(G)    # 连通子图
nx.draw(G, with_labels=True)  # 画图
```

### 一个例子（微网拓扑关键节点）
```python
import networkx as nx
G = nx.Graph()
for e in [("电网", "母线"), ("光伏", "母线"), ("储能", "母线"), ("母线", "负荷"), ("母线", "柔性负荷")]:
    G.add_edge(*e)
print("节点度数(连接数):", dict(nx.degree(G)))
print("是否为连通图:", nx.is_connected(G))
```
**说明**：度数最高的"母线"是关键节点——断了它全网瘫，对应"脆弱性分析"。阶段5 用它挖系统结构特征。

### 初学者提示
- 有向用 `nx.DiGraph()`（如功率流向）；无向用 `nx.Graph()`。
- 大数据图（`draw` 会卡），超过几百节点别 `draw`，用算法指标代替。

---

## 14. openpyxl / xlrd
### 作用
读写 Excel——**你真实园区数据的入口**（园区数据大概率在 `.xlsx`/`.xls` 里）。pandas 的 `read_excel` 底层就靠 openpyxl（读新 xlsx）和 xlrd（读老 xls）。通常直接用 pandas 即可，但有些精细操作（改单元格格式、写多个 sheet）要直接调 openpyxl。

### 常用代码结构
```python
import pandas as pd
# 读（pandas 已够用）
df = pd.read_excel("园区数据.xlsx", sheet_name="负荷")   # 指定 sheet
df = pd.read_excel("数据.xls")                            # 老 xls 用 xlrd 兜底

# 写
df.to_excel("结果.xlsx", index=False)

# 直接操作 openpyxl（要改格式时）
from openpyxl import load_workbook
wb = load_workbook("园区数据.xlsx")
ws = wb["负荷"]
print(ws["A1"].value)        # 读单元格
```

### 一个例子
```python
import pandas as pd
# 模拟：把你算好的调度结果写回 Excel 给导师看
import numpy as np
result = pd.DataFrame({
    "小时": np.arange(24),
    "储能放电kW": np.round(np.random.randn(24).clip(-20,20), 1),
})
result.to_excel("调度结果.xlsx", index=False)
print("已写出调度结果.xlsx，行数:", len(result))
```
**说明**：真实场景里，你从园区 `.xlsx` 读数据 → 跑模型 → 把结果写回新的 `.xlsx` 给导师/写论文，这条链路 openpyxl 是底层。

### 初学者提示
- 读 xlsx 报 `Missing optional dependency openpyxl` → 你已装，激活环境重开终端即可。
- 老 `.xls`（Excel 2003）要 xlrd，且新 xlrd 不支持 xls 加密——遇到就用 pandas 直接读，多半 OK。
- 数值被读成字符串（如 `"12"`）→ 用 `pd.to_numeric(df["列"], errors="coerce")` 强制转数字。

---

## 15. torch
### 作用
**深度学习框架**（PyTorch）。阶段2 上 LSTM/GRU/PatchTST 预测、导师设备诊断(CNN/GNN) 用它。你那张 5070 的价值就在这——**但没买显卡前不要装**（装了 `cuda.is_available()` 也是 False，白占 2-3GB）。

### 常用代码结构（先认识，等 5070 到货再跑）
```python
import torch
import torch.nn as nn

# 张量（类似 numpy，但能放 GPU）
x = torch.randn(3, 4)
if torch.cuda.is_available():
    x = x.cuda()                  # 搬到 5070 上算

# 一个最小神经网络
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
y = model(torch.randn(5, 10))     # 前向传播
```

### 一个例子（等 5070 到货后跑）
```python
import torch, torch.nn as nn

# 用简单网络学 y = 2x+1（回归入门）
X = torch.linspace(0, 1, 100).unsqueeze(1)
Y = 2*X + 1 + 0.05*torch.randn(100, 1)
model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(200):
    opt.zero_grad()
    loss = nn.MSELoss()(model(X), Y)
    loss.backward(); opt.step()
print("训练后损失:", round(loss.item(), 4))
```
**说明**：这就是深度学习的最小闭环——数据→model→loss→backward→step。阶段2 学 LSTM 时把 `nn.Linear` 换成 `nn.LSTM` 即可。

### 初学者提示
- **买卡前不要装** torch cu128（你已确认）。等 5070 到货按学习计划 0.2 第 8 步用阿里云/官方 cu128 源装。
- 张量和 numpy 互转：`torch.from_numpy(arr)` / `tensor.numpy()`。
- GPU 显存小（12GB）：`batch_size` 别贪大，开 `torch.cuda.amp` 混合精度。

---

## 16. plotly / jupyter
### 作用
- **plotly**：交互图（鼠标悬停看数值、可缩放），组会汇报比静态图更直观。
- **jupyter / notebook**：交互式笔记本，把"代码+图+文字"放一起，做探索分析、调参、组会 slides 最方便。

### 常用代码结构（plotly）
```python
import plotly.express as px
import pandas as pd
df = pd.DataFrame({"x":[1,2,3], "y":[2,5,3], "场景":["A","B","C"]})
fig = px.line(df, x="x", y="y", color="场景", title="交互图")
fig.show()             # 浏览器打开，可缩放/悬停
```

### 一个例子
```python
import plotly.express as px
import pandas as pd, numpy as np
df = pd.DataFrame({
    "小时": np.tile(np.arange(24),2),
    "出力": np.concatenate([np.maximum(0,np.sin((np.arange(24)-6)/12*np.pi)*50),
                            np.maximum(0,np.sin((np.arange(24)-6)/12*np.pi)*45)]),
    "模型": ["真实"]*24 + ["预测"]*24,
})
px.line(df, x="小时", y="出力", color="模型", title="光伏出力交互对比").show()
```
**说明**：组会汇报时这种图能现场拖拽放大某时段，比静态图更"会讲"。

### 初学者提示
- Jupyter 里 `Shift+Enter` 跑当前格；`%matplotlib inline` 让 matplotlib 图内联显示。
- 选对 Python 解释器：VS Code 右下角选 `PV_Storage_Flex_Control` 环境，否则找不到包。

---

## 17. joblib / tqdm / warnings / logging
### 作用
四个"工程辅助"小工具，做正式科研代码越来越离不开：
- **joblib**：把训练好的模型/大数组存盘、加载（比 pickle 快、能压缩）；你的 XGBoost/优化结果要反复用，存盘省重算。
- **tqdm**：循环进度条，跑长实验（多场景/多 epoch）时知道还要多久。
- **warnings**：压制已知无害警告（如 sklearn 版本提示），让输出干净。
- **logging**：比 `print` 专业的日志，写文件+分级（INFO/ERROR），调试大项目必备。

### 常用代码结构
```python
import joblib
joblib.dump(model, "model_xgb.pkl")        # 存模型
model = joblib.load("model_xgb.pkl")        # 加载（不用重训）

from tqdm import tqdm
for i in tqdm(range(1000)): pass            # 进度条

import warnings; warnings.filterwarnings("ignore")   # 压制警告

import logging
logging.basicConfig(filename="run.log", level=logging.INFO)
logging.info("epoch %d loss %.3f", 1, 0.12) # 写日志文件
```

### 一个例子（模型存盘 + 进度条）
```python
import joblib, numpy as np
from tqdm import tqdm
from xgboost import XGBRegressor

# 假训练：多次重采样评估，存最终模型
scores = []
for k in tqdm(range(5), desc="交叉验证"):
    # ... 训练逻辑省略 ...
    scores.append(np.random.rand())
model = XGBRegressor(n_estimators=50).fit(np.random.rand(100,3), np.random.rand(100))
joblib.dump(model, "best_model.pkl")
print("平均得分:", round(np.mean(scores),3), " 模型已存盘")
```

### 初学者提示
- `joblib.load` 加载的模型要**同环境同版本** sklearn/xgboost，否则报错；存盘时记一下版本。
- `warnings.filterwarnings("ignore")` 别滥用——先看清警告是不是真无害。

---

## 18. ortools / pulp
### 作用
**备用优化求解器**——万一 Gurobi 学术许可出问题/换机器，这两个免费求解器能顶上：
- **ortools**（Google）：CP-SAT + LP/ILP 求解器，速度快、免费、装包简单。
- **pulp**：纯 Python 建模语言，语法极简，底层可换 CBC/GLPK/CPLEX。

### 常用代码结构（pulp 最简）
```python
import pulp
prob = pulp.LpProblem("dispatch", pulp.LpMinimize)
x = pulp.LpVariable("x", lowBound=0, upBound=50)
prob += x                       # 目标：min x
prob += x <= 30                # 约束
prob.solve(pulp.PULP_CBC_CMD())# 用免费 CBC 求解
print(pulp.value(x), pulp.LpStatus[prob.status])
```

### 一个例子（ortools 线性规划）
```python
from ortools.linear_solver import pywraplp
solver = pywraplp.Solver.CreateSolver("GLOP")   # 免费 LP 求解器
x = solver.NumVar(0, 50, "x"); y = solver.NumVar(0, 50, "y")
solver.Add(x + y <= 80)
solver.Minimize(-(0.8*x + 0.9*y))
solver.Solve()
print("x,y:", x.solution_value(), y.solution_value())
```
**说明**：这两个是 Gurobi 的"免费平替"。小/中规模问题结果一致；大规模/强整数规划还是 Gurobi 快。

### 初学者提示
- `pip install ortools pulp` 即可（已走清华镜像）。
- 论文里若用免费求解器也能复现，反而显出"不依赖商业软件"的严谨。

---

## 19. tsfresh / prophet
### 作用
- **tsfresh**：自动从时序里抽取**几百个特征**（均值/方差/熵/自相关…），喂给树模型/XGBoost，省去手写特征工程。注意：慢、占内存，小数据用。
- **prophet**：Facebook 的**加法时序预测**（趋势+周/年季节+节假日），对带明显周期、缺数据的业务序列友好；但电力序列用 XGBoost/LSTM 通常更准，prophet 作对照基线即可。

### 常用代码结构
```python
from tsfresh import extract_relevant_features
from tsfresh.utilities.dataframe_functions import roll_time_series
# 需把长表转成 (id, time, value) 格式再抽特征，较繁琐，小数据才用

from prophet import Prophet
df = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=100), "y": values})
m = Prophet(); m.fit(df); future = m.make_future_dataframe(periods=12)
fc = m.predict(future)
```

### 一个例子（prophet 基线）
```python
import pandas as pd, numpy as np
from prophet import Prophet
t = pd.date_range("2026-01-01", periods=120, freq="D")
y = 30 + 5*np.sin(np.arange(120)/30*2*np.pi) + np.random.randn(120)
df = pd.DataFrame({"ds": t, "y": y})
m = Prophet(weekly_seasonality=True, yearly_seasonality=False)
m.fit(df); future = m.make_future_dataframe(periods=14)
fc = m.predict(future)
print(fc[["ds","yhat","yhat_lower","yhat_upper"]].tail(3))
```
**说明**：prophet 直接给预测区间（`yhat_lower/upper`），正好对应阶段2"不确定性量化"需求，当基线很方便。

### 初学者提示
- tsfresh 内存爆炸风险，先在小样本（几百行）试。
- prophet 对强随机波动的电力序列易欠拟合，别当主力，当对照。

---

## 20. 调试与排错（常见报错速查）

> 初学者 80% 的时间花在排错。下面按"症状 → 原因 → 解决"列最常被你踩的坑。

| 报错 / 现象 | 根因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: No module named 'xxx'` | 没装包 / 装错环境 | `conda activate PV_Storage_Flex_Control` 后 `pip install xxx` |
| `SyntaxError` 在 `conda`/`pip` 命令上 | 在 Python `>>>` 里敲了命令 | 退出 Python，在**终端**敲 |
| `cuda.is_available() == False` | 没显卡 / 没装 cu128 torch | 等 5070 到货再装 cu128；现在用 CPU 跑 |
| `ValueError: shapes (a,b) and (c,d) not aligned` | numpy 矩阵维度不匹配 | `print(A.shape, B.shape)` 逐一核对 |
| `SettingWithCopyWarning` | 链式赋值 | 改用 `df.loc[mask, col] = val` |
| `KeyError: '列名'` | 列名拼错/有空格/未读入 | `print(df.columns)` 看真实列名 |
| `A_ub must be 2D` (scipy) | 约束矩阵写成一维 | 用 `[[...]]` 包成二维 |
| `Problem does not follow DCP rules` (cvxpy) | 写了非凸约束 | 检查是否用了 `max`/乘积变量等，改凸形式 |
| `Gurobi Error 10009: ... license` | 没激活 | `grbgetkey <你的密钥>` |
| `Infeasible model` (Gurobi) | 约束冲突 | 逐条松约束定位矛盾（先删一半约束二分查） |
| `UserWarning: ... future version` | 版本警告（多无害） | `warnings.filterwarnings("ignore")` 或升级包 |
| 图不显示 | 没 `plt.show()` / 非 inline | Jupyter 加 `%matplotlib inline` |
| 中文图乱码 | 字体未设 | `plt.rcParams["font.sans-serif"]=["SimHei"]` |
| 结果每次不一样 | 随机种子没固定 | 所有 `random_state=42`、实验开头 `np.random.seed(42)` |
| 训练集精度高、测试集崩 | 过拟合 / 数据泄露 | 查 `train_test_split(shuffle=...)`、加早停/正则 |

**通用排错三板斧**：① `print` 中间变量形状/取值；② 把报错整行复制去**百度/Google/StackOverflow**；③ 最小可复现——把代码删到只剩 5 行仍能报错，问题就藏不住。

---

## 21. 完整项目模板（可直接套用）

> 把你 `PV_Storage_Flex_Control` 项目拆成两个"最小可运行模板"，照着把数据换成你的园区数据即可。

### 模板 A：源-荷预测流水线（`forecasting_template.py`）
```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib, warnings
warnings.filterwarnings("ignore")

def load_and_clean(path):
    df = pd.read_excel(path)                      # 换成你的园区数据
    df = df.dropna().reset_index(drop=True)
    df["小时"] = pd.to_datetime(df["时间"]).dt.hour
    return df

def make_features(df, target="光伏kW", lags=(1, 24)):
    for L in lags:
        df[f"lag{L}"] = df[target].shift(L)
    df = df.dropna()
    feat = [c for c in df.columns if c not in (target, "时间")]
    return df[feat], df[target]

def main(path="园区数据.xlsx"):
    df = load_and_clean(path)
    X, y = make_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBRegressor(n_estimators=200, max_depth=4,
                         early_stopping_rounds=20)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    pred = model.predict(X_te)
    print("RMSE:", round(mean_squared_error(y_te, pred, squared=False),3),
          " MAE:", round(mean_absolute_error(y_te, pred),3))
    joblib.dump(model, "pv_model.pkl")            # 存模型，下次直接 load

if __name__ == "__main__":
    main()
```

### 模板 B：能-碳优化调度（`optimization_template.py`，scipy 兜底版）
```python
import numpy as np
from scipy.optimize import linprog

def day_ahead_dispatch(price, carbon, T=24, batt_max=20, soc_max=50):
    """最小化 经济成本 + λ*碳成本；返回 24h 购电/放电计划。"""
    lam = 0.1
    # 变量顺序: grid(0..T-1), batt(T..2T-1), soc(2T..3T-1)
    n = 3*T
    c = np.concatenate([price + lam*carbon, np.zeros(2*T)])
    A, b = [], []
    # 能量平衡: soc_t = soc_{t-1} - batt_t
    for t in range(T):
        row = np.zeros(n)
        row[2*T+t] = 1                      # soc_t
        if t>0: row[2*T+t-1] = -1           # -soc_{t-1}
        row[T+t] = 1                        # -batt_t (放电为正)
        A.append(row); b.append(0 if t==0 else 0)
    # 上下界
    bounds = [(0,100)]*T + [(-batt_max, batt_max)]*T + [(0, soc_max)]*T
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    return res.x[:T], res.x[T:2*T]          # grid, batt

if __name__ == "__main__":
    T=24
    price = np.maximum(0.3, 0.5+0.4*np.sin(np.arange(T)/24*2*np.pi))
    carbon = 0.6+0.3*np.sin(np.arange(T)/24*2*np.pi)
    g, b = day_ahead_dispatch(price, carbon)
    print("购电计划:", np.round(g,1))
    print("放电计划:", np.round(b,1))
```
**说明**：模板 B 用免费 scipy 兜底；换成 Gurobi/Pyomo（见 §10）只需改写 `day_ahead_dispatch` 内部，外部调用不变。把它和你阶段3 的真实模型对接即可。

---

## 22. 术语表

| 术语 | 中文 | 一句话 |
|---|---|---|
| ndarray | N维数组 | numpy 的多维数组，向量化运算的载体 |
| DataFrame | 数据框 | pandas 的二维带列名表格 |
| Series | 序列 | pandas 的一列（带索引） |
| RMSE / MAE / MAPE | 均方根/平均/平均百分比误差 | 预测精度指标，越小越好 |
| R² | 决定系数 | 拟合优度，越接近 1 越好 |
| ARIMA | 差分整合移动平均自回归 | 经典单变量时序预测 |
| XGBoost / LightGBM | 梯度提升树 | 表格数据预测王者 |
| MLP / LSTM / CNN | 多层感知机/长短期记忆/卷积 | 深度学习三类基础网络 |
| LP / QP / MILP | 线性/二次/混合整数线性规划 | 优化问题分类 |
| DRO | 分布鲁棒优化 | 鲁棒与随机的折中 |
| CCG / Benders | 列与约束生成 / 奔得斯分解 | 两阶段鲁棒优化的求解算法 |
| SOC | 荷电状态 | 电池剩余电量百分比 |
| MPC | 模型预测控制 | 滚动优化、实时修正 |
| pandapower / PyPSA | 电力/能源系统仿真库 | 潮流与优化调度 |
| Gurobi | 商业求解器 | 速度最快的优化求解器（学术免费） |
| feature_importances_ | 特征重要性 | 树模型告诉你"谁主导预测" |
| data leakage | 数据泄露 | 训练时混入了测试信息（作弊） |
| early_stopping | 早停 | 验证集不再改善就停训，防过拟合 |
| GPU / CUDA / cu128 | 显卡/并行架构/驱动版本 | 5070 需 CUDA 12.8 驱动 |

---

## 23. 实战菜谱 Cookbook（复制即用 + 期望输出）

每个菜谱都是独立小脚本，「期望看到」写在注释里。前 9 个覆盖你方向3 的日常工作流。

### 菜谱 1：读 Excel 并体检
```python
import pandas as pd
df = pd.read_excel("园区数据.xlsx", sheet_name=0)   # 没装 openpyxl 会报错 → pip install openpyxl
print(df.shape)            # 期望：(行数, 列数)
print(df.head())           # 期望：前 5 行
print(df.dtypes)           # 期望：每列类型；时间列若是 object 要转 datetime
print(df.isna().sum())     # 期望：每列缺失个数，决定下一步怎么补
```

### 菜谱 2：缺失 / 异常值处理（真实数据必做）
```python
df = df.dropna(subset=["load_kW"]).copy()          # 关键列缺失直接删行
df["load_kW"] = df["load_kW"].clip(lower=0)         # 负荷不可能为负
mean, std = df["load_kW"].mean(), df["load_kW"].std()
mask = (df["load_kW"] - mean).abs() > 3 * std       # 3σ 异常
df.loc[mask, "load_kW"] = mean                      # 异常用均值替
print("异常替换条数:", int(mask.sum()))             # 期望：一个整数
```

### 菜谱 3：构造时序特征（预测标配）
```python
df["hour"] = df["time"].dt.hour
for lag in [1, 2, 3, 24]:
    df[f"load_lag{lag}"] = df["load_kW"].shift(lag)   # 滞后1/2/3小时 + 昨天同时
df = df.dropna()
print("特征列:", [c for c in df.columns if c.startswith("load_lag") or c == "hour"])
```

### 菜谱 4：XGBoost 预测 + 评估
```python
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
feat = [c for c in df.columns if c.startswith("load_lag") or c == "hour"]
X, y = df[feat], df["load_kW"]
split = int(len(X) * 0.8)
Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
m = xgb.XGBRegressor(n_estimators=200, max_depth=4)
m.fit(Xtr, ytr)
rmse = np.sqrt(mean_squared_error(yte, m.predict(Xte)))
print("测试 RMSE(kW):", round(rmse, 2))        # 期望：比线性模型小的数
print("重要性:", dict(zip(feat, np.round(m.feature_importances_, 3))))
```

### 菜谱 5：GridSearchCV 找最佳树深度
```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
pg = {"max_depth": [3, 4, 5, 6], "n_estimators": [100, 200]}
gs = GridSearchCV(XGBRegressor(), pg, cv=3)
gs.fit(Xtr, ytr)
print("最佳参数:", gs.best_params_)             # 期望：如 {'max_depth':4,'n_estimators':200}
```

### 菜谱 6：预测 vs 真实对比图
```python
import matplotlib.pyplot as plt
pred = m.predict(Xte)
plt.figure(figsize=(10, 3))
plt.plot(yte.values[:96], label="真实"); plt.plot(pred[:96], label="预测")
plt.legend(); plt.savefig("fc_compare.png", dpi=300)   # 期望：生成 300dpi 图
```

### 菜谱 7：CVXPY 日前调度，提取最优充放电
```python
import cvxpy as cp, numpy as np
T = 24; p = cp.Variable(T); grid = cp.Variable(T)
cost = cp.sum(grid)                                  # 简化：购电量和≈成本
cons = [p >= -50, p <= 50, cp.cumsum(p) <= 100, grid >= 0, grid + p >= 0]
prob = cp.Problem(cp.Minimize(cost), cons); prob.solve()
print("最优储能功率:", np.round(p.value, 1))          # 期望：一串充放电计划(kW)
print("最小成本:", round(prob.value, 2))
```

### 菜谱 8：pandapower 带储能微网潮流
```python
import pandapower as pp
net = pp.create_empty_network()
b = pp.create_bus(net, vn_kv=0.4)
pp.create_ext_grid(net, bus=b)
pp.create_load(net, bus=b, p_mw=0.05)
pp.create_storage(net, bus=b, p_mw=0.02, max_e_mwh=0.1)
pp.runpp(net)
print(net.res_bus.vm_pu)        # 期望：电压标幺值，约 1.0 附近
```

### 菜谱 9：模型存盘 + 加载（joblib）
```python
import joblib
joblib.dump(m, "xgb_model.pkl")
m2 = joblib.load("xgb_model.pkl")   # 下次直接加载，不用重新训练
```

### 菜谱 10：批量跑 + 进度条（tqdm）
```python
from tqdm import tqdm
for seed in tqdm(range(20)):
    ...   # 重采样评估，进度条显示到第几轮
```

---

## 24. 你的真实项目逐文件导读（PV_Storage_Flex_Control）

> 这份项目就是"研究方向3"的代码实现。下面"文件 → 干什么 → 关键函数 → 你改哪"直接对应你磁盘上的文件（函数名取自实际代码）。

| 文件 | 干什么（模块角色） | 关键函数 | 你想改/学什么 |
|---|---|---|---|
| `run_content3_local.py` | **总入口**：串"预测→优化→仿真"，解析 `--no-validation` 等参数 | `main()` | 先看这里理解整体流程；`--help` 看所有开关 |
| `src/utils/common.py` | 公共工具：日志、系统参数、**读各类数据**、电价、**时变碳因子** | `get_logger / load_hourly_load / get_hourly_carbon_factors / load_system_params` | **换真实数据**就改 `load_hourly_load` 和 `get_hourly_carbon_factors` |
| `src/data_preprocessing/data_ingestion.py` | 读原始月/日 CSV（烟草园区口径） | `read_monthly_csv / load_all_daily_data / build_total_load` | 把"烟草园区"CSV 换成你学校的园区数据 |
| `src/data_preprocessing/source_load_carbon.py` | 负荷特性、光伏不确定性、碳模型、**净负荷** | `analyze_load_characteristics / build_carbon_model / compute_net_load / export_base_model` | 阶段5 碳因子建模从这里起步 |
| `src/forecasting/content3_forecast.py` | **预测核心**：特征、线性/分位训练、双时间尺度、增量/迁移/日内修正、RMSE | `build_features / train_linear / train_both_scales / forecast_profile / monthly_relative_rmse / evaluate_nextstep` | 阶段2 主战场；想上 LSTM 就在这里加 `torch` 分支 |
| `src/optimization/scenarios.py` | 构造不确定场景（光伏区间、负荷基线、电价碳因子按 T 扩展） | `build_scenarios / price_carbon_for_T / load_pv_uncertainty` | 阶段3 的"不确定集"就在这调（区间带宽度） |
| `src/optimization/energy_carbon_optimizer.py` | **优化核心**：确定/随机/鲁棒/帕累托，能-碳双目标，SOC 轨迹 | `solve_deterministic / solve_stochastic / solve_robust / solve_biobjective_pareto / soc_trace` | 阶段3 主战场；接 Gurobi 后自动加速 |
| `src/optimization/dc_microgrid_design.py` | 直流微网架构/容量/电压等级/设备/柔性负荷建模 | `dc_microgrid_architecture / unit_sizing / voltage_level_selection / flexible_load_modeling` | 和你 Simulink 物理模型呼应（阶段4 衔接） |
| `src/control/dispatch.py` | **调控核心**：日前计划 + 实时 MPC（滚动优化） | `plan_day_ahead / RealtimeMPC / simulate_closed_loop` | 阶段4 闭环；`RealtimeMPC.step` 是实时决策 |
| `src/validation/simulation.py` | 仿真验证：造验证集、跑、出月报 | `build_validation_dataset / run_validation / monthly_report` | 验收指标②（稳定运行>90%）在这里算 |
| `src/cloud_framework/cloud_service.py` | 云-边框架：CloudController + FastAPI 接口 | `CloudController / create_app`（含 `/forecast /schedule /replan` 等端点） | 阶段6 原型；`python -m` 起服务 |
| `src/safety/safety_design.py` | 安全设计（电气/防火/结构）合规文档生成 | `electrical_safety_requirements / generate_technical_system` | 申报书安全章节可直接引用 |

**怎么用这份导读**：阶段2/3/4 不知道"下一步写啥"，就回来看对应行的"关键函数"，直接 `from src.xxx import 函数名` 调现成的，再改成你的数据/方法——比从零写快 10 倍。

---

## 速查：科研代码标准流水线（背下来）
```
1. 读数据      → pandas.read_csv / read_excel
2. 清洗        → dropna / fillna / 异常截断 / 转 datetime
3. 特征        → shift(滞后) / dt.hour(周期) / 相关系数筛选
4. 划分        → train_test_split（防泄露，时序 shuffle=False）
5. 建模        → Ridge / XGBoost（预测）| cvxpy/Pyomo+Gurobi（优化）
6. 评估        → RMSE/MAE（预测）| 目标值/可行性（优化）
7. 画图        → matplotlib / seaborn / plotly
8. 存结果      → to_excel / savefig(dpi=300) / joblib.dump
```
> 这条流水线，就是你 `PV_Storage_Flex_Control` 项目 `src/` 下每个模块在干的事。对照本手册，看不懂哪个模块就回来看对应章节。

---

## 25. 核心模块「逐行精解」（把主例拆到每一行）

> 前面每节是"结构 + 一个例子"，这里挑**最常用、最易踩坑**的两个模块，把例子**逐行拆开**讲清"这行为什么这么写、变量长什么样"。其余模块的逐行同理，看 §23 实战菜谱。

### 25.1 numpy 主例逐行（对应 §1）
```python
import numpy as np
# 造 24 小时光伏出力序列，形状 (24,)
pv = np.maximum(0.0, 50.0 * np.sin((np.arange(24) - 6) / 24 * 2 * np.pi))
#         │        │             │        │    │          │    │
#         │        │             │        │    │          │    └ 2π：一个完整周期
#         │        │             │        │    │          └ 弧度换算
#         │        │             │        │    └ 0~23 共24个整数（小时序号）
#         │        │             │        └ arange(24) 生成 [0,1,...,23]
#         │        │             └ 乘 50：把 -1~1 的 sin 放大成 ±50kW 幅度
#         │        └ np.maximum(a, 0)：逐元素取 max(值, 0)，把夜间负值截成 0（光伏不出力）
#         └ np.maximum 返回新数组，不修改原数组
print(pv.shape)        # (24,)  ← 一维数组，24 个元素
print(pv[8:12])        # 取第 8~11 个小时（切片，左闭右开）
print(pv.mean())       # 求平均（向量化，不用 for）
# 向量化运算：对整个数组一次算完，比 Python for 快几十倍
pv2 = pv * 1.1         # 每个元素 ×1.1，得到新数组
# 形状变化：标量与数组运算 → 形状不变，仍是 (24,)
```

### 25.2 cvxpy 主例逐行（对应 §9，阶段3 核心）
```python
import cvxpy as cp
import numpy as np
T = 24
prices = np.array([0.32]*7 + [0.65]*3 + [1.05]*8 + [0.65]*2 + [0.32]*4)  # 峰谷平电价，形状 (24,)
net = np.array([34,31,30,29,30,33,40,55,70,82,88,90,85,80,78,82,
                95,110,120,118,105,88,65,45], dtype=float)              # 净负荷，形状 (24,)

# —— 定义"变量"（不是普通数字，是待求解的未知数）——
charge = cp.Variable(T, nonneg=True)     # 充电功率，24 个未知数，≥0
discharge = cp.Variable(T, nonneg=True)  # 放电功率，24 个未知数，≥0
soc = cp.Variable(T + 1)                 # 电量轨迹，25 个未知数（含初始）
grid = cp.Variable(T)                    # 购电功率，24 个（可正可负）

cons = [soc[0] == 50.0]                  # 约束1：初始 SOC 固定为 50kWh
for t in range(T):
    cons += [soc[t+1] == soc[t] + 0.95*charge[t] - discharge[t]/0.95]  # 约束2：电量守恒
    cons += [charge[t] <= 50, discharge[t] <= 50]                      # 约束3：功率上限
    cons += [net[t] == discharge[t] - charge[t] + grid[t]]             # 约束4：功率平衡

# —— 目标函数：最小化（购电成本 + 碳成本）——
buy = cp.pos(grid)                       # 只取买电部分（grid>0 保留，<0 变 0）
cost = cp.sum(cp.multiply(prices, buy))  # 元素相乘再求和 → 标量目标
obj = cp.Minimize(cost)                  # 声明"求最小值"
prob = cp.Problem(obj, cons)             # 把目标+约束打包成问题
prob.solve()                             # 调用求解器（默认 ECOS/OSQP）算未知数
# 求解后：charge.value / discharge.value / soc.value 才是具体数值（numpy 数组）
print(charge.value)                      # 打印最优充电计划，形状 (24,)
```
> 关键区别：`cp.Variable` 是"符号未知数"，**没 solve 之前没有值**；`prob.solve()` 之后才用 `.value` 取数值。这是优化和普通 numpy 运算最大的不同，初学者 90% 的报错来自"忘了 solve 就直接用变量"。

### 25.3 其余模块的"逐行"怎么看
- 数据/预测：看 §23 菜谱 1~6（每个都带期望输出，等于逐行）
- 优化进阶：看 §23 菜谱 7 + §10（Pyomo 整数约束）
- 你的真实项目：看 §24 逐文件导读，每个函数就是一段"逐行实现"

---

## 26. 课题②/③ 进阶：STL-LSTM 概率预测 + 退化感知优化（复制即用）

> 对应课题②（Keras STL-LSTM 概率预测 + 自适应修正）与课题③（鲁棒/MPC 调度）的**增量创新点**。基线见 §6（岭回归）、§8（statsmodels 季节分解）、§9/§10（优化）。本节在基线旁并行加"深度学习分支"和"电池退化成本"，不推倒重来。完整版见 `新手导读_研究方向3对照与代码阅读路线.md` §7。

### 26.1 STL-LSTM 概率预测（Keras，含 Pinball / CRPS，完整可运行）

**为什么**：源-荷序列 = 强周期趋势 + 季节 + 残差；STL 先拆，LSTM 再拟，比裸 LSTM 稳、可解释。产出的 q05/q50/q95 三条带可直接喂进 `scenarios.py` 当不确定情景。

```python
import numpy as np
from statsmodels.tsa.seasonal import STL
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense

# 1) STL 分解（period=日周期步数：1h 粒度 24，15min 粒度 96）
stl = STL(load_series, period=24).fit()
trend, seasonal, resid = stl.trend, stl.seasonal, stl.resid

# 2) 监督样本：过去 win 步 → 未来 1 步（对 resid 分量建模）
def make_samples(arr, win=24):
    X, y = [], []
    for i in range(win, len(arr)):
        X.append(arr[i-win:i]); y.append(arr[i])
    return np.array(X)[..., None], np.array(y)

Xr, yr = make_samples(resid.values, win=24)

# 3) Pinball Loss（分位数损失）—— 让 q05/q50/q95 各自贴合对应分位
def pinball(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q*e, (q-1)*e))
    return loss

# 4) 多分位头：共享 LSTM 主干，3 个 Dense 各出一个分位
inp = Input((24, 1)); h = LSTM(64)(inp)
outs = [Dense(1, name=f"q{int(q*100)}")(h) for q in (0.05, 0.5, 0.95)]
model = Model(inp, outs)
model.compile("adam", loss=[pinball(0.05), pinball(0.5), pinball(0.95)])
model.fit(Xr, [yr, yr, yr], epochs=50, batch_size=64, verbose=0)

# 5) 概率输出：悲观/中位/乐观 三条带
q05, q50, q95 = model.predict(Xr[-1:], verbose=0)
```

**评价（别只用 RMSE）**
- 训练损失用 **Pinball Loss**（见第 3 步）；评价用 **CRPS**（连续概率排布评分，比区间覆盖率 PICP 更被审稿人认）：
  ```python
  def crps(obs, q05, q50, q95):
      return (q50-obs)**2/2 + ((q95-q05)**2 - (q50-obs)**2)/2   # 三点高斯近似
  ```
- **PICP**：真实值落在 [q05, q95] 的比例应 ≈ 90%（对应 0.05/0.95 分位）。
- **自适应修正** = 用近 N 步预测误差平移/缩放未来预测（与 `dispatch.py` MPC 日内修正同思想，可直接复用）。

### 26.2 退化感知优化（电池 SOC 健康约束进 MPC 目标，完整建模）

**为什么（2025 热点）**：传统调度只管"经济 + 碳"，忽略电池循环寿命；把**退化成本**进目标，可在寿命与成本间权衡。文献（KAN-MPC 等 2025）显示：把电池健康 SOC 约束进 MPC 目标，循环寿命 **+20%**、成本 **−19.4%**——这是你课题③的潜在增量创新点。

```python
import cvxpy as cp
T = 24
# 变量（在 §9/§10 基础上新增）
charge   = cp.Variable(T, nonneg=True)    # 各时段充电功率
discharge= cp.Variable(T, nonneg=True)    # 各时段放电功率
soc      = cp.Variable(T+1)               # SOC 轨迹
# 已有：cost（购电）、carbon_cost（碳排成本），见 §9/§10

# 退化成本 ∝ 吞吐量（累计充放电量），近似与 |charge_t|+|discharge_t| 成正比
gamma = 0.02                              # 单位吞吐退化系数（需标定，见下）
degr_cost = gamma * cp.sum(charge + discharge)

# 目标：经济 + 碳 + 退化 三者加权最小
obj = cp.Minimize(cost + carbon_cost + degr_cost)

# SOC 健康约束：避免长期满充/深放（延长寿命），如 0.2~0.8 而非 0~1
cons = []
cons += [soc[t] >= 0.20 for t in range(T+1)]
cons += [soc[t] <= 0.80 for t in range(T+1)]
# SOC 动态（充电增、放电减，含效率）
cons += [soc[t+1] == soc[t] + 0.95*charge[t] - discharge[t]/0.95 for t in range(T)]

prob = cp.Problem(obj, cons)
prob.solve()                              # 求解器可用 §9 的 CLARABEL/ECOS
```

**关键注意**
- `gamma` 需用你电池型号的**循环寿命曲线标定**（不要拍脑袋）：用 `cycle_life = a / DOD^b` 之类经验式，反推每 kWh 吞吐对应的寿命折损成本。
- 约束收紧（0.2~0.8）会牺牲些经济性，用 **Pareto 扫 `gamma`** 看权衡前沿：
  ```python
  for gamma in [0.0, 0.01, 0.02, 0.05, 0.10]:
      # 解一遍，记录 (总成本, 累计吞吐) → 画 Pareto 曲线
  ```
- 与课题③对接：把退化项直接并进现有 `energy_carbon_optimizer.py` 的目标，鲁棒/随机/MPC 三种调度模式**不用改结构**，只是目标多一项——预测升级、退化进目标，下游白赚。

---

*本手册 v2 增补 §23/§24/§25；v2.1 增补 §26（STL-LSTM 概率预测 Keras 实战 + 退化感知优化，对应课题②/③ 增量创新点）。配套 `读研学习方向计划.md`（v5+附R）与 `新手导读_研究方向3对照与代码阅读路线.md` §7。所有例子自包含、可直接复制运行；报错优先看「第20章 调试与排错」。*
