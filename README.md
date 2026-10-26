# CooCCur-matriX

从英文 Markdown 文本构建相邻词共现网络，导出六种邻接矩阵与热力图，并对词的邻接模式进行层次聚类。

## 环境与安装

项目在 Python 3.12 上运行验证。依赖为 NumPy、matplotlib 和 SciPy。以下命令在项目根目录的 PowerShell 中执行。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-clustering.txt
```

如果已有 `.venv`，直接执行安装命令即可。后续始终显式使用虚拟环境中的 Python，无需激活环境，也不需要修改 PowerShell 执行策略。

## 快速开始

用默认输入 `1.md` 生成六种矩阵和热力图，再把默认的无向、对角线为零的矩阵分为 6 簇：

```powershell
.\.venv\Scripts\python.exe cooccurrence.py --figure-size 18
.\.venv\Scripts\python.exe cluster_words.py --clusters 6 --figure-size 18
```

所有结果默认保存到 `output/`。默认仅选择出现在至少 2 个片段中的前 40 个高频词；这里的频率按“包含该词的片段数”计算。

## 文本处理与共现定义

`cooccurrence.py` 的处理顺序如下：

1. 清理常见 Markdown 标记、代码块、行内代码、URL、完整表格行和分隔线。标题与链接文字保留；参考文献文字没有单独排除。
2. 对段落去重：忽略大小写和连续空白差异，保留第一份，防止重复粘贴正文重复计数。
3. 按换行 `\n`、英文句点 `.` 和英文逗号 `,` 切分片段，不跨片段统计边。
4. 用正则提取英文词，转为小写，移除脚本中的 `STOPWORDS`，默认移除不足 3 个字符的词。保留词内连字符和英文撇号；不做词形还原。
5. 对过滤后的连续词对计数。相同词紧邻时不统计自环。
6. 根据片段频率筛选词表，再提取对应子矩阵，行列按词的字母顺序排列。

这里的“相邻”是**过滤停用词和短词之后的相邻**，例如 `alpha and beta` 会变成 `alpha beta` 并计一条边。词数上限和最低频率筛选不会把原本不相邻的词重新连接。当前分词针对英文，不包含中文分词。

例如过滤后的片段为：

```text
alpha beta alpha gamma
```

有向边分别计为 `alpha → beta = 1`、`beta → alpha = 1`、`alpha → gamma = 1`。有向矩阵的行表示前词，列表示后词。

无向矩阵合并两个方向：`alpha — beta = 2`、`alpha — gamma = 1`，并在矩阵的两个对称位置写入相同数值。布尔矩阵则把正数边权改为 `1`。

**对角线的 count 是词的片段频率**：在上面的一个片段里，`alpha` 虽然出现了两次，对角线 count 仍为 `1`。非对角线统计每一次相邻配对，因此二者的统计单位不同。

## 六种输出

每种结果生成一份 `.csv` 和一张 `_heatmap.png`，共 12 个文件。CSV 第一行和第一列是词标签，编码为 UTF-8 BOM，便于在 Excel 中打开。

| 文件名前缀 | 边权 | 方向 | 对角线 |
| --- | --- | --- | --- |
| `directed_diag_count` | 相邻词对次数 | 有向 | 词的片段频率 |
| `directed_diag_zero` | 相邻词对次数 | 有向 | 0 |
| `undirected_diag_count` | 两个方向的相邻次数之和 | 无向 | 词的片段频率 |
| `undirected_diag_zero` | 两个方向的相邻次数之和 | 无向 | 0 |
| `directed_bool` | 有边为 1，无边为 0 | 有向 | 0 |
| `undirected_bool` | 有边为 1，无边为 0 | 无向 | 0 |

例如有向布尔图的文件为 `output/directed_bool.csv` 和 `output/directed_bool_heatmap.png`。

计数热力图使用 `inferno` 色阶和 `PowerNorm(gamma=0.55)` 增强低频差异；布尔图用两色显示 0 和 1。颜色映射不改变 CSV 数值。每张计数图根据自身最大值设置色阶，跨图比较时需看色条数值。

当前脚本输出原始计数或布尔值，未输出 count 归一化矩阵，也未进行双随机归一化。

## 共现脚本参数

```powershell
.\.venv\Scripts\python.exe cooccurrence.py [input] [选项]
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `input` | `1.md` | 输入 Markdown 路径 |
| `--outdir` | `output` | 输出目录 |
| `--top-words` | `40` | 保留片段频率最高的 N 个词；`0` 表示不限制数量 |
| `--min-frequency` | `2` | 词至少出现于多少个片段 |
| `--min-length` | `3` | 最小词长 |
| `--figure-size` | 按词数计算 | 同时设置六张热力图的宽度，单位为英寸；高度为宽度的 0.88 倍 |

未指定 `--figure-size` 时，宽度为 `max(10, min(28, 词数 × 0.55))` 英寸，输出分辨率为 220 DPI。

保留所有满足默认频次要求的词：

```powershell
.\.venv\Scripts\python.exe cooccurrence.py --top-words 0 --figure-size 28
```

连只出现于一个片段的词也保留：

```powershell
.\.venv\Scripts\python.exe cooccurrence.py --top-words 0 --min-frequency 1
```

词数较多时，CSV 仍保存完整矩阵，但图片上的词标签会变密。`--figure-size` 仅改变图像尺寸，不改变词数。

## 词语聚类

`cluster_words.py` 使用**余弦距离 + 平均连接的凝聚层次聚类**，将邻接模式相似的词分为一组。

- 无向矩阵：每个词的邻接行向量作为特征。
- 有向矩阵：拼接该词的出边行向量和入边列向量，同时考虑前后关系。
- 默认先清零对角线，防止对角线词频影响邻接模式比较；可用 `--keep-diagonal` 保留。
- 特征全零的词没有连接证据，标为 `cluster = 0`，不参与聚类、不出现在树状图中。其他簇从 1 开始编号。

这一方法比较“是否连接到相似的邻词”，不直接按两个词之间的边权分组，也不是语义嵌入或网络社区发现。聚类结果用于探索文本结构，不保证每簇对应一个主题。

用无向计数矩阵分成 6 簇：

```powershell
.\.venv\Scripts\python.exe cluster_words.py output/undirected_diag_zero.csv --clusters 6
```

用有向计数矩阵聚类：

```powershell
.\.venv\Scripts\python.exe cluster_words.py output/directed_diag_zero.csv --direction directed --clusters 6 --figure-size 18
```

也可以输入 `directed_bool.csv` 或 `undirected_bool.csv`，此时比较的是邻居存在与否。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `input` | `output/undirected_diag_zero.csv` | 输入邻接矩阵 CSV |
| `--clusters` | `6` | 非孤立词的目标簇数；不得超过可聚类词数 |
| `--direction` | `auto` | `auto` 按矩阵是否对称判断；也可指定 `directed` 或 `undirected` |
| `--outdir` | `output` | 输出目录 |
| `--prefix` | 输入文件名去掉扩展名 | 输出文件名前缀，不含目录 |
| `--figure-size` | 按词数计算 | 树状图宽度，单位为英寸；高度固定为 8 英寸 |
| `--keep-diagonal` | 关闭 | 在聚类特征中保留原始对角线 |

`--direction undirected` 会将输入矩阵与其转置取平均；`auto` 仅依据数值对称性判断，已知为有向网络时可显式指定。

以默认输入为例，输出：

- `output/undirected_diag_zero_clusters.csv`：`word,cluster` 两列，按簇编号和词排序。
- `output/undirected_diag_zero_dendrogram.png`：层次聚类树状图；纵轴为平均连接余弦距离，叶子文字颜色对应簇编号。树枝颜色采用绘图库默认距离阈值，不表示 `--clusters` 的精确切分。

**聚类脚本只能处理输入 CSV 中已有的词。** 想增加词数，应先调整 `cooccurrence.py` 的 `--top-words`、`--min-frequency` 等参数重新生成矩阵，再重新聚类。`--clusters` 只控制组数。

## 文件与重跑

| 路径 | 用途 |
| --- | --- |
| `1.md` | 默认输入文本 |
| `cooccurrence.py` | 生成六种邻接矩阵和热力图 |
| `cluster_words.py` | 读取邻接矩阵，生成聚类结果和树状图 |
| `requirements-clustering.txt` | 运行两个脚本所需的依赖列表 |
| `output/` | 默认输出目录 |

两个脚本均使用无界面的 matplotlib `Agg` 后端，图片直接保存，无需 Tk 或弹窗。重跑会覆盖同名输出，但不会删除目录中的其他旧结果。需要保留不同实验时，使用不同 `--outdir`，聚类也可用不同 `--prefix`。

查看完整命令帮助：

```powershell
.\.venv\Scripts\python.exe cooccurrence.py --help
.\.venv\Scripts\python.exe cluster_words.py --help
```
