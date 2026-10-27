# CooCCur-matriX

从**英文 Markdown** 构建相邻词共现矩阵，挖掘并合并二词组合，输出热力图、PPMI/NPMI 关联度和层次聚类结果。

| 脚本 | 输入 | 用途 | 默认输出目录 |
| --- | --- | --- | --- |
| `cooccurrence.py` | Markdown，默认 `1.md` | 生成六种单词邻接矩阵及热力图 | `output/` |
| `phrase_enhance.py` | Markdown，默认 `1.md` | 挖掘词组、合并原文匹配并重新生成六种矩阵及热力图 | `output/phrase/` |
| `association.py` | 必填的计数矩阵 CSV | 计算 PPMI/NPMI 并输出热力图 | `output/` |
| `phrase_entropy.py` | Markdown，默认 `1.md`；可选候选表 | 计算有向连续词组的左、右及联合边界熵，输出独立图、左右对照图和联合熵图 | `output/phrase_entropy/` |
| `phrase_metrics.py` | Markdown，默认 `1.md`；可选候选表 | 输出四分格总览及左右熵、联合熵、熵差、PPMI、NPMI 独立图 | `output/phrase_metrics/` |
| `select_bigrams.py` | Markdown，默认 `1.md` | 按频次、PPMI/NPMI 和低边界熵筛选有向二词候选 | `output/bigram_candidates/` |
| `trigram_potential.py` | Markdown 和二词候选 CSV | 分析候选二词组的左右三词扩展，输出频次、左熵、右熵对照柱状图 | `output/trigram_potential/` |
| `cluster_words.py` | 矩阵 CSV，默认 `output/undirected_diag_zero.csv` | 按邻接模式聚类，输出分组及树状图 | `output/` |
| `bool_hop_stability.py` | 有向或无向布尔邻接矩阵 CSV | 计算累计 Boolean n-hop 可达关系的稳定 hop，并逐步输出可达、加权矩阵和热力图 | `output/bool_hop_stability/` |

词组增强可独立运行。关联度分析和聚类都可以直接读取它生成的矩阵，不必先运行 `cooccurrence.py`。

## 环境与安装

项目已在 Python 3.12.8 上运行验证，依赖 NumPy、matplotlib 和 SciPy。以下命令在项目根目录的 **Windows PowerShell** 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-clustering.txt
```

如果已有 `.venv`，直接安装依赖即可。后续显式调用虚拟环境中的 Python，无需激活环境或修改 PowerShell 执行策略。路径含空格时，用双引号包住路径，例如 `"doc/my notes.md"`。

## 快速开始

### 单词共现、关联度与聚类

```powershell
.\.venv\Scripts\python.exe cooccurrence.py 1.md
.\.venv\Scripts\python.exe association.py output/directed_diag_zero.csv --outdir output/association
.\.venv\Scripts\python.exe cluster_words.py output/directed_diag_zero.csv --direction directed --clusters 6 --outdir output/clusters
```

第一步默认保留至少出现在 2 个片段中的前 40 个高频词；频率按**包含该词的片段数**计算。后两步分别分析第一步生成的矩阵，可以独立选择运行。

### 词组增强、左右熵与后续分析

`phrase_metrics.py` 除四分格总览外，还分别输出五张图：左右熵对照图（每格左上为左熵、右下为右熵，二者共用 nat 色标）、联合边界熵、左熵减右熵、PPMI 和 NPMI。差值图以零为中心，正值为朱红、负值为青绿。对应 CSV 包含左熵、右熵、联合熵、左熵减右熵、PPMI、NPMI 和词组计数。独立图色系与四分格一致；未观察到的单元保持白色。若没有任何词轴，PNG 仍会生成并显示空轴提示。

```powershell
.\.venv\Scripts\python.exe phrase_enhance.py 1.md --outdir output/phrase --top-words 0
.\.venv\Scripts\python.exe phrase_entropy.py 1.md --outdir output/phrase_entropy --phrase-candidates output/phrase/phrase_candidates.csv
.\.venv\Scripts\python.exe phrase_metrics.py 1.md --outdir output/phrase_metrics
.\.venv\Scripts\python.exe select_bigrams.py 1.md --outdir output/bigram_candidates
.\.venv\Scripts\python.exe association.py output/phrase/phrase_enhanced_directed_diag_zero.csv --outdir output/phrase/association
.\.venv\Scripts\python.exe cluster_words.py output/phrase/phrase_enhanced_directed_diag_zero.csv --direction directed --clusters 6 --outdir output/phrase/clusters
```

`select_bigrams.py` 默认要求二词至少出现 3 次、PPMI ≥ 0.5、NPMI ≥ 0.2，且左熵、右熵或联合熵至少一项 ≤ 0.5 nat。阈值可分别通过 `--min-count`、`--min-ppmi`、`--min-npmi` 和 `--max-entropy` 调整。输出 CSV 按 NPMI、PPMI 和频次排序，包含主要左右上下文；`left`、`right` 列可直接交给 `phrase_entropy.py --phrase-candidates` 复核。

用筛出的二词候选继续检查三词扩展：

```powershell
.\.venv\Scripts\python.exe trigram_potential.py 1.md --phrase-candidates output/bigram_candidates/1_bigram_candidates.csv
```

脚本逐个查找左扩展 `X A B` 和右扩展 `A B Y`，默认只保留至少出现 2 次的三词组合。输出 `_trigram_potential.csv` 包含三词计数、段落频次、原二词的左右边界熵与联合边界熵，以及以全语料连续三词窗口为事件空间计算的 PPMI/NPMI。静态 PNG 按三词频次排序，分别显示频次柱状图和左右熵并排对照图；频次柱颜色表示 NPMI。频次与熵采用独立横轴，熵单位为 nat。`--min-trigram-count` 调整频次下限，`--top` 调整图中展示数量（0 表示全部）。

### 有向或无向布尔图的稳定 hop

`bool_hop_stability.py` 读取带顶点标签的方阵 CSV，同时支持有向图和无向图，不要求对称。行是出发词，列是目标词；`B[i,j] > 0` 表示从 i 到 j 有边，反向是否有边由 `B[j,i]` 决定。输入对角线仍须为零，暂不支持自环。

```powershell
.\.venv\Scripts\python.exe bool_hop_stability.py output/undirected_bool.csv
```

分析有向图时直接传入有向邻接矩阵，无需新增参数。建议指定独立输出目录：

```powershell
.\.venv\Scripts\python.exe bool_hop_stability.py output/base/directed_bool.csv --outdir output/directed_bool_hop_stability
```

命令格式：

```text
python bool_hop_stability.py <input.csv> [--outdir DIR] [--cell-size INCHES]
```

| 参数 | 必填 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `input` | 是 | 无 | 带标签的方阵邻接矩阵 CSV |
| `--outdir DIR` | 否 | `output/bool_hop_stability/` | 每步矩阵和热力图的输出目录 |
| `--cell-size INCHES` | 否 | `0.35` | 热力图单格目标尺寸，单位英寸；图像画布最大为 18 英寸 |

CSV 第一行是列标签，第一列是行标签；左上角可以写 `word` 或 `vertex`。行标签必须与列标签按相同顺序对应。示例：

```csv
word,alpha,beta,gamma
alpha,0,1,0
beta,1,0,1
gamma,0,1,0
```

矩阵必须是方阵且对角线为零；无向图矩阵对称，有向图可以不对称。脚本按 `B > 0` 转换为布尔邻接矩阵。

脚本按 `R_n = B ∨ B² ∨ … ∨ Bⁿ` 计算累计可达矩阵，并使用布尔乘法迭代 `R_(k+1) = R_k ∨ (R_k @ B)`。当 `R_(k+1) = R_k` 时，返回 k 作为稳定步数，不额外输出重复的第 k+1 步。停止判断包含对角线：例如有向三环在第 3 步才产生回到自身的可达项，因此稳定步数为 3。无边图按第 1 步已稳定处理。

输出目录中每个 hop 都有一份带词名行列标签的 `reachability_NNN.csv` 和对应的 `reachability_NNN.png` 热力图；稳定步的 PNG 标有 `(stable)`。有向输入的结果保留方向，不会自动补反向边。热力图使用从 0 开始的数字顶点编号，词很多时每隔若干顶点显示一个编号。`--cell-size` 控制画布大小，画布最大为 18 英寸。可用 `--outdir` 指定输出目录。

每一轮还输出加权矩阵 `weighted_NNN.csv` 和热力图 `weighted_NNN.png`：

```text
S_k = k·B¹ + (k−1)·B² + … + Bᵏ
S_1 = B
S_2 = 2B + B²
S_3 = 3B + 2B² + B³
```

这里 `Bⁿ` 沿用布尔矩阵幂：恰好 n 步存在走法时为 1，否则为 0；各项乘权和相加使用普通整数运算。同一对词在多个 hop 可达会累加，较早的 hop 权重更高。k 每轮变化，之前各项的系数都会增加。实现使用等价递推 `C_k = C_(k−1) + Bᵏ`、`S_k = S_(k−1) + C_k`（初值均为零），每轮结果与按当前 k 重新求和完全一致，无需重算之前的幂。

加权图值越大颜色越深，色条显示该轮数值范围。加权矩阵在布尔可达关系稳定的那一轮停止输出；这只是计算终点，加权数值本身不保证稳定。最终结果就是最后一轮的 `weighted_NNN.csv`。此功能仍使用完整矩阵存储，三万顶点的计算和绘图内存问题尚未解决。



#### `phrase_metrics.py` 使用说明

该脚本从 Markdown 原文提取有向连续词组 `(A,B)`，在同一词轴上计算边界熵和关联度。矩阵行表示前词 `A`，列表示后词 `B`；`A → B` 与 `B → A` 是不同词组。PPMI/NPMI 使用 `association.py` 相同的行列边际定义；熵使用 `phrase_entropy.py` 的上下文和边界规则，单位为 nat。

```powershell
.\.venv\Scripts\python.exe phrase_metrics.py 1.md --outdir output/phrase_metrics --plot-top-words 0
```

可选参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `input` | `1.md` | 输入 Markdown 文件 |
| `--outdir` | `output/phrase_metrics` | 输出目录，不存在时自动创建 |
| `--prefix` | 输入文件名（不含扩展名） | 输出文件名前缀 |
| `--phrase-candidates` | 不启用 | 候选 CSV；读取 `left`、`right` 两列以限制统计的有序词组 |
| `--plot-min-phrase-frequency` | `0` | 只筛选图片：正数 `N` 隐藏出现次数少于 `N` 的词组，并仅显示达标词组涉及的词轴；所有 CSV 始终保存完整统计矩阵 |
| `--min-length` | `3` | 词组两端词语的最小长度；停用词仍会排除 |
| `--plot-top-words` | `0` | 只限制图片显示的词轴数；`0` 显示完整词轴，正数按词组计数的入度加出度选取。不会改变 CSV |
| `--figure-size` | 自动 | 图片宽度/高度（英寸）；不指定时按显示词数估算 |

每次运行输出 7 个矩阵 CSV、1 份边界报告和 6 张 PNG。矩阵 CSV 的空单元表示该指标未定义或该词组未观察到，不会以 0 代替：

| 文件后缀 | 内容 |
| --- | --- |
| `_phrase_counts.csv` | 有向连续词组计数 |
| `_left_entropy.csv`、`_right_entropy.csv` | 左、右边界熵 |
| `_joint_entropy.csv` | 左右边界联合熵 `H(L,R | A B)`，不是左右熵之和 |
| `_left_minus_right_entropy.csv` | 左熵减右熵；仅在左右熵都有定义时计算 |
| `_ppmi.csv`、`_npmi.csv` | 词组的 PPMI 和 NPMI |
| `_phrase_boundaries.csv` | 每个词组的左右上下文计数及联合边界摘要 |

图片文件为 `_four_metrics_heatmap.png`、`_left_right_entropy_heatmap.png`、`_joint_entropy_heatmap.png`、`_left_minus_right_entropy_heatmap.png`、`_ppmi_heatmap.png` 和 `_npmi_heatmap.png`。四分格总览中每个词组占一个格子：左上为左熵、右上为右熵、左下为 PPMI、右下为 NPMI。单独的左右熵对照图每格分成两个三角形，左上三角显示左熵、右下三角显示右熵，两者共用同一色标；它不表示相减。左熵减右熵由独立差值图显示，以 0 为中心，负值为青绿、正值为朱红。另三张独立图分别展示联合熵、PPMI 和 NPMI。

左熵、右熵和 PPMI 使用从暖白 `#F3E6E2` 经灰朱 `#D69588` 到深朱 `#A64036` 的顺序色阶；左右熵在对照图中共享同一范围。差值与 NPMI 使用青绿 `#2F6F73`、暖灰 `#E9E4E1`、深朱 `#A64036` 的发散色阶，其中 NPMI 固定范围为 `[-1, 1]`。未观察/缺失单元显示为白色。`--plot-top-words 0` 时保留全部基础词轴，包括没有任何合格邻接的词；这些单元为空白。没有有效词轴时仍会生成所有 PNG，并在图中标明空轴。

这里的 `--top-words 0` 保留全部满足片段频率要求的词和词组，增强热力图仍默认只显示高频 40 项。终端会打印候选数、实际合并次数、保留词数和输出路径。词表很小时，应相应调低 `--clusters`。

## 文本处理与共现定义

基础共现脚本 `cooccurrence.py` 的处理顺序：

1. 清理三反引号代码块、行内代码、URL、完整表格行、分隔线和常见 Markdown 标记。保留标题和链接文字；参考文献没有单独排除。
2. 对段落去重：忽略大小写和连续空白差异，仅保留第一份。重复粘贴的段落不会重复计数。
3. 按换行 `\n`、英文句点 `.` 和英文逗号 `,` 切分片段，不跨片段统计边。这不是完整的自然语言句子切分。
4. 用 `[A-Za-z][A-Za-z'-]*` 提取英文词并转为小写，过滤 `STOPWORDS` 和短词。默认最小词长为 3；保留词内连字符和英文撇号，不做词形还原。
5. 对过滤后连续的词对计数；两个相同词紧邻时不统计自环。
6. 根据片段频率和词数上限筛选词表，取对应子矩阵，行列按字母顺序排列。

停用词定义在 `cooccurrence.py` 的 `STOPWORDS`，主要为冠词、介词、连词、代词、助动词和常见修饰词。词组脚本复用该集合；当前没有自定义停用词文件参数，也不自动挖掘额外停用词。

**基础矩阵中的相邻是删除停用词和短词之后的相邻。** 例如 `alpha and beta` 会变成 `alpha beta`，因此产生一条边。后续词频或词数筛选只截取矩阵，不会越过被筛掉的词重新连边。当前分词针对英文，不支持中文分词。

例如过滤后的一个片段是：

```text
alpha beta alpha gamma
```

有向边为 `alpha → beta = 1`、`beta → alpha = 1`、`alpha → gamma = 1`。**有向矩阵的行是前词，列是后词**。

无向矩阵把两个方向相加：`alpha — beta = 2`、`alpha — gamma = 1`，并在对称位置写入相同数值。布尔矩阵把所有正数边权改为 `1`。

**对角线的 count 是片段频率，不是词的出现总次数。** 上面一个片段内 `alpha` 出现两次，对角线仍为 `1`；非对角线则统计每一次相邻配对。两者的统计单位不同，关联度和聚类脚本默认忽略对角线。

## 六种矩阵与 CSV 格式

基础共现脚本为每种矩阵生成一份 `.csv` 和一张 `_heatmap.png`，共 12 个文件：

| 文件名前缀 | 非对角线边权 | 方向 | 对角线 |
| --- | --- | --- | --- |
| `directed_diag_count` | 相邻词对次数 | 有向 | 词的片段频率 |
| `directed_diag_zero` | 相邻词对次数 | 有向 | 0 |
| `undirected_diag_count` | 两个方向的相邻次数之和 | 无向 | 词的片段频率 |
| `undirected_diag_zero` | 两个方向的相邻次数之和 | 无向 | 0 |
| `directed_bool` | 有边为 1，无边为 0 | 有向 | 0 |
| `undirected_bool` | 有边为 1，无边为 0 | 无向 | 0 |

例如 `output/directed_diag_zero.csv` 对应 `output/directed_diag_zero_heatmap.png`。增强结果使用同样的六种后缀，并增加 `phrase_enhanced_` 前缀。

矩阵 CSV 使用 UTF-8 BOM 编码，便于在 Excel 中打开。第一行和第一列为标签，标签顺序必须一致，数值部分为方阵：

```csv
word,alpha,beta
alpha,0,3
beta,1,0
```

上例表示 `alpha → beta` 出现 3 次、`beta → alpha` 出现 1 次。词组标签直接使用空格，如 `machine learning`。读取矩阵时要求标签唯一且非空，数值有限、非负；`phrase_candidates.csv` 和 `*_clusters.csv` 是报告表，不能作为矩阵输入。

## 单词共现：cooccurrence.py

```powershell
.\.venv\Scripts\python.exe cooccurrence.py 1.md --outdir output/base --plot-top-words 0 --min-frequency 1 --figure-size 28
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `input` | `1.md` | 输入 Markdown 路径 |
| `--outdir` | `output` | 输出目录 |
| `--plot-top-words` | `40` | 热力图显示片段频率最高的 N 个词；`0` 显示全部，不影响 CSV |
| `--min-frequency` | `2` | 词至少出现于多少个片段，整数且至少为 1 |
| `--min-length` | `3` | 最小词长，整数且至少为 1 |
| `--figure-size` | 按词数计算 | 六张热力图的宽度，单位为英寸 |

上例连仅出现于一个片段的词也写入完整矩阵 CSV，并在热力图中显示全部词。CSV 始终包含所有满足 `--min-frequency` 的词；`--plot-top-words` 只限制热力图显示的词数，正数按片段频率选取，`0` 显示完整词表。

## 词组增强：phrase_enhance.py

输入为 `.md` 或 `.markdown` 文件。脚本读取原文中连续的二词组合，按出现次数和 NPMI 筛选，将具体匹配合并为一个词组，再重建整数计数矩阵。仅有共现矩阵无法恢复每次匹配的前后词序，因此增强脚本需要 Markdown 输入。

```powershell
.\.venv\Scripts\python.exe phrase_enhance.py 1.md --outdir output/phrase --top-words 0 --min-phrase-frequency 3 --min-npmi 0.2
```

### 匹配与合并规则

1. 复用基础脚本的 Markdown 清理、段落去重、分段和英文词提取规则。
2. **先保留停用词和短词**，统计原文片段内的相邻词对。NPMI 的总量和左右边际频率使用全部相邻词对计算。
3. 候选组合必须由两个不同的非停用词组成，两个词都满足 `--min-length`，词对次数满足 `--min-phrase-frequency`。
4. 以 `--min-npmi` 筛选候选，在原文词序上从左到右合并。每个词最多参与一次合并，不跨片段、不递归挖掘更长词组。
5. 再删除剩余的停用词和短词，按合并后的片段重新统计矩阵、片段频率，最后进行词表筛选。

例如 `machine learning` 可以成为候选，`machine and learning` 不会被合并成该词组；后者在最终单词矩阵里仍可能因删除 `and` 而产生普通邻接边。

若 `alpha beta` 和 `beta gamma` 都入选，`alpha beta gamma` 会合并为 `[alpha beta] [gamma]`。重叠冲突按位置解决，不比较 NPMI 高低。词组在一次实际匹配中替代组成词；组成词在其他位置独立出现时仍可保留。

这是一种统计筛选，不保证候选都是语言学上合适的短语。词组评分使用原文相邻词对，`association.py` 使用输入矩阵中的计数，二者的样本空间不同，NPMI 数值不应直接等同。

### 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `input` | `1.md` | 输入 Markdown，不接受矩阵 CSV |
| `--outdir` | `output/phrase` | 输出目录 |
| `--min-phrase-frequency` | `3` | 去重后原文中连续词对的最低出现次数，正整数 |
| `--min-npmi` | `0.2` | 词对 NPMI 下限，范围为 -1 至 1 |
| `--min-frequency` | `2` | 合并后词或词组至少出现于多少个片段，正整数 |
| `--top-words` | `40` | 合并后保留的高频词和词组总数；`0` 表示全部 |
| `--min-length` | `3` | 单词及每个词组组成词的最小长度，正整数 |
| `--plot-top-words` | `40` | 热力图显示片段频率最高的 N 项；`0` 显示全部，不影响 CSV |
| `--figure-size` | 按显示词数计算 | 热力图宽度，单位为英寸 |
| `--no-plots` | 关闭 | 仅生成 CSV，跳过绘图 |

`--min-phrase-frequency` 统计词对的出现次数，`--min-frequency` 统计合并后包含词或词组的片段数，两个门槛分别生效。相同段落会先去重，重复复制一段文本不会抬高频次。

### 输出与候选表

默认输出 **6 份矩阵 CSV + 6 张热力图 + 1 份候选表**。矩阵名为 `phrase_enhanced_<六种后缀>.csv`，热力图名为 `phrase_enhanced_<六种后缀>_heatmap.png`。计数来自合并后的实际词序，`diag_count` 对角线仍是片段频率。

`phrase_candidates.csv` 按 NPMI 降序、次数降序排列，记录所有达到词对最低次数的合格候选，包括未达到 NPMI 门槛的候选：

| 字段 | 含义 |
| --- | --- |
| `phrase` | 词组标签，以空格连接两个词 |
| `left`、`right` | 前词、后词 |
| `count` | 去重后原文中的连续词对出现次数，包含可能重叠的匹配 |
| `npmi` | 原文词对的归一化互信息 |
| `selected` | 是否通过 NPMI 门槛，值为 `True` 或 `False` |
| `merged_count` | 解决重叠后实际合并的次数 |
| `segment_frequency` | 合并后包含该词组的片段数 |
| `included_in_matrix` | 通过最终片段频率和词数筛选后是否进入矩阵 |

`selected=True` 不保证进入最终矩阵：重叠可能阻止实际合并，片段频率和词数上限也可能将词组筛掉。没有候选通过阈值时，仍输出普通词矩阵；没有任何词通过最终筛选时，脚本会报错。

仅生成 CSV，适合调整挖掘阈值时使用：

```powershell
.\.venv\Scripts\python.exe phrase_enhance.py 1.md --outdir output/phrase_csv --top-words 0 --no-plots
```

## 可视化与显示数量

所有图直接保存为 PNG，使用 matplotlib 的 `Agg` 后端，不弹出窗口。矩阵热力图为 220 DPI；默认宽度为 `max(10, min(28, 显示词数 × 0.55))` 英寸，高度为宽度的 0.88 倍。`--figure-size` 只改变图宽，不改变数据或显示词数。

| 图类型 | 默认显示范围 | 配色与含义 |
| --- | --- | --- |
| 基础计数热力图 | 全部已筛选词 | 灰白至棕红色，使用 `PowerNorm(gamma=0.55)` 增强低频差异 |
| 增强计数热力图 | 合并后片段频率最高的 40 项 | 同上，词组标签加粗并标为棕红色 |
| 布尔热力图 | 随对应基础/增强脚本 | 灰白和深棕红两色区分 0、1 |
| PPMI 热力图 | 输入矩阵中行和与列和之和最高的 40 项 | 灰白至棕红色，线性色阶从 0 到完整 PPMI 矩阵最大值 |
| NPMI 热力图 | 与 PPMI 相同 | `coolwarm`，固定色阶 -1 至 1 |

**矩阵保留词数和热力图显示词数分别控制：**

- `--top-words` 决定基础/增强 CSV 中有多少个词或词组；被排除的项无法在后续分析中恢复。
- `--plot-top-words` 只控制增强/关联度热力图的展示子集；CSV 和关联度计算仍使用完整输入矩阵。
- `--plot-top-words 0` 显示全部。词数较多时标签会密集，建议先查看 40～60 项。

例如，保留完整增强矩阵，图中显示高频 60 项：

```powershell
.\.venv\Scripts\python.exe phrase_enhance.py 1.md --top-words 0 --plot-top-words 60 --figure-size 24
```

增强热力图筛选显示项后仍按字母顺序排列；关联度图按共现强度排名展示。计数图按各自显示矩阵的最大值设定色阶，跨图比较应查看色条数值。色阶变换不改变 CSV 中的原始计数，也不是归一化矩阵；当前没有双随机归一化步骤。


## PPMI 与 NPMI：association.py

输入非负计数矩阵 CSV（有向或无向均可）。脚本只负责矩阵关联度，不计算左右熵；左右熵由独立的 `phrase_entropy.py` 从原始 Markdown 计算。

```powershell
.\.venv\Scripts\python.exe association.py output/directed_diag_zero.csv --outdir output/association
```

设矩阵单元计数为 $c_{ij}$，总计数为 $N$，以行和、列和估计边际频率：

$$
PMI(i,j)=\ln\frac{c_{ij}N}{c_{i\cdot}c_{\cdot j}},\qquad
PPMI(i,j)=\max(PMI(i,j),0),\qquad
NPMI(i,j)=\frac{PMI(i,j)}{-\ln(c_{ij}/N)}.
$$

默认清零对角线。未观察到的词对在输出中填 `0`；如全部配对事件集中在一个单元，该单元的 NPMI 约定为 `1`。热力图默认显示共现强度最高的 40 项，CSV 始终保留完整输入词表。

输出：`{prefix}_ppmi.csv`、`{prefix}_npmi.csv`、两张对应热力图和 `{prefix}_ppmi_npmi_comparison.png`。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `input` | 必填 | 计数矩阵 CSV 路径 |
| `--outdir` | `output` | 输出目录 |
| `--prefix` | 输入文件名去掉扩展名 | 输出文件名前缀，不能包含目录 |
| `--include-diagonal` | 关闭 | 将对角线计数纳入关联度计算 |
| `--plot-top-words` | `40` | 热力图显示强度最高的 N 项；`0` 显示全部 |
| `--figure-size` | 按显示词数计算 | 单张热力图宽度，单位为英寸 |

## 词组边界熵：phrase_entropy.py

边界熵针对**原文中的连续有序词组**，因此必须读取 Markdown。输入不是共现矩阵；脚本仅支持有向词序。矩阵格 `(A,B)` 表示短语 `A B`，反向 `B A` 是另一个独立短语。左图统计 `X A B` 中左邻词 `X` 的分布熵，右图统计 `A B Y` 中右邻词 `Y` 的分布熵；联合图统计左右邻词对 `(X,Y)` 的联合分布熵。熵以自然对数计算，单位为 nat。停用词和短词会阻止词组匹配，但若在词组外侧仍作为真实上下文计入。

对联合熵，给定词组 `A B`，将每次出现的左右上下文视为一个有序组合：

$$
H(L,R\mid A B)=-\sum_{x,y}p(x,y\mid A B)\ln p(x,y\mid A B).
$$

这直接衡量左右边界组合的多样性，不等同于左熵与右熵相加；不同的 `(X,Y)` 组合会分别计数。

```powershell
.\.venv\Scripts\python.exe phrase_entropy.py 1.md --outdir output/phrase_entropy
```

可用词组增强脚本输出的候选表限制要统计的有序词对；按 `left,right` 列读取全部候选行，不要求候选已被选中：

```powershell
.\.venv\Scripts\python.exe phrase_entropy.py 1.md --phrase-candidates output/phrase/phrase_candidates.csv --outdir output/phrase_entropy
```

脚本按基础规则清理 Markdown、去重段落，并保留停用词/短词识别原始相邻边界。不会跨越标点分段、段落或被过滤 token 拼接词组。每次真实出现独立计数；左右单侧熵忽略缺失的那一侧，并在报告记录 `*_missing_count`。联合熵则将片段边界分别记作 `<BOS>`（左边界）和 `<EOS>`（右边界），把每次出现的 `(X,Y)` 都纳入联合分布。无上下文时单侧熵留空；只有一种上下文时熵为真实的 `0`。无该短语的位置标灰；单侧无上下文的熵格也标灰。

输出文件均使用输入文件名作为默认前缀：`{prefix}_phrase_counts.csv`、`{prefix}_left_entropy.csv`、`{prefix}_right_entropy.csv`、`{prefix}_joint_entropy.csv`、`{prefix}_phrase_boundaries.csv`，以及六张 PNG。原始四张图为 `{prefix}_left_entropy_heatmap.png`、`{prefix}_right_entropy_heatmap.png`、`{prefix}_left_right_entropy_heatmap.png` 和 `{prefix}_joint_entropy_heatmap.png`；额外两张删空轴图为 `{prefix}_nonempty_left_right_entropy_heatmap.png` 和 `{prefix}_nonempty_joint_entropy_heatmap.png`。左、右熵各有独立图；左右对照图每格分为两个三角形：左上表示左熵，右下表示右熵；共享同一色阶，缺失值标灰。联合熵另图显示。所有图只表示有向顺序，不生成无向图。默认保留全部真实出现的合格双词组（包括仅出现 1 次的词组），不做默认频次截断。`--min-phrase-frequency 0`（默认，省略参数等效）时，四张原始图及 CSV 使用清理、段落去重后的完整基础词表（仍沿用最小词长和停用词规则），不从词组端点反推词表：低频词、单独出现的词、没有合格邻接的词都保留原始词轴，空格标灰。额外两张图只从同一矩阵删除整行和整列均无邻接的词，保留只有一条邻接的词，不重新筛选词组、不重算熵、不修改统计值或用零替代缺失值。全部无邻接时，删空轴图显示无非空词轴提示，不回退引入其他词。默认 `--plot-top-words 0` 显示完整词表；仅显式指定正数时限制图中显示数量，不影响 CSV。传入 `--min-phrase-frequency N` 且 N > 0 时，先保留出现次数至少为 N 的词组，再将全部 CSV 和六张图的词轴缩减为这些词组包含的词；即使显式传入 `1` 也会删除不参与任何合格词组的词轴。若没有词组满足阈值，则 CSV 仅含表头，六张图显示无非空词轴提示。候选表本身只限制词组，不缩减基础词轴。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `input` | `1.md` | 原始 Markdown |
| `--outdir` | `output/phrase_entropy` | 输出目录 |
| `--prefix` | 输入文件名去掉扩展名 | 输出文件名前缀 |
| `--phrase-candidates` | 不启用 | 可选候选 CSV，须有 `left`、`right` 列 |
| `--min-phrase-frequency` | `0` | `0` 不做频次筛选、不删基础词轴；N > 0 时筛掉出现次数不足 N 的词组，并删除不参与保留词组的词轴；不接受负数 |
| `--min-length` | `3` | 词组词元最小长度；停用词仍排除 |
| `--plot-top-words` | `0` | 默认显示完整基础词表（含空轴）；正数才限制显示数量 |
| `--figure-size` | 按显示词数计算 | 单张图宽度，单位为英寸 |

## 词语聚类：cluster_words.py

使用**余弦距离 + 平均连接的凝聚层次聚类**，按邻接模式分组：

- 无向输入：每个词的邻接行向量作为特征；非对称输入先与转置取平均。
- 有向输入：拼接出边行向量和入边列向量，同时考虑前后关系。
- 默认清零对角线；`--keep-diagonal` 可保留片段频率特征。
- 特征全零的词标为 `cluster=0`，不参与聚类，也不出现在树状图中；其余簇从 1 开始编号。

```powershell
.\.venv\Scripts\python.exe cluster_words.py output/phrase/phrase_enhanced_directed_diag_zero.csv --direction directed --clusters 6 --outdir output/phrase/clusters --prefix enhanced --figure-size 18
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `input` | `output/undirected_diag_zero.csv` | 输入矩阵 CSV |
| `--clusters` | `6` | 目标簇数，至少为 1；存在非孤立词时不得超过其数量 |
| `--direction` | `auto` | `auto` 按数值对称性判断；或显式指定 `directed`、`undirected` |
| `--outdir` | `output` | 输出目录；不会自动跟随输入文件目录 |
| `--prefix` | 输入文件名去掉扩展名 | 输出文件名前缀，不能包含目录 |
| `--figure-size` | 按词数计算 | 树状图宽度，单位为英寸；高度固定为 8 英寸 |
| `--keep-diagonal` | 关闭 | 在聚类特征中保留原始对角线 |

已知输入为有向矩阵时建议使用 `--direction directed`，因为有向计数也可能恰好对称。计数矩阵和布尔矩阵都可聚类，分别比较连接强度模式和邻居存在模式。也可输入非负 PPMI 矩阵；包含负值的 NPMI 矩阵会被拒绝。

上例生成：

- `output/phrase/clusters/enhanced_clusters.csv`：`word,cluster` 两列，按簇编号和词排序。
- `output/phrase/clusters/enhanced_dendrogram.png`：树状图，纵轴为平均连接余弦距离。叶子文字按簇着色；同簇子树的分支使用对应颜色，跨簇分支为灰色。超过 6 簇时颜色循环使用，具体归属以 CSV 为准。

这一步比较“是否连接到相似的邻词”，不直接按两个词之间的边权分组，也不保证每簇对应一个语义主题。`--clusters` 只改变组数；基础共现 CSV 的词数由 `--min-frequency` 决定，增强 CSV 的词数可由 `--top-words` 或 `--min-frequency` 调整，重新生成 CSV 后再聚类。

## 常见问题与调参

| 现象 | 检查与调整 |
| --- | --- |
| 不知道哪里传入文件 | 文件路径是脚本后的第一个位置参数：基础/增强脚本传 Markdown，关联度/聚类脚本传矩阵 CSV |
| 路径不存在或输出位置不对 | 从项目根目录运行；相对路径均相对于当前工作目录。分析增强结果时显式填写 `--outdir` |
| 没有词组通过筛选 | 先看 `phrase_candidates.csv`；适当降低 `--min-phrase-frequency` 或 `--min-npmi`，确认原文是连续匹配且不是重复段落 |
| 候选已入选，矩阵里却没有 | 看 `merged_count` 和 `included_in_matrix`；再检查重叠、`--min-frequency` 和 `--top-words` |
| 基础共现热图只显示 40 个词 | 调整 `cooccurrence.py` 的 `--plot-top-words`；CSV 仍包含全部满足 `--min-frequency` 的词 |
| 增强结果只保留 40 个词或词组 | 将 `phrase_enhance.py` 的 `--top-words` 调大或设为 `0`；必要时降低片段频率门槛 |
| CSV 词很多，增强/关联度图上只有 40 项 | 调整对应脚本的 `--plot-top-words`，无需改变 CSV |
| 图片太密或生成较慢 | 减少显示项、调整 `--figure-size`；增强脚本可加 `--no-plots`。完整矩阵存储量随词数平方增长 |
| `no words/terms remain` | 检查英文内容，降低 `--min-frequency` 或 `--min-length`；空文件和中文正文不会得到英文词表 |
| `no off-diagonal co-occurrences` | 输入没有可评分的配对事件；检查是否只剩孤立词，并重新生成包含有效邻接边的计数矩阵 |
| 聚类提示簇数过多 | 降低 `--clusters`，其上限取决于有连接的词数；查看终端列出的孤立词 |
| 修改停用词后结果没变 | 两个 Markdown 脚本都需重新运行，下游再读取新的 CSV；已有矩阵不会自动更新 |

调词组阈值时建议先保留 `--top-words 0`，查看候选表，再决定最终词表大小。提高词对次数门槛能减少稀有组合，提高 NPMI 门槛会保留关联更强的组合；仍需检查候选是否符合具体文本含义。

## 文件、重跑与验证

| 路径 | 用途 |
| --- | --- |
| `1.md` | 默认输入文本，可换成自己的 Markdown |
| `cooccurrence.py` | 文本清理、分词、基础矩阵及热力图 |
| `phrase_enhance.py` | 原文词组挖掘、合并、增强矩阵及热力图 |
| `association.py` | PPMI/NPMI 矩阵、热力图和对照图 |
| `phrase_entropy.py` | 原文连续词组左/右/联合边界熵矩阵、报告和六张有向热力图（四张原轴图、两张删全空行列图） |
| `phrase_metrics.py` | 原文左/右熵及 PPMI/NPMI 对齐矩阵、一张四分格热力图 |
| `cluster_words.py` | 聚类分组 CSV 和树状图 |
| `bool_hop_stability.py` | 每步 Boolean 可达矩阵 CSV 和热力图 |
| `test_phrase_enhance.py` | 词组挖掘、精确重建和 CLI 输出测试 |
| `test_association.py` | PPMI/NPMI 矩阵与 CLI 回归测试 |
| `test_phrase_entropy.py` | 原文左/右/联合边界熵、词序、上下文边界和输出测试 |
| `requirements-clustering.txt` | 五个脚本共用的依赖列表 |
| `output/` | 默认结果根目录，已在 Git 忽略规则中排除 |

输出目录不存在时会自动创建。重跑覆盖同名文件，目录中其他旧结果仍然保留；例如 `--no-plots` 不会删除已有 PNG。比较不同参数时使用不同的 `--outdir`，关联度/聚类也可配合不同 `--prefix`。当前基础/增强 CSV 为原始计数或布尔值，关联度变换由 `association.py` 单独完成。

查看完整命令帮助：

```powershell
.\.venv\Scripts\python.exe cooccurrence.py --help
.\.venv\Scripts\python.exe phrase_enhance.py --help
.\.venv\Scripts\python.exe association.py --help
.\.venv\Scripts\python.exe phrase_entropy.py --help
.\.venv\Scripts\python.exe cluster_words.py --help
.\.venv\Scripts\python.exe bool_hop_stability.py --help
```

运行项目测试：

```powershell
.\.venv\Scripts\python.exe -m unittest test_phrase_enhance -v
.\.venv\Scripts\python.exe -m unittest test_association test_phrase_entropy -v
.\.venv\Scripts\python.exe -m unittest test_phrase_metrics -v
```

测试覆盖词组原文边界、停用词隔断、段落去重、左右及联合熵的有序词对与边界处理、关联度计算，以及 CSV/PNG 输出和命令行参数校验。测试使用临时目录，不依赖 `1.md` 的具体内容。
