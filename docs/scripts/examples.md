# 2.md 全量输出示例

本文按数据依赖拆解所有脚本例图的复现流程。示例统一使用仓库根目录的 `2.md`，输出写入 `output/docs_examples/2/`；不会覆盖项目默认的 `output/` 结果。请在仓库根目录的同一个 PowerShell 会话中，按步骤执行变量初始化和所需命令。

## 数据范围与图表规则

- 基础共现矩阵保留满足英文分词、默认停用词、最小词长和 `--min-frequency 1` 规则的全部 481 个词，不按词频截取 Top N。
- 词组增强仍根据二词频次、NPMI 和合并后片段频次筛选候选；`--top-words 0` 保留筛选后全部词和词组，本例得到 491 个矩阵节点。
- 矩阵热力图均显示脚本筛选后矩阵的完整词轴，不再按 Top-N 截断。词组熵不设短语频次过滤。
- Bool-hop 的可达矩阵和加权矩阵显示完整 481 节点；逐词统计按 `min(incoming, outgoing)` 排序，仅显示前 50 个词。
- 其他排名统计设 `--top 0` 显示全部条目。散点图包含全部节点；分页只改变词标签，每页仍绘制所有点。PCA 图绘制全部节点并标注前 15 个。

## 步骤 0：准备输出目录变量

设置 Python 路径、统一输出根目录和后续脚本共用的文件路径。以下命令假设已按 README 安装依赖；Leiden 社区分析还需要安装 `requirements-community.txt`。

```powershell
$python = ".\.venv\Scripts\python.exe"
$root = "output/docs_examples/2"
$base = "$root/cooccurrence"
$matrix = "$base/directed_diag_zero.csv"
$boolean = "$base/directed_bool.csv"
$hop = "$root/bool_hop_stability"
```

## 步骤 1：从 Markdown 生成基础共现矩阵

`cooccurrence.py` 清理并分词 `2.md`，输出六种矩阵及热力图。后续关联度、聚类、重排、Bool-hop、Katz 和 TextRank 示例都使用这里生成的矩阵；本例将有向、对角线为零的计数矩阵保存为 `$matrix`，Boolean 邻接矩阵保存为 `$boolean`。

```powershell
& $python cooccurrence.py 2.md --outdir "$base" --min-frequency 1 --plot-top-words 0
```

## 步骤 2：执行词组分析

以下分析都直接读取原文，因为要使用段落、词序和短语边界信息。

`phrase_enhance.py` 挖掘并合并连续二词组合，输出增强矩阵。候选最低出现 3 次、NPMI 不低于 0，合并后至少出现在 1 个片段；`--top-words 0 --plot-top-words 0` 保留筛选后全部节点并绘制完整词轴。

```powershell
& $python phrase_enhance.py 2.md --outdir "$root/phrase_enhance" --min-phrase-frequency 3 --min-npmi 0 --min-frequency 1 --top-words 0 --plot-top-words 0
```

`phrase_entropy.py` 计算所有符合默认分词规则的有向二词组左右边界熵；短语频次设为 0，不因频次删去短语或源词轴。

```powershell
& $python phrase_entropy.py 2.md --outdir "$root/phrase_entropy" --min-phrase-frequency 0 --plot-top-words 0
```

`phrase_metrics.py` 在统一词轴上计算左熵、右熵、PPMI 和 NPMI，不设置绘图频次门槛并显示完整词轴。

```powershell
& $python phrase_metrics.py 2.md --outdir "$root/phrase_metrics" --plot-top-words 0
```

## 步骤 3：筛选二词候选并分析三词扩展

`select_bigrams.py` 从原文筛选二词候选，输出候选 CSV，不生成图。这里将候选表写入 `$root/select_bigrams/`，供下一条命令使用。

```powershell
& $python select_bigrams.py 2.md --outdir "$root/select_bigrams" --min-count 3 --min-ppmi 0 --min-npmi 0 --max-entropy 2
```

`trigram_potential.py` 必须同时读取原文和上一步的候选 CSV，统计候选二词组左右相邻的三词扩展。`--top 0` 绘制所有通过 `--min-trigram-count 1` 的扩展。

```powershell
& $python trigram_potential.py 2.md --phrase-candidates "$root/select_bigrams/2_bigram_candidates.csv" --outdir "$root/trigram_potential" --min-trigram-count 1 --top 0
```

## 步骤 4：聚类与社区重排

下面两个脚本都读取步骤 1 生成的 `$matrix`，彼此独立。

`cluster_words.py` 按有向邻接模式做层次聚类，划分 6 类并输出聚类标签和树状图。

```powershell
& $python cluster_words.py "$matrix" --direction directed --clusters 6 --outdir "$root/cluster_words"
```

`block_diagonal_reorder.py` 用 Leiden 检测社区，重排完整矩阵并绘制社区块结构。此步骤需要社区分析依赖。

```powershell
& $python block_diagonal_reorder.py "$matrix" --method leiden --outdir "$root/block_diagonal"
```

## 步骤 5：计算 Boolean 多跳可达性与 Katz 矩阵

`bool_hop_stability.py` 读取 `$boolean`，逐阶计算累计可达矩阵，直到稳定，并同时生成线性加权 hop 矩阵。本例稳定于第 21 hop；逐词统计按 `min(incoming, outgoing)` 排序并取前 50。

```powershell
& $python bool_hop_stability.py "$boolean" --outdir "$hop" --top 50 --sort-by min --cell-size 0.18
```

`katz_stability.py` 也读取 `$boolean`，但使用普通数值矩阵乘法累加路径贡献。本例 `beta=0.01`，迭代至收敛，并输出完整 Katz 矩阵和统计图。

```powershell
& $python katz_stability.py "$boolean" --outdir "$root/katz_stability" --beta 0.01 --top 0 --cell-size 0.18
```

## 步骤 6：分析稳定 hop 的节点分数

以下脚本共用 Bool-hop 稳定阶的线性加权矩阵。先从 `$hop` 中选择最大阶矩阵，保存为 `$weighted`；这一步依赖步骤 5 的 Bool-hop 命令已成功完成。

```powershell
$weighted = (Get-ChildItem "$hop/weighted_*.csv" | Where-Object { $_.Name -match "^weighted_[0-9]+[.]csv$" } | Sort-Object Name | Select-Object -Last 1).FullName
```

`weighted_word_stats.py` 输出全部 481 个节点的入边、出边统计，图表也显示全部条目。

```powershell
& $python weighted_word_stats.py "$weighted" --outdir "$root/weighted_word_stats" --top 0
```

`weighted_inout_scatter.py` 绘制入边和出边总分。全部节点都会显示在每张图中，标签按 40 个一页分页。

```powershell
& $python weighted_inout_scatter.py "$weighted" --outdir "$root/weighted_inout_scatter" --top 0 --page-size 40
```

`inout_uv_scatter.py` 将相同的入/出分数转换到 `u-v` 坐标；分页同样只切换标签，不过滤散点。

```powershell
& $python inout_uv_scatter.py "$weighted" --outdir "$root/inout_uv_scatter" --top 0 --page-size 40
```

## 步骤 7：运行其他网络分析

以下脚本读取步骤 1 的矩阵，彼此独立。

`adjacency_embedding_viz.py` 将全部节点投影到二维，并标记前 15 个高权重节点。

```powershell
& $python adjacency_embedding_viz.py "$matrix" --outdir "$root/adjacency_embedding_viz" --label-top 15
```

`node_removal_connectivity.py` 分析逐个移除节点对弱连通分量的影响，`--top 0` 保留完整排名图。

```powershell
& $python node_removal_connectivity.py "$boolean" --outdir "$root/node_removal_connectivity" --top 0
```

`textrank.py` 根据有向共现矩阵计算节点排名并绘制完整结果。

```powershell
& $python textrank.py "$matrix" --outdir "$root/textrank" --top 0
```

`compare_rankings.py` 对照 Boolean 多跳、Katz 和 TextRank 的节点排名；`--top 0` 显示全部节点。

```powershell
& $python compare_rankings.py "$boolean" --outdir "$root/compare_rankings" --top 0
```

每篇脚本文档中的例图从上述流程对应的输出生成，并复制到 `docs/scripts/assets/`，因此查看文档图片不需要先运行这些命令。热力图的“全量”是指完整显示对应脚本筛选后矩阵的词轴；停用词、最小词长及明确设置的候选阈值仍然生效。
