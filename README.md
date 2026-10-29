# CooCCur-matriX

从英文 Markdown 提取相邻词关系，生成邻接矩阵，并支持词组分析、关联度计算、聚类和网络结构分析。

## 安装

在项目根目录的 Windows PowerShell 中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-clustering.txt
```

Leiden/Louvain 社区分析需要额外安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-community.txt
```

## 常用流程

以下示例以 `1.md` 为输入，结果写入 `output/` 及其子目录。可以将 `1.md` 换成其他英文 Markdown 路径。

### 共现矩阵、关联度与聚类

```powershell
.\.venv\Scripts\python.exe cooccurrence.py 1.md
.\.venv\Scripts\python.exe association.py output/directed_diag_zero.csv --outdir output/association
.\.venv\Scripts\python.exe cluster_words.py output/directed_diag_zero.csv --direction directed --clusters 6 --outdir output/clusters
```

关联度和聚类是可选的下游步骤，也可以直接读取 `phrase_enhance.py` 生成的矩阵。

### 词组增强

```powershell
.\.venv\Scripts\python.exe phrase_enhance.py 1.md --outdir output/phrase
```

此脚本从原文识别连续二词组合并重建矩阵；单靠矩阵无法还原词序和原文匹配位置。

### 社区检测与重排

```powershell
.\.venv\Scripts\python.exe block_diagonal_reorder.py output/undirected_diag_zero.csv --method leiden
```

更多选项和默认值见各脚本的 `--help`。

## 脚本索引

每篇说明配有基于 `2.md` 的全量输出例图；[数据范围与复现命令](docs/scripts/examples.md)统一说明参数和分页规则。

| 脚本 | 功能与参数 |
| --- | --- |
| `cooccurrence.py` | [说明](docs/scripts/cooccurrence.md)：基础共现矩阵 |
| `phrase_enhance.py` | [说明](docs/scripts/phrase_enhance.md)：词组挖掘与矩阵增强 |
| `association.py` | [说明](docs/scripts/association.md)：PPMI/NPMI 关联度 |
| `select_bigrams.py` | [说明](docs/scripts/select_bigrams.md)：二词候选筛选 |
| `phrase_entropy.py` | [说明](docs/scripts/phrase_entropy.md)：词组边界熵 |
| `phrase_metrics.py` | [说明](docs/scripts/phrase_metrics.md)：词组综合指标 |
| `trigram_potential.py` | [说明](docs/scripts/trigram_potential.md)：三词扩展分析 |
| `cluster_words.py` | [说明](docs/scripts/cluster_words.md)：邻接模式层次聚类 |
| `block_diagonal_reorder.py` | [说明](docs/scripts/block_diagonal_reorder.md)：社区检测与矩阵重排 |
| `bool_hop_stability.py` | [说明](docs/scripts/bool_hop_stability.md)：Boolean 多跳可达性 |
| `katz_stability.py` | [说明](docs/scripts/katz_stability.md)：Katz 分数与收敛分析 |
| `weighted_word_stats.py` | [说明](docs/scripts/weighted_word_stats.md)：加权节点统计 |
| `weighted_inout_scatter.py` | [说明](docs/scripts/weighted_inout_scatter.md)：入/出边散点图 |
| `inout_uv_scatter.py` | [说明](docs/scripts/inout_uv_scatter.md)：入/出分数 u-v 图 |
| `adjacency_embedding_viz.py` | [说明](docs/scripts/adjacency_embedding_viz.md)：PCA 邻接特征图 |
| `node_removal_connectivity.py` | [说明](docs/scripts/node_removal_connectivity.md)：节点移除连通性分析 |
| `textrank.py` | [说明](docs/scripts/textrank.md)：TextRank 排名 |
| `compare_rankings.py` | [说明](docs/scripts/compare_rankings.md)：多种排名方法比较 |

## 数据约定

- 清理 Markdown 并对段落去重后，按换行、英文句点和逗号切分片段；只统计片段内的相邻词。
- 基础共现先过滤停用词和短于 3 个字符的词，再统计相邻关系，因此 `alpha and beta` 会产生 `alpha → beta`。词组挖掘要求原文连续匹配，不跨过停用词。
- 基础共现默认保留至少出现在 2 个片段中的词；词频按包含该词的片段数计算，不是总出现次数。
- CSV 矩阵首行和首列为词标签，数值区为方阵；行标签与列标签顺序一致。有向矩阵的行是前词、列是后词。
- 六种矩阵如下；词组增强结果使用相同名称，并增加 `phrase_enhanced_` 前缀。关联度和聚类默认忽略对角线。

| 矩阵名称（不含 `.csv`） | 边权 | 对角线 |
| --- | --- | --- |
| `directed_diag_count` | 有向相邻次数 | 片段频率 |
| `directed_diag_zero` | 有向相邻次数 | 0 |
| `undirected_diag_count` | 两个方向的次数之和 | 片段频率 |
| `undirected_diag_zero` | 两个方向的次数之和 | 0 |
| `directed_bool` | 有向边是否存在（0/1） | 0 |
| `undirected_bool` | 无向边是否存在（0/1） | 0 |

## 调参与输出

- `--plot-top-words` 只限制图片显示的词数，设为 `0` 显示全部，不改变 CSV。基础共现和词组增强的图片默认显示 40 项。
- `phrase_enhance.py` 的 `--top-words` 会限制矩阵词表，默认 40，设为 `0` 保留全部满足频率要求的词和词组。
- 没有词组通过筛选时，查看 `phrase_candidates.csv`，再调整 `--min-phrase-frequency` 和 `--min-npmi`；已入选但未出现在矩阵中的候选，可检查 `merged_count`、`included_in_matrix` 及词表筛选条件。
- 聚类的 `--clusters` 不得超过非孤立词数；已知输入有向时，显式传入 `--direction directed`。
- 输出目录会自动创建。重跑会覆盖同名文件，不会清理目录中的其他旧结果；不同参数的结果建议写入不同 `--outdir`。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py" -v
```

单独查看脚本选项，例如：

```powershell
.\.venv\Scripts\python.exe phrase_enhance.py --help
```
