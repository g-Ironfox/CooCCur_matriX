# select_bigrams.py

## 功能

按出现次数、PPMI/NPMI 和边界熵筛选有向二词候选。

## 输入与输出

输入为英文 Markdown；输出候选 CSV。

## 输出说明

此脚本只生成候选表，不生成 PNG 图片。候选结果可作为 `phrase_entropy.py`、`phrase_metrics.py` 或 `trigram_potential.py` 的输入；仓库中的真实文本 `3.md` 可用于复现实例候选表。

## 参数

以下为脚本当前提供的完整命令行参数和默认值。

```text
usage: select_bigrams.py [-h] [--outdir OUTDIR] [--prefix PREFIX]
                         [--min-count MIN_COUNT] [--min-ppmi MIN_PPMI]
                         [--min-npmi MIN_NPMI] [--max-entropy MAX_ENTROPY]
                         [--min-length MIN_LENGTH]
                         [input]

Select directed bigrams with strong association and low boundary entropy.

positional arguments:
  input                 Source Markdown (default: 1.md)

options:
  -h, --help            show this help message and exit
  --outdir OUTDIR       Output directory
  --prefix PREFIX       Output filename prefix (default: input filename stem)
  --min-count MIN_COUNT
                        Minimum occurrences after paragraph deduplication
                        (default: 3)
  --min-ppmi MIN_PPMI   Minimum PPMI in natural-log units (default: 0.5)
  --min-npmi MIN_NPMI   Minimum NPMI in [-1, 1] (default: 0.2)
  --max-entropy MAX_ENTROPY
                        Maximum accepted left, right, or joint entropy in nats
                        (default: 0.5)
  --min-length MIN_LENGTH
                        Minimum token length; stopwords are excluded (default:
                        3)
```

参数变更以脚本的 `--help` 输出为准。
