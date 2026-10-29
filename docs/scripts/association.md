# association.py

## 功能

从共现计数矩阵计算 PPMI/NPMI，并生成矩阵和热力图。

## 输入与输出

输入为共现计数矩阵 CSV；输出目录包含 PPMI/NPMI CSV 和图像。

## 输出示例

下图使用 `2.md` 生成的完整 `directed_diag_zero.csv`，PPMI/NPMI 对照图包含全部 481 个词项。复现时使用 `--plot-top-words 0`。

![association.py 输出示例](assets/association.png)

## 参数

以下为脚本当前提供的完整命令行参数和默认值。

```text
usage: association.py [-h] [--outdir OUTDIR] [--prefix PREFIX]
                      [--include-diagonal] [--plot-top-words PLOT_TOP_WORDS]
                      [--figure-size FIGURE_SIZE]
                      input

Compute and plot PPMI/NPMI from a co-occurrence count matrix.

positional arguments:
  input                 Count matrix CSV from cooccurrence.py

options:
  -h, --help            show this help message and exit
  --outdir OUTDIR       Output directory (default: output)
  --prefix PREFIX       Output filename prefix (default: input filename)
  --include-diagonal    Include diagonal counts in probability estimates
                        (default: exclude)
  --plot-top-words PLOT_TOP_WORDS
                        Words shown in heatmaps by total co-occurrence; use 0
                        for all (default: 40)
  --figure-size FIGURE_SIZE
                        Figure width in inches (default: based on word count)
```

参数变更以脚本的 `--help` 输出为准。
