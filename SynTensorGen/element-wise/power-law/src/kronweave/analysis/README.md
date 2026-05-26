# analysis

Optional analysis helpers for generated TensorSuite tensors.

- `__init__.py`: marks this directory as the analysis subpackage.
- `power_law_analysis.py`: computes mode-wise row-degree samples and fits empirical power-law summaries with the optional `powerlaw` package.
- `visualizer.py`: loads TensorSuite-TNS tensors, unfolds modes, slices tensors, and saves mode-wise degree-distribution plots with `matplotlib`.

Install these tools with `pip install -e ".[analysis]"`. The power-law fitting helper cites [powerlaw-devs/powerlaw](https://github.com/powerlaw-devs/powerlaw) and Clauset, Shalizi, and Newman, "Power-Law Distributions in Empirical Data", SIAM Review, 2009.
