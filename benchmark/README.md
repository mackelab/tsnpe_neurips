# Benchmark tasks

This folder contains code to run TSNPE on the benchmark tasks (Fig. 4). In order to run this code, you have to execute, for every task:
```
python run.py -m task=sir task.num_observation=1,2,3,4,5,6,7,8,9,10 task.num_simulations=1000,10000,100000
```

The raw outputs of this code as well as the individual configuration files are saved [here](https://github.com/mackelab/tsnpe_neurips_dev/benchmark/tree/main/results/). After all runs were finished, we collected the the data into a pickle file with:
```python
from benchmark.utils import compile_df
df = compile_df("../../../benchmark/results/results_thr_point01percent")
df.to_pickle("../../results/benchmark_fig/results_thr_point01percent.pkl")
```
in the notebooks in `paper/fig4/01_generate_data.ipynb`.
