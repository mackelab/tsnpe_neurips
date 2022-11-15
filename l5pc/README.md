# Neuroscience examples (L5PC and pyloric network)

Below are the commands we ran to produce the results on the neuroscience examples. Note: the identifiers in brackets, e.g. `2022_05_17__09_36_25__multirun` indicate the location of the exact log file of that run. This allows to recover the full configurations by inspecting the [development github account of this project](https://github.com/mackelab/tsnpe_neurips_dev/tree/main/l5pc/results).

### Pyloric network

For the APT run without transformation, we ran (`2022_05_17__09_36_25__multirun`, `2022_05_16__11_36_58__multirun`):
``` 
python python multiround_pyloric.py cores=48 sims_per_round=30000 num_initial=30000 num_rounds=5 ensemble_size=1 thr_proposal=False atomic_loss=True sigmoid_theta=False num_atoms=2
python multiround_pyloric.py cores=32 sims_per_round=30000 num_initial=30000 num_rounds=2 ensemble_size=1 start_round=3 path_to_prev_inference=2022_05_16__11_36_58__multirun/0 sigmoid_theta=False thr_proposal=False atomic_loss=True num_atoms=2
```

For the APT run with transformation, we ran (`2022_05_14__00_13_47__multirun`, `2022_05_14__13_21_13__multirun`, `2022_05_14__23_49_35__multirun`):
```
python multiround_pyloric.py cores=32 sims_per_round=30000 num_initial=30000 num_rounds=10 ensemble_size=1 thr_proposal=False atomic_loss=True num_atoms=2
python multiround_pyloric.py cores=32 sims_per_round=30000 num_initial=30000 num_rounds=10 ensemble_size=1 thr_proposal=False atomic_loss=True num_atoms=2 start_round=7 path_to_prev_inference=2022_05_14__00_13_47__multirun/0
python multiround_pyloric.py cores=32 sims_per_round=30000 num_initial=30000 num_rounds=10 ensemble_size=1 thr_proposal=False atomic_loss=True num_atoms=2 start_round=11 path_to_prev_inference=2022_05_14__13_21_13__multirun/0
```

For the TSNPE run, we ran (`2022_05_16__08_53_55__multirun`, `2022_05_17__01_40_01__multirun`):
```
python multiround_pyloric.py cores=32 sims_per_round=30000 num_initial=30000 num_rounds=15 sampling_method=rejection ensemble_size=1
python multiround_pyloric.py  cores=8 sims_per_round=30000 num_initial=30000 num_rounds=10 sampling_method=sir ensemble_size=1 start_round=8 path_to_prev_inference=2022_05_16__08_53_55__multirun/0
```

### Layer 5 pyramidal cell

For producing the results of the L5PC, run (`2022_04_27__19_06_37_snpe`, `2022_05_05__07_47_34_snpe_m/0`, `2022_05_05__21_28_21_snpe`, `2022_05_06__13_50_43_snpe`, `2022_05_07__08_17_53_snpe`, `2022_05_08__08_59_11_snpe`):
```
python train_from_disk.py id=l20_7 num_initial=30000 load_nn_from_prev_inference=False ensemble_size=10
python train_from_disk.py id=l20_7 num_initial=30000 previous_inference=2022_04_27__19_06_37_snpe
python train_from_disk.py id=l20_7 num_initial=30000 previous_inference=2022_05_05__07_47_34_snpe_m/0
python train_from_disk.py id=l20_7 num_initial=30000 previous_inference=2022_05_05__21_28_21_snpe
python train_from_disk.py id=l20_7 num_initial=30000 previous_inference=2022_05_06__13_50_43_snpe
python train_from_disk.py id=l20_7 num_initial=30000 previous_inference=2022_05_07__08_17_53_snpe
```
Please replace the `previous_inference` attribute with the location of the previous run you made (it depends on date and time and thus will not match the above date/time). Alternatively, download all saved networks [here](https://github.com/mackelab/tsnpe_neurips_dev/tree/main/l5pc/results).

Note that this procedure will simply load the simulated data [from disk](https://github.com/mackelab/tsnpe_neurips/tree/main/l5pc/results/simulations_pickle). If you want indeed want to run the simulations yourself, you have to run (before every round of training) the script `sample_and_simulate.py` with configurations specified [here](https://github.com/mackelab/tsnpe_neurips_dev/tree/main/l5pc/results/l20_7/simulations), e.g.
```
python sample_and_simulate.py id=l20_7 cores=48 sims=32000 proposal=2022_04_27__19_06_37_snpe
```

Running `sample_and_simulate.py` requires to set up a [datajoint](https://www.datajoint.org/) database. Note that this is not required for the results shown in Fig. 1-5 and is also not required to run the notebook which generates Fig. 6. 

If you still want to set up the database, you have to set up a database server. This is described in detail [here](https://github.com/datajoint/mysql-docker). Here is an outline of the steps:
1) [Install docker](https://docs.docker.com/engine/install/ubuntu/)
2) [Install docker compose](https://docs.docker.com/compose/install/linux/)

Then, set up the database:
```
mkdir mysql-docker
cd mysql-docker
wget https://raw.githubusercontent.com/datajoint/mysql-docker/master/docker-compose.yaml
docker-compose up -d
```
