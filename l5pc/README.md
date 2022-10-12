# Neuroscience examples (L5PC and pyloric network)

### Pyloric network

For the APT run without transformation, we ran (`2022_05_17__09_36_25__multirun` and `2022_05_16__11_36_58__multirun`):
``` 
multiround_pyloric.py cores=48 sims_per_round=30000 num_initial=30000 num_rounds=5 ensemble_size=1 thr_proposal=False atomic_loss=True sigmoid_theta=False num_atoms=2
multiround_pyloric.py cores=32 sims_per_round=30000 num_initial=30000 num_rounds=2 ensemble_size=1 start_round=3 path_to_prev_inference=2022_05_16__11_36_58__multirun/0 sigmoid_theta=False thr_proposal=False atomic_loss=True num_atoms=2
```

For the APT run with transformation, we ran (`2022_05_14__00_13_47__multirun`, `2022_05_14__13_21_13__multirun`, `2022_05_14__23_49_35__multirun`):
```
multiround_pyloric.py cores=32 sims_per_round=30000 num_initial=30000 num_rounds=10 ensemble_size=1 thr_proposal=False atomic_loss=True num_atoms=2
multiround_pyloric.py cores=32 sims_per_round=30000 num_initial=30000 num_rounds=10 ensemble_size=1 thr_proposal=False atomic_loss=True num_atoms=2 start_round=7 path_to_prev_inference=2022_05_14__00_13_47__multirun/0
multiround_pyloric.py cores=32 sims_per_round=30000 num_initial=30000 num_rounds=10 ensemble_size=1 thr_proposal=False atomic_loss=True num_atoms=2 start_round=11 path_to_prev_inference=2022_05_14__13_21_13__multirun/0
```

For the TSNPE run, we ran (`2022_05_16__08_53_55__multirun`, `2022_05_17__01_40_01__multirun`):
```
multiround_pyloric.py cores=32 sims_per_round=30000 num_initial=30000 num_rounds=15 sampling_method=rejection ensemble_size=1
multiround_pyloric.py  cores=8 sims_per_round=30000 num_initial=30000 num_rounds=10 sampling_method=sir ensemble_size=1 start_round=8 path_to_prev_inference=2022_05_16__08_53_55__multirun/0
```
