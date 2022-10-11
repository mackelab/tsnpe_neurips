# Explanation for config keywords

### Train
- id: The id of the run, e.g `l20_0`
- seed_train: The seed used for training the nn
- previous_inference: If multi-round is run, this is the path to the previous inference object. E.g.
  `"2021_11_02__15_30_53__snpe"`. If `load_nn_from_prev_inference=False`, this is only used to infer the round we are
  (which in turn influences the used dataset). In addition, if `choose_features="valid_unused"`, it is used also to
  get which features were trained on in the first round.
- load_nn_from_prev_inference: If `True`, the inference object specified in `previous_inference` will be loaded and used
  to train. If `False`, a new inference object will be created from scratch.
- nan_fraction_threshold_to_exclude: The fraction of simulations that have to be valid for **a specific feature** for
  the feature to be included in the training procedure. E.g. `0.9` requires 90% valid.
- max_num_epochs: Max number of epochs to train.
- training_batch_size: Training batchsize.
- num_train: Max number of training datapoints.
- num_initial: Number of datapoints that are loaded in the beginning.
- density_estimator: Density estimator.
- observation_noise: Only used if `observation_noise_type="sims"`. In that case, it is the multiplier to the standard
  deviation of the prior samples in order to obtain the std of the Gaussian observation noise.
- observation_noise_type: Either of `["data", "sims"]`. If `"sims"`, the prior simulations are used to infer the std of
  the observation noise. If `"data"`, the standard deviation reported in Hay et al 2011 Table 1 is used.
- ensemble_size: How many networks in ensemble.
- choose_features: Either of `["valid", "valid_unused"]`. If `"valid"`, train on all features that pass the criterion
  specified by `nan_fraction_threshold_to_exclude`. If `"valid_unused"`, it additionally excludes features that had
  already been trained on in previous rounds. This mode is envisioned if one wants to fit e.g. protocols sequentially
  and use the previous posterior as new prior. Assuming that the likelihood factorizes, one does not have to use
  correction terms.
- replace_nan_values: Whether or not to replace `NaN` values with a value that is the min of the feature (in the prior
  predictives) minus 2 stds. Note that, even if this feature is `True`, we will only train on the data that passes the
  test given by `nan_fraction_threshold_to_exclude`. If `True`, then we automatically use `temper_xo=True`.
- train_on_all: If `True`, we load all simulations and replace the `NaN` with the same 
  replacement values as used when using `replace_nan_values`. If `True`, then 
  `nan_fraction_threshold_to_exclude` is ignored.
- temper_xo: Used only if `replace_nan_values=True`. Then, if `temper_xo=True`, we set 
  those features of `xo` which have too many NaN values (according to `nan_fraction_threshold_to_exclude`) to 
  the replacement value
- data_path: only used if `model=="pyloric"`. Then, this path is the path to the simulations that are used to train.
- simulation_loaded_from_id: Allows to train on a different id than the id of the data. This can be useful for SNPE-C (because it works for any proposal).


### Evaluation
- id: The id of the run, e.g `l20_0`
- seed: The seed used for the coverage.
- num_predictives: Number of posterior predictives that are simulated and evaluated.
- posterior: The posterior to evaluate, e.g. `"2021_11_02__15_30_53__snpe"`
- cores: The number of cores used to generate the posterior predictives.

### Model
- id: The id of the run, e.g `l20_0`
- proposal: The proposal from which the parameters are drawn. Default is `prior`.
- seed_prior: The seed used for the proposal samples.
- cores: The number of cores used to generate the simulations.
- sims: Number of simulations to run.
- sims_until_save: `sims` are run in batches of this size. After each batch, the data is written to the database.
- sims_per_worker: Number of simulations that are passed to a specific worker.
- thr_proposal: If `False`, draw parameters from the `proposal`. If `True`, draw parameters from the support of the proposal.
- save_sims: Whether or not to write the generated simulations to the database.