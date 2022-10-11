import logging
from multiprocessing import Pool
import pickle
import time
import os
from tokenize import Name
import torch
from omegaconf import DictConfig
import hydra

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# These files live in utils because I otherwise had problems with SLURM and
# multiprocessing. See this error: https://www.pythonanywhere.com/forums/topic/27818/
from l5pc.utils.simulation_utils import (
    assemble_prior,
    assemble_simulator,
    assemble_pyloric,
    assemble_db,
    write_to_dj,
)
from l5pc.model.utils import return_gt, return_names, return_xo
from sbi.utils import BoxUniform
from sbi.utils.support_posterior import PosteriorSupport
from os.path import join
from torch import tensor, as_tensor, float32
from pyloric import create_prior, simulate, summary_stats
from sbi.inference import SNPE
from sbi.utils import posterior_nn
import dill
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
from l5pc.utils.model_utils import replace_nan
import matplotlib as mpl
import matplotlib.pyplot as plt
from l5pc.utils.evaluation_utils import show_traces_pyloric
from sbi.analysis import pairplot

from sbi.utils.user_input_checks import process_prior


log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="multiround")
def multiround(cfg: DictConfig) -> None:
    print(cfg)
    assert cfg.id is not None, "Specify an ID. Format: [model][dim]_[run], e.g. j2_3"

    for r in range(cfg.start_round, cfg.num_rounds + cfg.start_round):
        log.info(f"============= Starting round {r} =============")
        if r > 1:
            simulate_mp(cfg, r)
        train(cfg, r)
        evaluate(cfg, r)


def simulate_mp(cfg, round_):
    start_time = time.time()
    prior = create_prior().numerical_prior
    sim_and_stats = assemble_pyloric()

    log.info(f"Assembled! {time.time() - start_time}")

    seed = (
        int((time.time() % 1) * 1e7)
        if cfg.seed_simulator is None
        else cfg.seed_simulator
    )
    _ = torch.manual_seed(seed)
    np.savetxt(f"seed.txt", [seed], fmt="%d")

    log.info(f"Starting loop! {time.time() - start_time}")

    log.info(f"Round {round_} in simulate_mp")
    if round_ == 1:
        log.info(f"Using prior as proposal")
        proposal = prior
    else:
        log.info(f"Loading posterior")
        if cfg.path_to_prev_inference is None or round_ > cfg.start_round:
            posterior, _ = load_pyloric_posterior(round_ - 1)
        else:
            posterior, _ = load_pyloric_posterior_from_file(
                cfg.path_to_prev_inference, round_ - 1
            )
        log.info(f"Loaded posterior")
        if cfg.thr_proposal:
            _ = torch.manual_seed(0)  # Set seed=0 only for building the proposal.
            proposal = PosteriorSupport(
                prior=prior,
                posterior=posterior,
                num_samples_to_estimate_support=cfg.num_samples_to_estimate_support,
                allowed_false_negatives=cfg.allowed_false_negatives,
                use_constrained_prior=cfg.use_constrained_prior,
                constrained_prior_quanitle=cfg.constrained_prior_quanitle,
                sampling_method=cfg.sampling_method,
            )
            _, acceptance_rate = proposal.sample((10,), return_acceptance_rate=True)
            log.info(f"Acceptance rate of support proposal: {acceptance_rate}")
            log.info("Built support")
            _ = torch.manual_seed(seed)
        else:
            log.info("Setting posterior as proposal")
            proposal = posterior

    log.info(f"num_to_simulate {cfg.sims_per_round}")
    theta = proposal.sample((cfg.sims_per_round,))

    # log.info(f"prior of proposal: {proposal._prior}")

    log.info(f"Sampled proposal {theta.shape}")
    if isinstance(theta, torch.Tensor):
        prior_pd = create_prior()
        sss = prior_pd.sample((1,))
        pyloric_names = sss.columns
        theta = pd.DataFrame(theta.numpy(), columns=pyloric_names)

    theta_full = theta

    log.info(f"Time to obtain theta: {time.time() - start_time}")

    # Each worker should process a batch of simulations to reduce the overhead of
    # loading neuron.
    num_splits = cfg.sims_per_round
    batches = np.array_split(theta_full, num_splits)
    batches = [b.iloc[0] for b in batches]

    log.info(f"Time to obtain batches: {time.time() - start_time}")

    with Pool(cfg.cores) as pool:
        x_list = pool.map(sim_and_stats, batches)

    log.info(f"Sims done {time.time() - start_time}")
    x = pd.concat(x_list, ignore_index=True)

    stats = x.to_numpy()
    valid = np.invert(np.any(np.isnan(stats), axis=1))
    log.info(f"Fraction of valid sims in simulate: {np.sum(valid) / len(stats)}")

    log.info(f"Sims concatenated {time.time() - start_time}")

    log.info(f"Saving {len(x)} simulations")
    theta_full.to_pickle(f"sims_theta_r{round_}.pkl")
    x.to_pickle(f"sims_x_r{round_}.pkl")


def train(cfg, round_):

    if round_ == 1:
        previous_inferences = [None] * cfg.ensemble_size
        train_proposal = None
    else:
        if cfg.path_to_prev_inference is None or round_ > cfg.start_round:
            log.info("Load posterior from current directory")
            prev_posterior, previous_inferences = load_pyloric_posterior(round_ - 1)
        else:
            log.info("Load posterior from a different directory")
            prev_posterior, previous_inferences = load_pyloric_posterior_from_file(
                cfg.path_to_prev_inference, round_ - 1
            )
        if cfg.atomic_loss:
            # This is just a dummy. The proposal is never evaluated in APT anyways.
            train_proposal = prev_posterior.posteriors[0]
        else:
            train_proposal = None
        # round_ = previous_inferences[0]._round + 1 + 1
    log.info(f"Round: {round_}")

    if round_ == 1:
        path = "/mnt/qb/macke/mdeistler57/tsnpe_collection/l5pc/results/p31_4/prior_predictives_energy_paper"
        theta = pd.read_pickle(join(path, "all_circuit_parameters.pkl"))
        x = pd.read_pickle(join(path, "all_simulation_outputs.pkl"))
        log.info(f"Pre-loaded {len(x)} simulations from file.")
    else:
        theta = pd.read_pickle(f"sims_theta_r{round_}.pkl")
        x = pd.read_pickle(f"sims_x_r{round_}.pkl")
    theta = as_tensor(np.asarray(theta), dtype=float32)
    x = as_tensor(np.asarray(x), dtype=float32)
    x = x[:, :18]

    theta = theta[: cfg.num_initial]
    x = x[: cfg.num_initial]

    prior = create_prior().numerical_prior

    log.info(f"theta dim after loading id: {theta.shape}")
    log.info(f"x dim after loading id: {x.shape}")

    if cfg.replace_nan_values:
        x, replacement_values = replace_nan(x, model="pyloric")

    log.info(f"theta dim to train: {theta.shape}")
    log.info(f"x dim to train: {x.shape}")

    prior_for_bounds = create_prior().numerical_prior
    dens_estim = posterior_nn(
        cfg.density_estimator,
        sigmoid_theta=cfg.sigmoid_theta,
        prior=prior_for_bounds,
    )

    if round_ > 1 or cfg.retrain_first_round:
        if cfg.parallel_training:
            raise NotImplementedError
            inferences = Parallel(n_jobs=cfg.ensemble_size)(
                delayed(train_given_seed)(
                    cfg,
                    previous_inferences[seed],
                    prior,
                    dens_estim,
                    theta,
                    x,
                    seed,
                    train_proposal,
                    round_,
                )
                for seed in range(cfg.ensemble_size)
            )
        else:
            inferences = []
            for seed in range(cfg.ensemble_size):
                _ = torch.manual_seed(cfg.seed_train + seed)
                if round_ > 1:
                    inference = previous_inferences[seed]
                else:
                    log.info(f"dens_estim {dens_estim}")
                    log.info(f"Initializiating SNPE with prior {prior}")
                    inference = SNPE(prior=prior, density_estimator=dens_estim)
                log.info(f"train_proposal {train_proposal}")
                _ = inference.append_simulations(
                    theta, x, proposal=train_proposal
                ).train(
                    max_num_epochs=cfg.max_num_epochs,
                    training_batch_size=cfg.training_batch_size,
                    stop_after_epochs=cfg.stop_after_epochs,
                    force_first_round_loss=True,
                    num_atoms=cfg.num_atoms,
                )
                inferences.append(inference)
                log.info(f"_best_val_log_prob {inference._best_val_log_prob}")
    else:
        raise NotImplementedError
        log.info("Loading first round inference from file.")
        pa = "/mnt/qb/macke/mdeistler57/tsnpe_collection/l5pc/results/p31_2/multiround"
        with open(join(pa, "2022_05_04__08_18_59/inference_r1.pkl"), "rb") as handle:
            inferences = pickle.load(handle)
        inferences = inferences[: cfg.ensemble_size]

    with open(f"inference_r{round_}.pkl", "wb") as handle:
        dill.dump(inferences, handle)

    all_val_log_probs = [infer._summary["validation_log_probs"] for infer in inferences]
    all_best_val = [infer._best_val_log_prob for infer in inferences]
    all_epochs = [infer.epoch for infer in inferences]
    with open("val_log_probs.pkl", "wb") as handle:
        pickle.dump(all_val_log_probs, handle)
    np.savetxt("best_val_log_prob.txt", all_best_val, fmt="%10.10f")
    np.savetxt("epochs.txt", all_epochs, fmt="%5.5f")


def evaluate(cfg, round_):
    _ = torch.manual_seed(cfg.seed_evaluate)
    posterior, inferences = load_pyloric_posterior(round_)
    prior = create_prior()
    lower = prior.numerical_prior.support.base_constraint.lower_bound
    upper = prior.numerical_prior.support.base_constraint.upper_bound
    prior_bounds = torch.stack([lower, upper]).T.numpy()

    theta_torch = posterior.sample((cfg.num_predictives,))
    prior_samples = prior.sample((1,))
    theta = pd.DataFrame(theta_torch.numpy(), columns=prior_samples.columns)
    num_splits = cfg.num_predictives
    batches = np.array_split(theta, num_splits)
    batches = [b.iloc[0] for b in batches]

    log.info("Starting to simulate in evaluate()")
    with Pool(cfg.cores) as pool:
        x_list = pool.map(simulate, batches)
    log.info("Finished simulation in evaluate()")

    stats = pd.concat([summary_stats(xx) for xx in x_list])
    stats = stats.to_numpy()
    valid = np.invert(np.any(np.isnan(stats), axis=1))
    log.info(f"Fraction of valid sims: {np.sum(valid) / cfg.num_predictives}")
    with mpl.rc_context(
        fname="/mnt/qb/macke/mdeistler57/tsnpe_collection/l5pc/.matplotlibrc"
    ):
        show_traces_pyloric(x_list)
        plt.savefig(f"traces_r{round_}.png")

        posterior_samples = posterior.sample((1000,), show_progress_bars=False)
        _ = pairplot(
            posterior_samples,
            limits=prior_bounds,
            upper=["kde"],
            ticks=prior_bounds,
            figsize=(10, 10),
        )
        plt.savefig(f"pairplot_r{round_}.png")

    # if round_ == 1:
    #     path = "/mnt/qb/macke/mdeistler57/tsnpe_collection/l5pc/results/p31_2/prior_predictives_energy_paper"
    #     theta = pd.read_pickle(join(path, "all_circuit_parameters.pkl"))
    #     x = pd.read_pickle(join(path, "all_simulation_outputs.pkl"))
    #     log.info(f"Evaluation: Pre-loaded {len(x)} simulations from file.")
    # else:
    #     theta = pd.read_pickle(f"sims_theta_r{round_}.pkl")
    #     x = pd.read_pickle(f"sims_x_r{round_}.pkl")
    # theta = as_tensor(np.asarray(theta), dtype=float32)
    # x = as_tensor(np.asarray(x), dtype=float32)
    # x = x[:, :18]

    # alpha, cov = coverage(posterior, theta, x, used_features)


def train_given_seed(
    cfg,
    previous_inferences_specific,
    prior,
    dens_estim,
    theta,
    x,
    seed,
    train_proposal,
    round_,
):
    _ = torch.manual_seed(cfg.seed_train + seed)
    if round_ > 1:
        inference = previous_inferences_specific
    else:
        inference = SNPE(prior=prior, density_estimator=dens_estim)

    x_np = x.numpy()
    valid = np.invert(np.any(np.isnan(x_np), axis=1))
    log.info(f"Fraction of valid sims in train_given_seed: {np.sum(valid) / len(x)}")

    _ = inference.append_simulations(theta, x, proposal=train_proposal).train(
        max_num_epochs=cfg.max_num_epochs,
        training_batch_size=cfg.training_batch_size,
        stop_after_epochs=cfg.stop_after_epochs,
        force_first_round_loss=True,
        num_atoms=cfg.num_atoms,
    )
    log.info(f"_best_val_log_prob {inference._best_val_log_prob}")
    return inference


# def load_classifier_prior(wrap: bool = True):
#     with open(
#         "/mnt/qb/macke/mdeistler57/tsnpe_collection/l5pc/results/pyloric_restricted_prior.pkl",
#         "rb",
#     ) as handle:
#         classifier = pickle.load(handle)
#     if wrap:
#         classifier, _, _ = process_prior(classifier)
#     return classifier


def load_pyloric_posterior(round_):
    with open(f"inference_r{round_}.pkl", "rb") as handle:
        inferences = dill.load(handle)
    xo = torch.as_tensor(
        [
            1.17085859e03,
            2.06036434e02,
            2.14307031e02,
            4.12842187e02,
            1.75970382e-01,
            1.83034085e-01,
            3.52597820e-01,
            4.11600328e-01,
            6.30544893e-01,
            4.81925781e02,
            2.56353125e02,
            2.75164844e02,
            4.20460938e01,
            2.35011166e-01,
            3.59104797e-02,
            2.5,
            2.5,
            2.5,
        ]
    )
    xo = xo.unsqueeze(0)
    log.info(f"xo {xo.shape}")
    posteriors = [infer.build_posterior() for infer in inferences]
    ensemble_post = NeuralPosteriorEnsemble(posteriors=posteriors).set_default_x(xo)
    return ensemble_post, inferences


def load_pyloric_posterior_from_file(path, round_):
    log.info("loading posterior which had been trained in another folder!!!!")
    base = "/mnt/qb/macke/mdeistler57/tsnpe_collection/l5pc/results/p31_4/multiround"
    with open(join(base, path, f"inference_r{round_}.pkl"), "rb") as handle:
        inferences = dill.load(handle)
    xo = torch.as_tensor(
        [
            1.17085859e03,
            2.06036434e02,
            2.14307031e02,
            4.12842187e02,
            1.75970382e-01,
            1.83034085e-01,
            3.52597820e-01,
            4.11600328e-01,
            6.30544893e-01,
            4.81925781e02,
            2.56353125e02,
            2.75164844e02,
            4.20460938e01,
            2.35011166e-01,
            3.59104797e-02,
            2.5,
            2.5,
            2.5,
        ]
    )
    xo = xo.unsqueeze(0)
    log.info(f"xo {xo.shape}")
    posteriors = [infer.build_posterior() for infer in inferences]
    ensemble_post = NeuralPosteriorEnsemble(posteriors=posteriors).set_default_x(xo)
    return ensemble_post, inferences


if __name__ == "__main__":
    multiround()
