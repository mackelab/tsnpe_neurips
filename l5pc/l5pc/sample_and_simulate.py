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

# These files live in utils because I otherwise had problems with SLURM and
# multiprocessing. See this error: https://www.pythonanywhere.com/forums/topic/27818/
from l5pc.utils.simulation_utils import (
    assemble_prior,
    assemble_simulator,
    assemble_db,
    write_to_dj,
)
from l5pc.utils.common_utils import load_posterior
from l5pc.model.utils import return_gt, return_names, return_xo
from sbi.utils import BoxUniform
from sbi.utils.support_posterior import PosteriorSupport

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="sim_model")
def sample_and_simulate(cfg: DictConfig) -> None:
    print(cfg)
    start_time = time.time()
    log.info(f"Starting run! {time.time() - start_time}")

    assert cfg.id is not None, "Specify an ID. Format: [model][dim]_[run], e.g. j2_3"

    prior = assemble_prior(cfg)
    sim_and_stats = assemble_simulator(cfg.model)
    theta_db, x_db = assemble_db(cfg)

    log.info(f"Assembled! {time.time() - start_time}")

    seed = int((time.time() % 1) * 1e7) if cfg.seed_prior is None else cfg.seed_prior
    _ = torch.manual_seed(seed)
    np.savetxt(f"seed.txt", [seed], fmt="%d")

    remaining_sims = cfg.sims

    log.info(f"Starting loop! {time.time() - start_time}")

    if cfg.proposal is None:
        proposal = prior
        round_ = 1
    else:
        inference, posterior, _, round_train = load_posterior(cfg.id, cfg.proposal)
        round_ = round_train + 1
        log.info(f"Loaded posterior, round", round_)
        if cfg.thr_proposal:
            _ = torch.manual_seed(0)  # Set seed=0 only for building the proposal.
            proposal = PosteriorSupport(
                prior=prior.prior_torch,
                posterior=posterior,
                num_samples_to_estimate_support=cfg.num_samples_to_estimate_support,
                allowed_false_negatives=cfg.allowed_false_negatives,
                use_constrained_prior=cfg.use_constrained_prior,
                constrained_prior_quanitle=cfg.constrained_prior_quanitle,
                sampling_method=cfg.sampling_method,
            )
            log.info("Built support")
            _ = torch.manual_seed(seed)
        else:
            proposal = posterior

    counter = 0
    while remaining_sims > 0:
        num_to_simulate = min(remaining_sims, cfg.sims_until_save)
        log.info(f"num_to_simulate", num_to_simulate)
        # samples_list = Parallel(n_jobs=10)(
        #     delayed(sample_n)(proposal, int(num_to_simulate / 10), seed + s)
        #     for s in range(10)
        # )
        # theta = torch.cat(samples_list)
        theta = proposal.sample((num_to_simulate,))

        log.info(f"Sampled proposal", theta.shape)
        if isinstance(theta, torch.Tensor):
            if cfg.model.name.startswith("l5pc"):
                theta = pd.DataFrame(theta.numpy(), columns=return_names())
            else:
                sss = prior.sample((1,))
                pyloric_names = sss.columns
                theta = pd.DataFrame(theta.numpy(), columns=pyloric_names)

        if cfg.model.name.startswith("l5pc"):
            gt = return_gt()
            theta_full = pd.concat([gt] * theta.shape[0], ignore_index=True)
            for specified_parameters in theta.keys():
                theta_full[specified_parameters] = theta[
                    specified_parameters
                ].to_numpy()
        else:
            theta_full = theta

        log.info(f"Time to obtain theta: {time.time() - start_time}")

        # Each worker should process a batch of simulations to reduce the overhead of
        # loading neuron.
        num_splits = max(1, num_to_simulate // cfg.sims_per_worker)
        batches = np.array_split(theta_full, num_splits)
        if cfg.model.name.startswith("pyloric"):
            batches = [b.iloc[0] for b in batches]

        log.info(f"Time to obtain batches: {time.time() - start_time}")

        with Pool(cfg.cores) as pool:
            x_list = pool.map(sim_and_stats, batches)

        log.info(f"Sims done {time.time() - start_time}")
        x = pd.concat(x_list, ignore_index=True)
        log.debug(f"Sims concatenated {time.time() - start_time}")

        if cfg.save_sims:
            if cfg.model.name.startswith("l5pc"):
                write_to_dj(
                    theta_full,
                    x,
                    theta_db,
                    x_db,
                    round_,
                    cfg.id,
                    increase_by_1000=cfg.increase_dj_ind_by_1000,
                )
            elif cfg.model.name.startswith("pyloric"):
                log.info(f"Saving {len(x)} simulations")
                theta_full.to_pickle(f"sims_theta_{counter}.pkl")
                x.to_pickle(f"sims_x_{counter}.pkl")
                counter += 1
            else:
                raise ValueError

        log.info(f"Written to dj {time.time() - start_time}")

        remaining_sims -= num_to_simulate


def sample_n(proposal, num_samples, seed):
    _ = torch.manual_seed(seed)
    theta = proposal.sample((num_samples,))
    return theta


if __name__ == "__main__":
    sample_and_simulate()
