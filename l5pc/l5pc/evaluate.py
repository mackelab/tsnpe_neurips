import logging

import torch
from torch import as_tensor, tensor, Tensor, float32, float64, ones, zeros, eye
from omegaconf import DictConfig
import hydra
import numpy as np
import pandas as pd
from sbi.analysis import pairplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

# These files live in utils because I otherwise had problems with SLURM and
# multiprocessing. See this error: https://www.pythonanywhere.com/forums/topic/27818/
from l5pc.utils.evaluation_utils import (
    predictive_traces,
    plot_traces,
    plot_summstats,
    compare_gt_log_probs,
    gt_log_prob,
    coverage,
    plot_coverage,
    show_traces_pyloric,
)
from multiprocessing import Pool

from l5pc.utils.model_utils import (
    replace_nan,
    add_observation_noise,
)
from l5pc.utils.common_utils import (
    load_prior,
    extract_bounds,
    load_posterior,
)
from l5pc.model.utils import return_gt, return_x_names, return_names
from l5pc.model import L5PC_20D_theta, L5PC_20D_x, summstats_l5pc
from sbi.utils.support_posterior import PosteriorSupport
from pyloric import create_prior, simulate, summary_stats

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="evaluation")
def evaluate(cfg: DictConfig) -> None:

    if cfg.model.name.startswith("l5pc"):
        l5pc_evaluation(cfg)
    elif cfg.model.name.startswith("pyloric"):
        pyloric_evaluation(cfg)
    else:
        raise ValueError


def pyloric_evaluation(cfg):
    _ = torch.manual_seed(cfg.seed)
    inference, posterior, used_features, round_ = load_posterior(cfg.id, cfg.posterior)
    log.info(f"Posterior after round: {round_}")
    prior = create_prior()
    lower = prior.numerical_prior.support.base_constraint.lower_bound
    upper = prior.numerical_prior.support.base_constraint.upper_bound
    prior_bounds = torch.stack([lower, upper]).T.numpy()
    theta = posterior.sample((cfg.num_predictives,))
    prior_samples = prior.sample((1,))
    theta = pd.DataFrame(theta.numpy(), columns=prior_samples.columns)
    num_splits = cfg.num_predictives
    batches = np.array_split(theta, num_splits)
    batches = [b.iloc[0] for b in batches]

    with Pool(cfg.cores) as pool:
        x_list = pool.map(simulate, batches)

    stats = pd.concat([summary_stats(xx) for xx in x_list])
    stats = stats.to_numpy()
    valid = np.invert(np.any(np.isnan(stats), axis=1))
    log.info(f"Fraction of valid sims: {np.sum(valid) / cfg.num_predictives}")
    with mpl.rc_context(
        fname="/mnt/qb/macke/mdeistler57/tsnpe_collection/l5pc/.matplotlibrc"
    ):
        show_traces_pyloric(x_list)
        plt.savefig("traces.png")

        posterior_samples = posterior.sample((1000,), show_progress_bars=False)
        _ = pairplot(
            posterior_samples,
            limits=prior_bounds,
            upper=["kde"],
            ticks=prior_bounds,
            figsize=(10, 10),
        )
        plt.savefig("pairplot.png")


def l5pc_evaluation(cfg):
    _ = torch.manual_seed(cfg.seed)
    inference, posterior, used_features, round_ = load_posterior(cfg.id, cfg.posterior)

    # round_ = inference[0]._round + 1
    log.info(f"Posterior after round: {round_}")
    prior = load_prior()
    prior_bounds = extract_bounds(prior).T.numpy()

    posterior._prior = prior.prior_torch

    posterior_support = PosteriorSupport(
        prior=prior.prior_torch,
        posterior=posterior,
        num_samples_to_estimate_support=cfg.num_samples_to_estimate_support,
        allowed_false_negatives=cfg.allowed_false_negatives,
        use_constrained_prior=cfg.use_constrained_prior,
        constrained_prior_quanitle=cfg.constrained_prior_quanitle,
    )

    log.info(f"Starting to simulate {cfg.num_predictives} predictive traces")
    traces, theta = predictive_traces(
        posterior=posterior, num_samples=cfg.num_predictives, num_cores=cfg.cores
    )
    log.info(f"Finished simulating predictives")
    with open("traces.pkl", "wb") as handle:
        pickle.dump(traces, handle)
    theta.to_pickle("parameters.pkl")

    posterior_stats = summstats_l5pc(traces)
    valid = np.invert(np.any(np.isnan(posterior_stats.to_numpy()), axis=1))
    log.info(f"Fraction of all valid sims: {np.sum(valid) / cfg.num_predictives}")
    posterior_stats.to_pickle("stats.pkl")

    x_db = L5PC_20D_x()
    theta_db = L5PC_20D_theta()

    data_id = "l20_0" if round_ == 1 else cfg.id
    theta = as_tensor(
        np.asarray(
            (theta_db & {"round": round_} & {"id": data_id}).fetch(*return_names())
        ),
        dtype=float32,
    ).T
    x = as_tensor(
        np.asarray(
            (x_db & {"round": round_} & {"id": data_id}).fetch(*return_x_names())
        ),
        dtype=float32,
    ).T

    x_pd = pd.DataFrame(x[-1000:].numpy(), columns=return_x_names())

    x = x[-1000:]
    theta = theta[-1000:]
    x, _ = replace_nan(x)
    x = add_observation_noise(
        x=x,
        id_=cfg.id,
        noise_multiplier=0.0,
        std_type="data",
        subset=None,
    )
    alpha, cov = coverage(posterior, theta, x, used_features)

    with mpl.rc_context(
        fname="/mnt/qb/macke/mdeistler57/tsnpe_collection/l5pc/.matplotlibrc"
    ):
        plot_traces(traces)
        plt.savefig("traces.png", dpi=200, bbox_inches="tight")
        plt.show()

        plot_summstats(posterior_stats, x_pd)
        plt.savefig("stats.png", dpi=200, bbox_inches="tight")
        plt.show()
        plot_summstats(posterior_stats, x_pd, used_features=used_features)
        plt.savefig("stats_fitted.png", dpi=200, bbox_inches="tight")
        plt.show()

        post_lp, prior_lp = compare_gt_log_probs(posterior)
        log.info(f"Posterior log-prob: {post_lp}")
        log.info(f"Prior log-prob:     {prior_lp}")

        plot_coverage(alpha, cov)
        plt.savefig("coverage.png", dpi=200, bbox_inches="tight")
        plt.show()

        posterior_samples = posterior.sample((1000,), show_progress_bars=False)
        _ = pairplot(
            posterior_samples,
            limits=prior_bounds,
            upper=["kde"],
            ticks=prior_bounds,
            figsize=(10, 10),
            labels=return_names(),
            points=return_gt(as_pd=False),
            points_colors="r",
        )
        plt.savefig("pairplot.png", dpi=200, bbox_inches="tight")
        plt.show()

        gt_log_prob(posterior)
        plt.savefig("log_probs.png", dpi=200, bbox_inches="tight")
        plt.show()

        _, acceptance_rate = posterior_support.sample(
            (100,), return_acceptance_rate=True
        )
        log.info(f"log10(acceptance rate of support): {acceptance_rate}%")


if __name__ == "__main__":
    evaluate()
