from multiprocessing import Pool
from os.path import join
from copy import deepcopy
from tqdm.auto import tqdm
from torch import tensor, as_tensor, Tensor, eye, zeros, ones, float32
import matplotlib.pyplot as plt
import matplotlib as mpl
from l5pc.model.utils import (
    return_gt,
    return_xo,
    return_x_names,
    return_names,
)
import torch
import dill
import pickle
from l5pc.model import (
    simulate_l5pc,
    setup_l5pc,
    summstats_l5pc,
)
import pandas as pd
import numpy as np
from l5pc.utils.common_utils import load_prior
from itertools import chain


def plot_single_trace_step1(traces, figsize=None):
    if figsize is None:
        figsize = (5, 2)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    traces_protocol = traces["Step1.soma.v"].response["voltage"].to_numpy()
    time_protocol = traces["Step1.soma.v"].response["time"].to_numpy()
    _ = ax.plot(
        np.asarray(time_protocol).T,
        np.asarray(traces_protocol).T,
        c="k",
        alpha=0.8,
    )
    return fig, ax


def plot_traces(traces, xo_trace, figsize=None, protocol=None, num_traces=10):
    if protocol is None:
        protocol = [
            "bAP.soma.v",
            "bAP.dend1.v",
            "bAP.dend2.v",
            "Step3.soma.v",
            "Step2.soma.v",
            "Step1.soma.v",
        ]
    if figsize is None:
        figsize = (12, 0.7 * num_traces)

    traces = traces[:num_traces]

    fig, ax = plt.subplots(num_traces + 1, len(protocol), figsize=figsize)
    # xo_trace = return_xo(summstats=False)[0]
    for i, p in enumerate(protocol):
        _ = ax[0, i].plot(
            xo_trace[p].response["time"].to_numpy(),
            xo_trace[p].response["voltage"].to_numpy(),
            c="#fc4e2a",
            alpha=0.9,
        )
        ax[0, i].set_title(p)
    for trace_ind, t in enumerate(traces):
        for i, p in enumerate(protocol):
            traces_protocol = t[p].response["voltage"].to_numpy()
            time_protocol = t[p].response["time"].to_numpy()
            _ = ax[trace_ind + 1, i].plot(
                np.asarray(time_protocol).T,
                np.asarray(traces_protocol).T,
                c="k",
                alpha=0.8,
            )
    for i in range(num_traces):
        for j in range(len(protocol)):
            ax[i, j].spines["bottom"].set_visible(False)
            ax[i + 1, j].spines["left"].set_visible(False)
            ax[i, j].set_xticks([])
            ax[i + 1, j].set_yticks([])
            xo = xo_trace[protocol[j]].response["voltage"].to_numpy()
            ax[i + 1, j].set_ylim([np.min(xo) - 15.0, np.max(xo) + 15.0])
            ax[0, j].set_ylim([np.min(xo) - 15.0, np.max(xo) + 15.0])
    ax[10, 2].set_xlabel(r"$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$Time [ms]")


def plot_summstats(posterior_stats, prior_stats, used_features=None):
    xo = return_xo(summstats=True, as_pd=False)[0]
    feature_names = return_x_names()
    posterior_stats = posterior_stats.to_numpy()
    prior_stats = prior_stats.to_numpy()
    # if used_features is not None:
    if False:
        posterior_stats = posterior_stats[:, used_features]
        prior_stats = prior_stats[:, used_features]
        xo = xo[used_features]
    dim_x = posterior_stats.shape[1]

    fig, ax = plt.subplots(dim_x, 2, figsize=(6, 1 * dim_x))

    for x_num in range(dim_x):
        valid_frac_prior, prior = extract_valid_fraction(prior_stats[:, x_num])
        valid_frac_posterior, posterior = extract_valid_fraction(
            posterior_stats[:, x_num]
        )
        if prior.tolist():
            prior_min = np.min(prior)
            prior_max = np.max(prior)
        else:
            prior_min = np.inf
            prior_max = -np.inf
        if posterior.tolist():
            posterior_min = np.min(posterior)
            posterior_max = np.max(posterior)
        else:
            posterior_min = np.inf
            posterior_max = -np.inf
        limit_min = np.min([prior_min, posterior_min, xo[x_num]])
        limit_max = np.max([prior_max, posterior_max, xo[x_num]])

        if prior.tolist():
            _ = ax[x_num, 0].hist(
                prior, bins=30, density=True, range=[limit_min, limit_max]
            )
        ax[x_num, 0].set_title(f"{valid_frac_prior}")
        if posterior.tolist():
            _ = ax[x_num, 1].hist(
                posterior, bins=30, density=True, range=[limit_min, limit_max]
            )
        ax[x_num, 1].set_title(f"{valid_frac_posterior}")

        ax[x_num, 0].axvline(xo[x_num], c="r")
        ax[x_num, 1].axvline(xo[x_num], c="r")

        ax[x_num, 0].set_xlim([limit_min, limit_max])
        ax[x_num, 1].set_xlim([limit_min, limit_max])

        ax[x_num, 0].set_xticks([limit_min, limit_max])
        ax[x_num, 1].set_xticks([limit_min, limit_max])

        name = feature_names[x_num][:20]
        ax[x_num, 0].set_ylabel(f"{name[:10]}\n{name[10:]}")
    plt.subplots_adjust(hspace=0.5)


def gt_log_prob(posterior):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.hist(
        posterior.log_prob(
            posterior.sample((10_000,), show_progress_bars=False)
        ).numpy(),
        bins=50,
    )
    ax.axvline(
        posterior.log_prob(as_tensor(return_gt(as_pd=False)[0], dtype=float32)).numpy(),
        c="r",
    )
    ax.set_xlabel("Posterior log-prob")


def extract_valid_fraction(x):
    """
    x should be a batch of single summstats and have shape [N].

    Returns the fraction of valids and returns all valids.
    """
    not_nan = np.invert(np.isnan(x))
    return np.sum(not_nan) / len(x), x[not_nan]


def predictive_traces(
    posterior=None, prior=None, num_samples: int = 1, num_cores: int = 1
):
    if posterior is not None:
        samples = posterior.sample((num_samples,), show_progress_bars=False)
    elif prior is not None:
        samples = prior.sample((num_samples,))
    else:
        raise NameError
    samples_pd = pd.DataFrame(samples.numpy(), columns=return_names())

    num_splits = max(1, num_samples // num_cores)
    batches = np.array_split(samples_pd, num_splits)

    setup_l5pc()
    np.savetxt("test_file.txt", [0, 1])
    with Pool(num_cores) as pool:
        x_list = pool.map(simulate_l5pc, batches)

    traces = list(chain.from_iterable(x_list))

    return traces, samples_pd


def compare_gt_log_probs(posterior, id="l20_0"):
    prior = load_prior(id)
    gt = torch.as_tensor(return_gt(as_pd=False), dtype=torch.float32)
    post_lp = posterior.log_prob(gt)
    prior_lp = prior.log_prob(gt)
    return post_lp, prior_lp


def coverage(
    posterior,
    theta,
    x,
    used_features,
    num_x: int = 100,
    alpha=torch.linspace(0, 1, 50),
    num_monte_carlo: int = 1_000,
):
    posterior = deepcopy(posterior)
    gt_is_covered = zeros(alpha.shape)
    theta = theta.numpy()
    x = x.numpy()
    x = x[:, used_features]
    nonan = np.invert(np.any(np.isnan(x), axis=1))
    x = x[nonan]
    theta = theta[nonan]
    x = x[:num_x]
    theta = theta[:num_x]
    with tqdm(total=len(theta)) as pbar:
        for params, summstats in tqdm(zip(theta, x)):
            xo = as_tensor(np.asarray([summstats]), dtype=float32)
            posterior.set_default_x(xo)
            lprobs = posterior.log_prob(
                posterior.sample((num_monte_carlo,), show_progress_bars=False)
            )
            gt_log_prob = posterior.log_prob(
                as_tensor(np.asarray([params]), dtype=float32)
            )
            rank_of_gt = compute_rank(gt_log_prob, lprobs)
            norm_rank = rank_of_gt / lprobs.shape[0]
            covered_in_alpha_quantile = norm_rank > alpha
            gt_is_covered += covered_in_alpha_quantile.float()
            pbar.update(1)
    gt_is_covered /= x.shape[0]
    return alpha, torch.flip(gt_is_covered, dims=[0])


def compute_rank(val: Tensor, vec: Tensor):
    c = torch.cat([vec, val])
    s = torch.argsort(c)
    ind = torch.where(s == len(c) - 1)
    return ind[0]  # .where returns a tuple


def plot_coverage(alpha, gt_is_covered):
    fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5))
    ax.plot([0, 1.0], [0, 1.0], c="grey")
    ax.plot(alpha, gt_is_covered)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Confidence level")
    ax.set_ylabel("Empirical coverage")


def show_traces_pyloric(simulation_output):
    fig, axes = plt.subplots(10, 3, figsize=(7.5, 5.5))
    for trace_ind in range(10):
        current_sim = simulation_output[trace_ind]
        traces = current_sim["voltage"]
        t = np.arange(0, current_sim["t_max"], current_sim["dt"])
        neuron_labels = ["AB/PD", "LP", "PY"]
        current_ax = axes[trace_ind]
        for i, ax in enumerate(current_ax):
            ax.plot(t, traces[i])
            ax.set_ylim([-90, 60])
            if i < 2 or trace_ind < 9:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_xlabel("Time (ms)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylabel(neuron_labels[i])
            if i == 0 and trace_ind == 0:
                ax.set_title("Votage (mV)")
    return fig, axes
