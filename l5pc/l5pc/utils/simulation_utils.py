import os
from functools import partial
from typing import Callable
from l5pc.model import (
    Priorl5pc,
    simulate_l5pc,
    setup_l5pc,
    summstats_l5pc,
)
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from pyloric import simulate, create_prior, summary_stats


def simulate_and_summstats(
    theta: pd.DataFrame, _simulator: Callable, _summstats: Callable
) -> pd.DataFrame:
    trace = _simulator(theta)
    x = _summstats(trace)
    return x


def assemble_prior(cfg: DictConfig):
    if cfg.model.name.startswith("l5pc"):
        prior = Priorl5pc(bounds=cfg.model.prior, dim=cfg.model.num_params)
    elif cfg.model.name.startswith("pyloric"):
        prior = create_prior()
    else:
        raise NameError
    return prior


def assemble_simulator(cfg: DictConfig):
    if cfg.name.startswith("l5pc"):
        neuron_simulator = partial(
            simulate_l5pc,
            protocol_subset=cfg.protocols,
            nois_fact_obs=cfg.noise,
        )
        summstats = partial(summstats_l5pc, protocol_subset=cfg.protocols)
        setup_l5pc()
        sim_and_stats = partial(
            simulate_and_summstats,
            _simulator=neuron_simulator,
            _summstats=summstats,
        )
    elif cfg.name.startswith("pyloric"):
        sim_and_stats = partial(
            simulate_and_summstats,
            _simulator=simulate,
            _summstats=summary_stats,
        )
    else:
        raise NameError

    return sim_and_stats


def sim_and_stat_pyloric(theta: pd.DataFrame) -> pd.DataFrame:
    trace = simulate(theta)
    x = summary_stats(trace, stats_customization={"plateau_durations": True})
    return x


def assemble_pyloric():
    return sim_and_stat_pyloric
