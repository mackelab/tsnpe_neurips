from distutils.log import warn
import os
import time
import ruamel.yaml as yaml
from typing import List
from functools import partial
from multiprocessing import Pool
from typing import Callable
from l5pc.model import (
    Priorl5pc,
    simulate_l5pc,
    setup_l5pc,
    summstats_l5pc,
)
from l5pc.model import L5PC_20D_theta, L5PC_20D_x
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from l5pc.model.utils import return_names
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


def assemble_db(cfg: DictConfig):
    if cfg.model.name.startswith("l5pc"):
        x_db = L5PC_20D_x()
        theta_db = L5PC_20D_theta()
    else:
        x_db = None
        theta_db = None
    return theta_db, x_db


def write_to_dj(
    theta: pd.DataFrame,
    x: pd.DataFrame,
    theta_db,
    x_db,
    round_: int,
    id_: str,
    increase_by_1000: bool = False,
) -> None:
    """
    Writes theta and x to the database server.

    - reads from the database to get the highest index that currently lies in the db
    - creates an index column
    - adds the previously highest value to the index column
    - creates a `round` column
    - uploads to dj
    """

    previous_indizes = theta_db.fetch("ind")
    if previous_indizes.size > 0:
        theta_starting_ind = np.max(previous_indizes) + 1
    else:
        theta_starting_ind = 0
    if increase_by_1000:
        theta_starting_ind += 1000

    x["ind"] = x.index
    x["ind"] += theta_starting_ind
    theta["ind"] = theta.index
    theta["ind"] += theta_starting_ind

    theta["round"] = round_
    x["round"] = round_

    theta["id"] = id_
    x["id"] = id_

    working_dir = os.getcwd()
    ind_of_last_slash = working_dir[::-1].index("/")
    ind_of_second_last_slash = working_dir[::-1][ind_of_last_slash + 1 :].index("/")
    current_folder = working_dir[-ind_of_last_slash - ind_of_second_last_slash - 1 :]
    theta["path"] = current_folder
    x["path"] = current_folder

    x_db.insert(x)
    theta_db.insert(theta)
