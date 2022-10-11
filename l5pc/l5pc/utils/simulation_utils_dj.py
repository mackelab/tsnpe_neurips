import os

from l5pc.model.table_definitions import L5PC_20D_theta, L5PC_20D_x
from omegaconf import DictConfig
import numpy as np
import pandas as pd


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
