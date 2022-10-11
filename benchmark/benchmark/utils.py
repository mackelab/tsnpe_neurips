import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import deneb as den
import numpy as np
import pandas as pd
import sbibm
import torch
from deneb.utils import rgb2hex
from omegaconf import OmegaConf
from sbibm.utils.io import get_float_from_csv
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


def get_colors(
    df: Optional[pd.DataFrame] = None,
    column: str = "algorithm",
    hex: bool = False,
    include_defaults: bool = False,
) -> Dict[str, str]:
    """Given a dataframe, builds a color dict with strings for algorithms
    Args:
        df: Dataframe
        column: Column containing algorithms
        hex: If True, will return hex values instead of RGB strings
        include_defaults: If True, will include default colors in returned dict

    Returns:
        Dictionary mapping algorithms to colors
    """
    COLORS_RGB = {
        "REJ": [74, 140, 251],
        "NLE": [96, 208, 152],
        "NPE": [204, 102, 204],
        "NRE": [255, 202, 88],
        "SMC": [33, 95, 198],
        "SNLE": [51, 153, 102],
        "SNPE": [153, 0, 153],
        "SNRE": [255, 166, 10],
        "PRIOR": [100, 100, 100],
        "POSTERIOR": [10, 10, 10],
        "TRUE": [249, 33, 0],
    }

    if include_defaults:
        COLORS = COLORS_RGB.copy()
    else:
        COLORS = {}

    if df is not None:
        for algorithm in df[column].unique():
            for color in COLORS_RGB.keys():
                if color.upper() in algorithm.upper():
                    COLORS[algorithm.strip()] = COLORS_RGB[color.upper()]

    COLORS_RGB_STR = {}
    COLORS_HEX_STR = {}
    for k, v in COLORS.items():
        COLORS_RGB_STR[k] = f"rgb({v[0]}, {v[1]}, {v[2]})"
        COLORS_HEX_STR[k] = rgb2hex(v[0], v[1], v[2])

    if hex:
        return COLORS_HEX_STR
    else:
        return COLORS_RGB_STR


def get_df(
    path: str,
    tasks: Optional[List[str]] = None,
    algorithms: Optional[List[str]] = None,
    observations: Optional[List[int]] = None,
    simulations: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Gets dataframe, and optionally subsets it
    Args:
        path: Path to dataframe
        tasks: Optional list of tasks to select
        algorithms: Optional list of algorithms to select
        observations: Optional list of observations to select
        simulations: Optional list of simulations to select
    Returns:
        Dataframe
    """
    df = pd.read_csv(path)

    # Subset dataframe
    if tasks is not None:
        df = df.query("task in (" + ", ".join([f"'{s}'" for s in tasks]) + ")")
    if algorithms is not None:
        df = df.query(
            "algorithm in (" + ", ".join([f"'{s}'" for s in algorithms]) + ")"
        )
    if observations is not None:
        df = df.query(
            "num_observation in (" + ", ".join([f"'{s}'" for s in observations]) + ")"
        )
    if simulations is not None:
        df = df.query(
            "num_simulations in (" + ", ".join([f"'{s}'" for s in simulations]) + ")"
        )

    return df


def get_float_from_csv(
    path: Union[str, Path],
    dtype: type = np.float32,
):
    """Get a single float from a csv file"""
    with open(path, "r") as fh:
        return np.loadtxt(fh).astype(dtype)


def compile_df(
    basepath: str,
) -> pd.DataFrame:
    """Compile dataframe for further analyses
    `basepath` is the path to a folder over which to recursively loop. All information
    is compiled into a big dataframe and returned for further analyses.
    Args:
        basepath: Base path to use
    Returns:
        Dataframe with results
    """
    df = []

    basepaths = [
        p.parent for p in Path(basepath).expanduser().rglob("posterior_samples.csv.bz2")
    ]

    for i, path_base in tqdm(enumerate(basepaths)):
        path_metrics = path_base / "metrics.csv"

        row = {}

        # Read hydra config
        path_cfg = path_metrics.parent / "run.yaml"
        if path_cfg.exists():
            cfg = OmegaConf.to_container(OmegaConf.load(str(path_cfg)))
        else:
            continue

        # Config file
        try:
            row["task"] = cfg["task"]["name"]
        except:
            continue
        row["num_simulations"] = cfg["task"]["num_simulations"]
        row["num_observation"] = cfg["task"]["num_observation"]
        row["algorithm"] = cfg["algorithm"]["name"]
        row["seed"] = cfg["seed"]

        # Metrics df
        if path_metrics.exists():
            metrics_df = pd.read_csv(path_metrics)
            for metric_name, metric_value in metrics_df.items():
                row[metric_name] = metric_value[0]
        else:
            continue

        # NLTP can be properly computed for NPE as part of the algorithm
        # SNPE's estimation of NLTP via rejection rates may introduce additional errors
        path_log_prob_true_parameters = (
            path_metrics.parent / "log_prob_true_parameters.csv"
        )
        row["NLTP"] = float("nan")
        if row["algorithm"][:3] == "NPE":
            if path_log_prob_true_parameters.exists():
                row["NLTP"] = -1.0 * get_float_from_csv(path_log_prob_true_parameters)

        # Runtime
        # While almost all runs were executed on AWS hardware under the same conditions,
        # this was not the case for 100% of the runs. To prevent uneven comparison,
        # the file `runtime.csv` was deleted for those runs where this was not the case.
        # If `runtime.csv` is absent from a run, RT will be set to NaN accordingly.
        path_runtime = path_metrics.parent / "runtime.csv"
        if not path_runtime.exists():
            row["RT"] = float("nan")
        else:
            row["RT"] = get_float_from_csv(path_runtime)

        # Runtime to minutes
        row["RT"] = row["RT"] / 60.0

        # Num simulations simulator
        path_num_simulations_simulator = (
            path_metrics.parent / "num_simulations_simulator.csv"
        )
        if path_num_simulations_simulator.exists():
            row["num_simulations_simulator"] = get_float_from_csv(
                path_num_simulations_simulator
            )

        # Path and folder
        row["path"] = str((path_base).absolute())
        row["folder"] = row["path"].split("/")[-1]

        # Exclude from df if there are no posterior samples
        if not os.path.isfile(f"{row['path']}/posterior_samples.csv.bz2"):
            continue

        # Demo run for the MMD figure, could be swap for another one
        if "8ad671f1-1e7b-4af8-b0e9-ec35a99d354f" in row["folder"]:
            continue

        df.append(row)

    df = pd.DataFrame(df)
    if len(df) > 0:
        df["num_observation"] = df["num_observation"].astype("category")

    return df


def apply_df(
    df,
    row_fn=None,
    *args,
    **kwargs,
):
    """Apply function for each row of dataframe
    Args:
        df: Dataframe
        row_fn: Function to apply to each row
    Returns:
        Dataframe with results
    """
    rows = []
    for r in tqdm(range(len(df))):
        row = row_fn(df.iloc[r].copy(), *args, **kwargs)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def clean(
    row,
    delete_unused_metrics=True,
    mmd_clip=True,
    mmd_clip_print=False,
):
    """Clean rows
    Args:
        row: Dataframe row
    Returns:
        row: Cleaned row or None if excluded
    """
    # Read hydra config
    path_cfg = f"{row['path']}/run.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(str(path_cfg)))

    # Renaming REJ-ABC because of sweeping different configs
    if "sbi.mcabc.run" in cfg["algorithm"]["run"]:
        row["algorithm"] = "REJ-ABC"
        algorithm_params = cfg["algorithm"]["params"]
        num_top_samples = int(algorithm_params["num_top_samples"])
        """
        row["algorithm"] = "REJ-ABC " + (
            str(num_top_samples / 1000)
            + "-"
            + str(num_top_samples / 10_000)
            + "-"
            + str(num_top_samples / 100_000)
        )
        """
        row["algorithm"] += f' {algorithm_params["num_top_samples"]}'
        if "learn_summary_statistics" in algorithm_params:
            if algorithm_params["learn_summary_statistics"]:
                row["algorithm"] += " SASS"
        if "sass" in algorithm_params:
            if algorithm_params["sass"]:
                row["algorithm"] += " SASS"

        if "linear_regression_adjustment" in algorithm_params:
            if algorithm_params["linear_regression_adjustment"]:
                row["algorithm"] += " LRA"

        if "lra" in algorithm_params:
            if algorithm_params["lra"]:
                row["algorithm"] += " LRA"

        if "kde_bandwidth" in algorithm_params:
            if algorithm_params["kde_bandwidth"] == "cv":
                row["algorithm"] += " KDE"

    # Renaming SMC-ABC because of sweeping different configs
    if "smc" in cfg["algorithm"]["run"]:
        algorithm_params = cfg["algorithm"]["params"]
        algo = ""

        if "pyabc" in cfg["algorithm"]["run"]:
            algo += "SMC-ABC (pyabc)"
        else:
            algo += "SMC-ABC (ours)"  # TODO: Add (ours) again

        if algorithm_params["population_size"] is not None:
            algo += f" {algorithm_params['population_size']}"

        if "kernel_variance_scale" in algorithm_params:
            algo += f' {algorithm_params["kernel_variance_scale"]}'

        if "epsilon_quantile" in algorithm_params:
            algo += f' {algorithm_params["epsilon_quantile"]}'
        else:
            algo += f' {algorithm_params["epsilon_decay"]}'

        if "learn_summary_statistics" in algorithm_params:
            if algorithm_params["learn_summary_statistics"]:
                algo += " SASS"
        if "sass" in algorithm_params:
            if algorithm_params["sass"]:
                algo += " SASS"

        if "linear_regression_adjustment" in algorithm_params:
            if algorithm_params["linear_regression_adjustment"]:
                algo += " LRA"
        if "lra" in algorithm_params:
            if algorithm_params["lra"]:
                algo += " LRA"

        if "kde_bandwidth" in algorithm_params:
            if algorithm_params["kde_bandwidth"] == "cv":
                algo += " KDE"

        row["algorithm"] = (
            algo.replace("0.2", ".2")
            .replace("0.5", ".5")
            .replace("0.8", ".8")
            .replace("2.0", "2")
        )

    # Renaming (S)NLE
    if "snle" in cfg["algorithm"]["run"]:
        algo = ""
        if cfg["algorithm"]["params"]["num_rounds"] > 1:
            algo += "S"
        algo += "NLE"
        if cfg["algorithm"]["params"]["neural_net"] == "maf":
            algo += "-MAF"
        elif cfg["algorithm"]["params"]["neural_net"] == "nsf":
            algo += "-NSF"
        row["algorithm"] = algo

    # Renaming (S)NPE
    if "snpe" in cfg["algorithm"]["run"]:
        algo = ""
        if cfg["algorithm"]["params"]["num_rounds"] > 1:
            algo += "S"
        algo += "NPE"
        if cfg["algorithm"]["params"]["neural_net"] == "maf":
            algo += "-MAF"
        elif cfg["algorithm"]["params"]["neural_net"] == "nsf":
            algo += "-NSF"
        row["algorithm"] = algo

    # Renaming (S)NRE
    if "snre" in cfg["algorithm"]["run"]:
        algo = ""
        if cfg["algorithm"]["params"]["num_rounds"] > 1:
            algo += "S"
        algo += "NRE"
        if cfg["algorithm"]["params"]["neural_net"] == "mlp":
            algo += "-MLP"
        elif cfg["algorithm"]["params"]["neural_net"] == "resnet":
            algo += "-RES"
        row["algorithm"] = algo

    if "prior" in cfg["algorithm"]["run"]:
        row["algorithm"] = "Prior"

    if "posterior" in cfg["algorithm"]["run"]:
        row["algorithm"] = "Posterior"

    # Convert num simulation to unicode
    row["num_simulations"] = den.latex2unicode(
        f"10^{len(str(row['num_simulations']))-1}"
    )

    # Number of simulations for SL runs (not be controlled by `num_simulations`)
    # Instead asssessed by: `df["num_simulations_simulator"].min()`
    if row["algorithm"] == "SL":
        row["num_simulations"] = den.latex2unicode(f">10^8")

    # For C2ST, use z-scored version (!)
    row["C2ST_NZ"] = row["C2ST"]
    row["C2ST_1K_NZ"] = row["C2ST_1K"]
    row["C2ST"] = row["C2ST_Z"]
    row["C2ST_1K"] = row["C2ST_1K_Z"]

    # Potentially delete unused metrics
    if delete_unused_metrics:
        del row["C2ST_NZ"]
        del row["C2ST_Z"]
        del row["C2ST_1K"]
        del row["C2ST_1K_NZ"]
        del row["C2ST_1K_Z"]
        del row["MMD_Z"]
        del row["MMD_1K"]
        del row["MMD_1K_Z"]
        del row["KSD_1K"]
        del row["MEDDIST_1K"]
        del row["num_simulations_simulator"]
        del row["seed"]
        del row["path"]

    # Cap MMD
    if row["MMD"] > 10_000_000:
        if mmd_clip_print:
            print("Warning: Encountered extremely large MMD for:")
            print(row)
        row["MMD"] = float("nan")

    return row
