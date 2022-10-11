import os
from itertools import chain

from copy import deepcopy
import numpy as np

from l5pc.model import L5PC_20D_theta, L5PC_20D_x
from l5pc.model.utils import return_names, return_x_names
from torch.distributions import MultivariateNormal
from torch import zeros, eye, tensor, float32, as_tensor, Tensor
import pandas as pd


def add_observation_noise(
    x: Tensor, id_: str, noise_multiplier: float = 1.0, std_type="data", subset=None
):
    print("Adding observation noise according to ", std_type)
    if std_type != "no_noise":
        if std_type == "sims":
            x_db = L5PC_20D_x()
            x_r1 = as_tensor(
                np.asarray(
                    (x_db & {"round": 1} & {"id": "l20_0"}).fetch(*return_x_names())
                ),
                dtype=float32,
            ).T
            x_r1_np = x_r1.numpy()
            stds = as_tensor(np.nanstd(x_r1_np, axis=0) * noise_multiplier)
        elif std_type == "data":
            stds = experimental_stds()
        else:
            raise NameError

        if subset is not None:
            stds = stds[subset]
        dim = len(stds)

        variances = stds**2
        noise = MultivariateNormal(zeros(dim), variances * eye(dim))
        noise_samples = noise.sample((x.shape[0],))
        assert noise_samples.shape == x.shape
        return x + noise_samples
    else:
        return x


def experimental_stds() -> Tensor:
    stds_step1 = [5.82, 4.58, 0.03, 4.97, 0.17, 0.0321, 0.0091, 33.48, 0.88, 7.32]
    stds_step2 = [5.57, 4.67, 0.027, 6.11, 0.28, 0.0368, 0.0056, 8.65, 0.56, 7.31]
    stds_step3 = [3.58, 3.92, 0.037, 6.93, 0.41, 0.0140, 0.0026, 0.83, 2.22, 1.0]
    stds_bap = [10.0, 9.33, 5.0, 0.5, 0.001]
    stds = [stds_step1, stds_step2, stds_step3, stds_bap]
    stds = as_tensor(np.asarray(list(chain.from_iterable(stds))), dtype=float32)
    return stds


def replace_nan(x: Tensor, stds_outside_data: float = 2.0, model="l5pc"):
    x_np = deepcopy(x.detach().numpy())

    if model == "l5pc":
        replacement_vals = get_replacement_vals(
            x.shape[0], x.shape[1], stds_outside_data
        )
    else:
        replacement_vals = get_replacement_vals_pyloric(
            x.shape[0], x.shape[1], stds_outside_data
        )
    nan_vals = np.isnan(x_np)
    x_np[nan_vals] = replacement_vals[nan_vals]
    return as_tensor(x_np), as_tensor(replacement_vals[0], dtype=float32)


def get_replacement_vals(
    batch: int, x_dim: int, stds_outside_data: float
) -> np.ndarray:
    x_db = L5PC_20D_x()
    x_r1 = as_tensor(
        np.asarray((x_db & {"round": 1} & {"id": "l20_0"}).fetch(*return_x_names())),
        dtype=float32,
    ).T
    x_r1_np = x_r1.numpy()

    x_min = np.nanmin(x_r1_np, axis=0)
    x_std = np.nanstd(x_r1_np, axis=0)
    x_std[7] = 3.0
    x_std[17] = 3.0
    x_std[27] = 3.0
    x_std[9] = 10.0
    x_std[19] = 10.0
    x_std[29] = 10.0
    replacement_vals = np.asarray([x_min - x_std * stds_outside_data])

    replacement_vals = np.tile(replacement_vals, (batch, 1))
    return replacement_vals[:, :x_dim]


def get_replacement_vals_pyloric(
    batch: int, x_dim: int, stds_outside_data: float
) -> np.ndarray:
    x_r0 = pd.read_pickle(
        "/mnt/qb/macke/mdeistler57/tsnpe_collection/l5pc/results/p31_4/prior_predictives_energy_paper/all_simulation_outputs.pkl"
    )
    x_r0 = as_tensor(np.asarray(x_r0), dtype=float32)
    x_r1_np = x_r0.numpy()
    x_min = np.nanmin(x_r1_np, axis=0)
    x_std = np.nanstd(x_r1_np, axis=0)
    replacement_vals = np.asarray([x_min - x_std * stds_outside_data])

    replacement_vals = np.tile(replacement_vals, (batch, 1))
    return replacement_vals[:, :x_dim]
