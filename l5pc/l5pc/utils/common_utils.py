from os.path import join
import os

import ruamel.yaml as yaml
from typing import List
import torch
from torch import Tensor, as_tensor, float32
import dill
import pickle
import pandas as pd

from l5pc.model import Priorl5pc
from l5pc.model import L5PC_20D_x
from l5pc.model.utils import return_xo
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble


def load_prior(id="l20_0", as_torch_dist: bool = False):
    prior = Priorl5pc(
        bounds=[[], []],
        dim=20,
    )
    if as_torch_dist:
        prior = prior.prior_torch
    return prior


def extract_bounds(prior) -> Tensor:
    if hasattr(prior, "prior_torch"):
        prior = prior.prior_torch
    lower_bound = prior.support.base_constraint.lower_bound
    upper_bound = prior.support.base_constraint.upper_bound
    return torch.stack([lower_bound, upper_bound])


def load_posterior(id: str, path: str):
    base_path = "/mnt/qb/macke/mdeistler57/tsnpe_collection/l5pc/"
    inference_path = join(base_path, f"results/{id}/inference/{path}")
    with open(join(inference_path, "inference.pkl"), "rb") as handle:
        inferences = dill.load(handle)
    with open(join(inference_path, "used_features.pkl"), "rb") as handle:
        used_features = dill.load(handle)
    with open(join(inference_path, "xo.pkl"), "rb") as handle:
        xo = pickle.load(handle)
    with open(join(inference_path, "round.pkl"), "rb") as handle:
        round_ = pickle.load(handle)
    xo = as_tensor(xo[:, used_features], dtype=float32)
    posteriors = [infer.build_posterior() for infer in inferences]
    posterior = NeuralPosteriorEnsemble(posteriors=posteriors).set_default_x(xo)
    return inferences, posterior, used_features, round_
