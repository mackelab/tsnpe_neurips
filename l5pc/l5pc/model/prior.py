from typing import Optional

import torch
import numpy as np
from sbi.utils import BoxUniform
from typing import List, Union, Tuple
import pandas as pd
from l5pc.model.utils import return_names
from l5pc.model.l5pc_model import define_parameters


class Priorl5pc:
    def __init__(
        self, bounds: List, dim: int, return_dataframe: Optional[bool] = None
    ) -> None:
        self.names = return_names()[:dim]

        all_names_for_bounds = return_names(dj=False)
        param_names_bounds = [
            (str(param.name), param.bounds)
            for param in define_parameters()
            if not param.frozen
        ]
        param_names_bounds.sort(key=lambda x: all_names_for_bounds.index(x[0]))
        param_bounds = [
            param_names_bounds[i][1] for i in range(len(param_names_bounds))
        ]
        param_bounds = np.array(param_bounds)
        param_bounds = param_bounds[:20, :]

        prior_min = param_bounds[:dim, 0]
        prior_max = param_bounds[:dim, 1]

        # prior_min_log = np.log(prior_min)
        # prior_max_log = np.log(prior_max)

        self.prior_torch = BoxUniform(prior_min, prior_max)
        self.rd = return_dataframe

    def sample(self, sample_shape: Tuple, return_dataframe: bool = True):
        samples = self.prior_torch.sample(sample_shape)
        rd = return_dataframe if self.rd is None else self.rd

        if rd:
            samples = pd.DataFrame(samples.numpy(), columns=self.names)
        return samples

    def log_prob(self, theta: Union[pd.DataFrame, torch.Tensor]) -> torch.Tensor:
        if isinstance(theta, pd.DataFrame):
            theta = torch.as_tensor(theta.to_numpy(), dtype=torch.float32)
        return self.prior_torch.log_prob(theta)
