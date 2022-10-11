from multiprocessing import Value
from typing import Optional
import bluepyopt.ephys as ephys
import json
import numpy as np
import os
import pandas as pd
from l5pc.model import l5pc_model
from l5pc.model import l5pc_evaluator
from l5pc.model.utils import (
    return_gt,
    return_names,
    return_protocol_subset,
    rename_theta_dj_to_bpo,
)
from neuron import h


def setup_l5pc(load_libraries: bool = True):
    # path = "/mnt/qb/macke/mdeistler57/multicompartment/multicompartment/models/l5pc/"
    path = "/mnt/qb/macke/mdeistler57/tsnpe_collection/l5pc/l5pc/model/"
    # See https://www.neuron.yale.edu/phpBB/viewtopic.php?t=4283
    # Without this line, we can not run this file from a different directory.
    if load_libraries:
        try:
            print("Loading neuron libraries")
            h.nrn_load_dll(path + "x86_64/.libs/libnrnmech.so")
            print("Successfully loaded libraries!")
        except RuntimeError:
            print("Failed to load neuron libraries")
            pass


def simulate_l5pc(
    theta: pd.DataFrame,
    seed=None,
    protocol_subset: Optional[str] = None,
    nois_fact_obs: float = 0.0,
):
    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()

    if protocol_subset is None:
        protocol_subset = ["Step1", "Step2", "Step3", "bAP"]

    # morphology
    dir_path = os.path.dirname(os.path.realpath(__file__))
    morphology = ephys.morphologies.NrnFileMorphology(
        dir_path + "/morphology/C060114A7.asc", do_replace_axon=True
    )

    parameters = l5pc_model.define_parameters()
    mechanisms = l5pc_model.define_mechanisms()

    # cell model
    l5pc_cell = ephys.models.CellModel(
        "l5pc", morph=morphology, mechs=mechanisms, params=parameters
    )
    param_names = [
        param.name for param in l5pc_cell.params.values() if not param.frozen
    ]

    # protocols
    fitness_protocols = l5pc_evaluator.define_protocols(protocol_subset)

    # eFeatures
    feature_configs = json.load(open(dir_path + "/config/features.json"))
    feature_configs = return_protocol_subset(feature_configs, protocol_subset)
    fitness_calculator = l5pc_evaluator.define_fitness_calculator(
        fitness_protocols, protocol_subset=protocol_subset
    )

    # simulator
    sim = ephys.simulators.NrnSimulator()

    # evaluator
    # `isolate_protocols=False` turns of multiprocessing of the protocols. If this is
    # set to `True`, we use mp within mp and thus get an error.
    evaluator = ephys.evaluators.CellEvaluator(
        cell_model=l5pc_cell,
        param_names=param_names,
        fitness_protocols=fitness_protocols,
        fitness_calculator=fitness_calculator,
        sim=sim,
        isolate_protocols=False,
    )

    gt = return_gt()
    theta_full = pd.concat([gt] * theta.shape[0])

    # relevant_names = np.asarray(return_names())[[5, 11, 13, 18]]
    # theta[relevant_names] += [0.0005, 20, 0.0005, 20]

    # theta = theta.reset_index()
    for specified_parameters in theta.keys():
        theta_full[specified_parameters] = theta[specified_parameters].to_numpy()

    # Use the param names that are internal to the L5PC model (see
    # `l5pc/config/params.json`)
    theta_full = rename_theta_dj_to_bpo(theta_full)

    # theta = np.exp(theta)

    trace = []

    for _, t in theta_full.iterrows():
        t_dict = t.to_dict()

        # simulation
        V = evaluator.run_protocols(
            protocols=fitness_protocols.values(), param_values=t_dict
        )

        for protocol_name, locations in sorted(feature_configs.items()):
            for location, features in sorted(locations.items()):
                recording_names1 = str("%s.%s.v" % (protocol_name, location))

                len_tr = len(V[recording_names1].response["voltage"])
                noise_obs = nois_fact_obs * rng.randn(len_tr)

                V[recording_names1].response["voltage"] = (
                    V[recording_names1].response["voltage"] + noise_obs
                )

        trace.append(V)

    # return np.array(V).reshape(-1,1) + nois_fact_obs*self.rng.randn(V.shape[0],1)
    return trace


if __name__ == "__main__":
    gt = return_gt()
    names = return_names()
    theta = pd.DataFrame([gt], columns=names)

    _ = simulate_l5pc(theta, load_libraries=False)
