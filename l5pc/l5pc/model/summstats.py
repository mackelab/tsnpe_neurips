from typing import Optional
import efel
import json
import numpy as np
import os
import pandas as pd
from l5pc.model import l5pc_evaluator
from l5pc.model.utils import return_protocol_subset


def summstats_l5pc(
    trace: np.ndarray, protocol_subset: Optional[str] = None
) -> pd.DataFrame:

    """Calculate summary statistics
    Parameters
    ----------
    repetition_list : list of dictionaries, one per repetition
        data list, returned by `gen` method of Simulator instance
    Returns
    -------
    np.array, 2d with n_reps x n_summary
    """
    if protocol_subset is None:
        protocol_subset = ["Step1", "Step2", "Step3", "bAP"]

    stats = []
    for states in trace:

        dir_path = os.path.dirname(os.path.realpath(__file__))
        feature_configs = json.load(open(dir_path + "/config/features.json"))
        feature_configs = return_protocol_subset(feature_configs, protocol_subset)

        fitness_protocols = l5pc_evaluator.define_protocols(protocol_subset)

        features_all = []
        feature_names_for_pd = []
        for protocol_name, locations in sorted(feature_configs.items()):
            for location, features in sorted(locations.items()):
                for efel_feature_name, meanstd in sorted(features.items()):
                    feature_name = "%s.%s.%s" % (
                        protocol_name,
                        location,
                        efel_feature_name,
                    )
                    feature_name_dj = feature_name.lower().replace(".", "_")
                    feature_names_for_pd.append(feature_name_dj)
                    recording_names1 = str("%s.%s.v" % (protocol_name, location))
                    stimulus = fitness_protocols[protocol_name].stimuli[0]

                    stim_start = stimulus.step_delay

                    if location == "soma":
                        threshold = -10
                    elif "dend" in location:
                        threshold = -55

                    if protocol_name == "bAP":
                        stim_end = stimulus.total_duration
                    else:
                        stim_end = stimulus.step_delay + stimulus.step_duration

                    # prepare trace data
                    V1 = states[recording_names1].response
                    V1 = V1.to_numpy()
                    traces = {}
                    traces["T"] = V1[:, 0]
                    traces["V"] = V1[:, 1]
                    traces["stim_start"] = [stim_start]
                    traces["stim_end"] = [stim_end]
                    traces["Threshold"] = [threshold]

                    # calculate specified eFeatures
                    efel_results = efel.getFeatureValues(
                        [traces], [efel_feature_name], raise_warnings=False
                    )

                    if (
                        efel_results[0][efel_feature_name] is None
                        or not efel_results[0][efel_feature_name].size
                    ):
                        efel_results = np.nan
                    else:
                        # efel_results_mn = np.nanmean(efel_results[0][efel_feature_name][0])
                        # efel_results = efel_results[0][efel_feature_name][0]
                        efel_results = np.nanmean(efel_results[0][efel_feature_name])

                    features_all.append(efel_results)

        stats.append(features_all)

    return pd.DataFrame(np.asarray(stats), columns=feature_names_for_pd)
