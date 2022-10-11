import os
import numpy as np
import pandas as pd


def return_names(dj: bool = True):
    """
    Args:
        dj: If True, all names are lowercase and without dots such that they are
            accepted by datajoint as attributes.
    """
    labels_params = [
        "gNaTs2_tbar_NaTs2_t.apical",
        "gSKv3_1bar_SKv3_1.apical",
        "gImbar_Im.apical",
        "gNaTa_tbar_NaTa_t.axonal",
        "gK_Tstbar_K_Tst.axonal",
        "gamma_CaDynamics_E2.axonal",
        "gNap_Et2bar_Nap_Et2.axonal",
        "gSK_E2bar_SK_E2.axonal",
        "gCa_HVAbar_Ca_HVA.axonal",
        "gK_Pstbar_K_Pst.axonal",
        "gSKv3_1bar_SKv3_1.axonal",
        "decay_CaDynamics_E2.axonal",
        "gCa_LVAstbar_Ca_LVAst.axonal",
        "gamma_CaDynamics_E2.somatic",
        "gSKv3_1bar_SKv3_1.somatic",
        "gSK_E2bar_SK_E2.somatic",
        "gCa_HVAbar_Ca_HVA.somatic",
        "gNaTs2_tbar_NaTs2_t.somatic",
        "decay_CaDynamics_E2.somatic",
        "gCa_LVAstbar_Ca_LVAst.somatic",
    ]
    if dj:
        lowercase = [l.lower() for l in labels_params]
        labels_params = [l.replace(".", "_") for l in lowercase]
    return labels_params


def return_x_names(dj: bool = True):
    labels_x = [
        "Step1.soma.AHP_depth_abs",
        "Step1.soma.AHP_depth_abs_slow",
        "Step1.soma.AHP_slow_time",
        "Step1.soma.AP_height",
        "Step1.soma.AP_width",
        "Step1.soma.ISI_CV",
        "Step1.soma.adaptation_index2",
        "Step1.soma.doublet_ISI",
        "Step1.soma.mean_frequency",
        "Step1.soma.time_to_first_spike",
        "Step2.soma.AHP_depth_abs",
        "Step2.soma.AHP_depth_abs_slow",
        "Step2.soma.AHP_slow_time",
        "Step2.soma.AP_height",
        "Step2.soma.AP_width",
        "Step2.soma.ISI_CV",
        "Step2.soma.adaptation_index2",
        "Step2.soma.doublet_ISI",
        "Step2.soma.mean_frequency",
        "Step2.soma.time_to_first_spike",
        "Step3.soma.AHP_depth_abs",
        "Step3.soma.AHP_depth_abs_slow",
        "Step3.soma.AHP_slow_time",
        "Step3.soma.AP_height",
        "Step3.soma.AP_width",
        "Step3.soma.ISI_CV",
        "Step3.soma.adaptation_index2",
        "Step3.soma.doublet_ISI",
        "Step3.soma.mean_frequency",
        "Step3.soma.time_to_first_spike",
        "bAP.dend1.AP_amplitude_from_voltagebase",
        "bAP.dend2.AP_amplitude_from_voltagebase",
        "bAP.soma.AP_height",
        "bAP.soma.AP_width",
        "bAP.soma.Spikecount",
    ]
    if dj:
        lowercase = [l.lower() for l in labels_x]
        labels_x = [l.replace(".", "_") for l in lowercase]
    return labels_x


def return_gt(as_pd: bool = True):
    gt = np.array(
        [
            [
                0.026145,
                0.004226,
                0.000143,
                3.137968,
                0.089259,
                0.002910,
                0.006827,
                0.007104,
                0.000990,
                0.973538,
                1.021945,
                287.198731,
                0.008752,
                0.000609,
                0.303472,
                0.008407,
                0.000994,
                0.983955,
                210.485284,
                0.000333,
            ]
        ],
        dtype=np.float32,
    )
    if as_pd:
        names = return_names(dj=True)
        gt = pd.DataFrame(gt, columns=names)
    return gt


def return_xo(summstats: bool = True, as_pd: bool = True):
    model_path = "/mnt/qb/macke/mdeistler57/multicompartment/multicompartment/models"
    if summstats:
        xo = pd.read_pickle(os.path.join(model_path, "l5pc/xo.pkl"))
        if not as_pd:
            xo = xo.to_numpy()
    else:
        xo = pd.read_pickle(os.path.join(model_path, "l5pc/xo_trace.pkl"))
    return xo


def return_protocol_subset(all_protocol_definitions, protocol_subset):
    if protocol_subset is not None:
        used_features = {}
        for key in all_protocol_definitions:
            if key in protocol_subset:
                used_features[key] = all_protocol_definitions[key]
        all_protocol_definitions = used_features
    return all_protocol_definitions


def theta_np_to_pd_l5pc(theta_np: np.ndarray, num_params: int) -> pd.DataFrame:
    names = return_names()[:num_params]
    theta = pd.DataFrame(theta_np, columns=names)
    return theta


def rename_theta_dj_to_bpo(theta: pd.DataFrame) -> pd.DataFrame:
    dj_names = return_names(dj=True)
    bpo_names = return_names(dj=False)
    renaming_dict = dict(zip(dj_names, bpo_names))
    theta.rename(columns=renaming_dict, inplace=True)
    return theta
