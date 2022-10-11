import datajoint as dj


schema = dj.schema("mdeistler_multicompartment", locals())


@schema
class L5PC_20D_theta(dj.Manual):
    definition = """
        ind: int
        ---
        round: int
        gnats2_tbar_nats2_t_apical: float
        gskv3_1bar_skv3_1_apical: float
        gimbar_im_apical: float
        gnata_tbar_nata_t_axonal: float
        gk_tstbar_k_tst_axonal: float
        gamma_cadynamics_e2_axonal: float
        gnap_et2bar_nap_et2_axonal: float
        gsk_e2bar_sk_e2_axonal: float
        gca_hvabar_ca_hva_axonal: float
        gk_pstbar_k_pst_axonal: float
        gskv3_1bar_skv3_1_axonal: float
        decay_cadynamics_e2_axonal: float
        gca_lvastbar_ca_lvast_axonal: float
        gamma_cadynamics_e2_somatic: float
        gskv3_1bar_skv3_1_somatic: float
        gsk_e2bar_sk_e2_somatic: float
        gca_hvabar_ca_hva_somatic: float
        gnats2_tbar_nats2_t_somatic: float
        decay_cadynamics_e2_somatic: float
        gca_lvastbar_ca_lvast_somatic: float
        id: varchar(10)
        path: varchar(30)
        """


@schema
class L5PC_20D_x(dj.Manual):
    definition = """
        ind: int
        ---
        round: int
        step1_soma_ahp_depth_abs = NULL : float
        step1_soma_ahp_depth_abs_slow = NULL : float
        step1_soma_ahp_slow_time = NULL : float
        step1_soma_ap_height = NULL : float
        step1_soma_ap_width = NULL : float
        step1_soma_isi_cv = NULL : float
        step1_soma_adaptation_index2 = NULL : float
        step1_soma_doublet_isi = NULL : float
        step1_soma_mean_frequency = NULL : float
        step1_soma_time_to_first_spike = NULL : float
        step2_soma_ahp_depth_abs = NULL : float
        step2_soma_ahp_depth_abs_slow = NULL : float
        step2_soma_ahp_slow_time = NULL : float
        step2_soma_ap_height = NULL : float
        step2_soma_ap_width = NULL : float
        step2_soma_isi_cv = NULL : float
        step2_soma_adaptation_index2 = NULL : float
        step2_soma_doublet_isi = NULL : float
        step2_soma_mean_frequency = NULL : float
        step2_soma_time_to_first_spike = NULL : float
        step3_soma_ahp_depth_abs = NULL : float
        step3_soma_ahp_depth_abs_slow = NULL : float
        step3_soma_ahp_slow_time = NULL : float
        step3_soma_ap_height = NULL : float
        step3_soma_ap_width = NULL : float
        step3_soma_isi_cv = NULL : float
        step3_soma_adaptation_index2 = NULL : float
        step3_soma_doublet_isi = NULL : float
        step3_soma_mean_frequency = NULL : float
        step3_soma_time_to_first_spike = NULL : float
        bap_dend1_ap_amplitude_from_voltagebase = NULL : float
        bap_dend2_ap_amplitude_from_voltagebase = NULL : float
        bap_soma_ap_height = NULL : float
        bap_soma_ap_width = NULL : float
        bap_soma_spikecount = NULL : float
        id: varchar(10)
        path: varchar(30)
        """


@schema
class L5PC_20D_gradient(dj.Manual):
    definition = """
        -> L5PC_20D_theta
        gradient_ind: int
        ---
        gradient_direction = NULL: varchar(35)
        diff = NULL: float
        step1_soma_ahp_depth_abs = NULL : float
        step1_soma_ahp_depth_abs_slow = NULL : float
        step1_soma_ahp_slow_time = NULL : float
        step1_soma_ap_height = NULL : float
        step1_soma_ap_width = NULL : float
        step1_soma_isi_cv = NULL : float
        step1_soma_adaptation_index2 = NULL : float
        step1_soma_doublet_isi = NULL : float
        step1_soma_mean_frequency = NULL : float
        step1_soma_time_to_first_spike = NULL : float
        step2_soma_ahp_depth_abs = NULL : float
        step2_soma_ahp_depth_abs_slow = NULL : float
        step2_soma_ahp_slow_time = NULL : float
        step2_soma_ap_height = NULL : float
        step2_soma_ap_width = NULL : float
        step2_soma_isi_cv = NULL : float
        step2_soma_adaptation_index2 = NULL : float
        step2_soma_doublet_isi = NULL : float
        step2_soma_mean_frequency = NULL : float
        step2_soma_time_to_first_spike = NULL : float
        step3_soma_ahp_depth_abs = NULL : float
        step3_soma_ahp_depth_abs_slow = NULL : float
        step3_soma_ahp_slow_time = NULL : float
        step3_soma_ap_height = NULL : float
        step3_soma_ap_width = NULL : float
        step3_soma_isi_cv = NULL : float
        step3_soma_adaptation_index2 = NULL : float
        step3_soma_doublet_isi = NULL : float
        step3_soma_mean_frequency = NULL : float
        step3_soma_time_to_first_spike = NULL : float
        bap_dend1_ap_amplitude_from_voltagebase = NULL : float
        bap_dend2_ap_amplitude_from_voltagebase = NULL : float
        bap_soma_ap_height = NULL : float
        bap_soma_ap_width = NULL : float
        bap_soma_spikecount = NULL : float
        id: varchar(10)
        path: varchar(30)
        """


@schema
class L5PC_20D_perturbation_theta(dj.Manual):
    definition = """
        ind: int
        ---
        gnats2_tbar_nats2_t_apical: float
        gskv3_1bar_skv3_1_apical: float
        gimbar_im_apical: float
        gnata_tbar_nata_t_axonal: float
        gk_tstbar_k_tst_axonal: float
        gamma_cadynamics_e2_axonal: float
        gnap_et2bar_nap_et2_axonal: float
        gsk_e2bar_sk_e2_axonal: float
        gca_hvabar_ca_hva_axonal: float
        gk_pstbar_k_pst_axonal: float
        gskv3_1bar_skv3_1_axonal: float
        decay_cadynamics_e2_axonal: float
        gca_lvastbar_ca_lvast_axonal: float
        gamma_cadynamics_e2_somatic: float
        gskv3_1bar_skv3_1_somatic: float
        gsk_e2bar_sk_e2_somatic: float
        gca_hvabar_ca_hva_somatic: float
        gnats2_tbar_nats2_t_somatic: float
        decay_cadynamics_e2_somatic: float
        gca_lvastbar_ca_lvast_somatic: float
        id: varchar(10)
        path: varchar(30)
        """


@schema
class L5PC_20D_perturbation_x(dj.Manual):
    definition = """
        ind: int
        ---
        step1_soma_ahp_depth_abs = NULL : float
        step1_soma_ahp_depth_abs_slow = NULL : float
        step1_soma_ahp_slow_time = NULL : float
        step1_soma_ap_height = NULL : float
        step1_soma_ap_width = NULL : float
        step1_soma_isi_cv = NULL : float
        step1_soma_adaptation_index2 = NULL : float
        step1_soma_doublet_isi = NULL : float
        step1_soma_mean_frequency = NULL : float
        step1_soma_time_to_first_spike = NULL : float
        step2_soma_ahp_depth_abs = NULL : float
        step2_soma_ahp_depth_abs_slow = NULL : float
        step2_soma_ahp_slow_time = NULL : float
        step2_soma_ap_height = NULL : float
        step2_soma_ap_width = NULL : float
        step2_soma_isi_cv = NULL : float
        step2_soma_adaptation_index2 = NULL : float
        step2_soma_doublet_isi = NULL : float
        step2_soma_mean_frequency = NULL : float
        step2_soma_time_to_first_spike = NULL : float
        step3_soma_ahp_depth_abs = NULL : float
        step3_soma_ahp_depth_abs_slow = NULL : float
        step3_soma_ahp_slow_time = NULL : float
        step3_soma_ap_height = NULL : float
        step3_soma_ap_width = NULL : float
        step3_soma_isi_cv = NULL : float
        step3_soma_adaptation_index2 = NULL : float
        step3_soma_doublet_isi = NULL : float
        step3_soma_mean_frequency = NULL : float
        step3_soma_time_to_first_spike = NULL : float
        bap_dend1_ap_amplitude_from_voltagebase = NULL : float
        bap_dend2_ap_amplitude_from_voltagebase = NULL : float
        bap_soma_ap_height = NULL : float
        bap_soma_ap_width = NULL : float
        bap_soma_spikecount = NULL : float
        id: varchar(10)
        path: varchar(30)
        """
