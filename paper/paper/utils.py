from turtle import position
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def mean_confidence_interval(data, confidence=0.95):
    return sns.utils.ci(sns.algorithms.bootstrap(data))
    
    
def fig_metric_mpl(
    df: pd.DataFrame,
    metric: str,
    width: float = 3.5,
    height: float = 0.8,
    xlabel: bool = True,
    title: bool = False,
    task_name: str = ""
):
    col_npe = "#8c6bb1"
    col_snpe = "#810f7c"
    col_tsnpe = "k"
    
    fig, ax = plt.subplots(1, 3, figsize=(width, height))
    
    snpe_1000 = df.query("algorithm == 'SNPE' & num_simulations == '10³'")[metric].mean()
    npe_1000 = df.query("algorithm == 'NPE' & num_simulations == '10³'")[metric].mean()
    tsnpe_1000 = df.query("algorithm == 'tsnpe' & num_simulations == 1000")[metric].mean()
    snpe_10000 = df.query("algorithm == 'SNPE' & num_simulations == '10⁴'")[metric].mean()
    npe_10000 = df.query("algorithm == 'NPE' & num_simulations == '10⁴'")[metric].mean()
    tsnpe_10000 = df.query("algorithm == 'tsnpe' & num_simulations == 10000")[metric].mean()
    snpe_100000 = df.query("algorithm == 'SNPE' & num_simulations == '10⁵'")[metric].mean()
    npe_100000 = df.query("algorithm == 'NPE' & num_simulations == '10⁵'")[metric].mean()
    tsnpe_100000 = df.query("algorithm == 'tsnpe' & num_simulations == 100000")[metric].mean()
    
    lower_conf_snpe_1000, upper_conf_snpe_1000 = mean_confidence_interval(df.query("algorithm == 'SNPE' & num_simulations == '10³'")[metric])
    lower_conf_npe_1000, upper_conf_npe_1000 = mean_confidence_interval(df.query("algorithm == 'NPE' & num_simulations == '10³'")[metric])
    lower_conf_tsnpe_1000, upper_conf_tsnpe_1000 = mean_confidence_interval(df.query("algorithm == 'tsnpe' & num_simulations == 1000")[metric])
    lower_conf_snpe_10000, upper_conf_snpe_10000 = mean_confidence_interval(df.query("algorithm == 'SNPE' & num_simulations == '10⁴'")[metric])
    lower_conf_npe_10000, upper_conf_npe_10000 = mean_confidence_interval(df.query("algorithm == 'NPE' & num_simulations == '10⁴'")[metric])
    lower_conf_tsnpe_10000, upper_conf_tsnpe_10000 = mean_confidence_interval(df.query("algorithm == 'tsnpe' & num_simulations == 10000")[metric])
    lower_conf_snpe_100000, upper_conf_snpe_100000 = mean_confidence_interval(df.query("algorithm == 'SNPE' & num_simulations == '10⁵'")[metric])
    lower_conf_npe_100000, upper_conf_npe_100000 = mean_confidence_interval(df.query("algorithm == 'NPE' & num_simulations == '10⁵'")[metric])
    lower_conf_tsnpe_100000, upper_conf_tsnpe_100000 = mean_confidence_interval(df.query("algorithm == 'tsnpe' & num_simulations == 100000")[metric])
    
    ax[0].plot([npe_1000, npe_10000, npe_100000], c=col_npe)
    ax[1].plot([snpe_1000, snpe_10000, snpe_100000], c=col_snpe)
    ax[2].plot([tsnpe_1000, tsnpe_10000, tsnpe_100000], c=col_tsnpe)
    ax[0].scatter([0, 1, 2], [npe_1000, npe_10000, npe_100000], s=20, c=col_npe)
    ax[1].scatter([0, 1, 2], [snpe_1000, snpe_10000, snpe_100000], s=20, c=col_snpe)
    ax[2].scatter([0, 1, 2], [tsnpe_1000, tsnpe_10000, tsnpe_100000], s=20, c=col_tsnpe)
    
    ax[0].plot([0, 0], [lower_conf_npe_1000, upper_conf_npe_1000], c=col_npe)
    ax[1].plot([0, 0], [lower_conf_snpe_1000, upper_conf_snpe_1000], c=col_snpe)
    ax[2].plot([0, 0], [lower_conf_tsnpe_1000, upper_conf_tsnpe_1000], c=col_tsnpe)
    ax[0].plot([1, 1], [lower_conf_npe_10000, upper_conf_npe_10000], c=col_npe)
    ax[1].plot([1, 1], [lower_conf_snpe_10000, upper_conf_snpe_10000], c=col_snpe)
    ax[2].plot([1, 1], [lower_conf_tsnpe_10000, upper_conf_tsnpe_10000], c=col_tsnpe)
    ax[0].plot([2, 2], [lower_conf_npe_100000, upper_conf_npe_100000], c=col_npe)
    ax[1].plot([2, 2], [lower_conf_snpe_100000, upper_conf_snpe_100000], c=col_snpe)
    ax[2].plot([2, 2], [lower_conf_tsnpe_100000, upper_conf_tsnpe_100000], c=col_tsnpe)
    
    for i in range(3):
        ax[i].set_ylim([0.46, 1.04])
        ax[i].spines["bottom"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
        ax[i].grid(axis="y", alpha=0.5)
        ax[i].set_axisbelow(True)
        ax[i].tick_params(axis=u'both', which=u'both',length=0)
        ax[i].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax[i].set_xticks([0, 1, 2])
        ax[i].set_xlim([-0.2, 2.2])
        if xlabel:
            ax[i].set_xticklabels([r"10$^3$", r"10$^4$", r"10$^5$"], rotation=90)
            ax[1].set_xlabel("Number of simulations")
        else:
            ax[i].set_xticklabels([])
        
    if title:
        ax[2].set_title(task_name)
    ax[0].set_ylabel("C2ST")
    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    return fig

def fig_diagnostic(
    df: pd.DataFrame,
    metric: str,
    width: float = 3.5,
    height: float = 0.7,
    xlabel: bool = True,
    ylabel: bool = False,
    legend: bool = False,
    title: bool = False,
    task_name: str = ""
):  
    acceptance_rate = extract_diagnostic(df.query("algorithm == 'tsnpe'"), "acceptance rate")
    for i in range(3):
        acceptance_rate[i][:, 1:] = np.power(10, acceptance_rate[i][:, 1:])
    gt_in_support = extract_diagnostic(df.query("algorithm == 'tsnpe'"))

    av_acceptance_rates = []
    for i in range(3):
        av_acceptance_rate = np.mean(acceptance_rate[i], axis=0)
        av_acceptance_rates.append(av_acceptance_rate)
    av_acceptance_rate = np.asarray(av_acceptance_rates)

    av_gt_in_supports = []
    for i in range(3):
        av_gt_in_support = np.mean(gt_in_support[i], axis=0)
        av_gt_in_supports.append(av_gt_in_support)
    av_gt_in_support = np.asarray(av_gt_in_supports)

    # av_gt_in_support = np.mean(gt_in_support, axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(width, height))
    ax[0].plot(np.arange(1, 11), av_acceptance_rate[0], c="k", alpha=0.3)
    ax[0].plot(np.arange(1, 11), av_acceptance_rate[1], c="k", alpha=0.55)
    ax[0].plot(np.arange(1, 11), av_acceptance_rate[2], c="k", alpha=1.0)
    ax[0].set_ylim([0, 1.01])
    ax[0].set_xlim([1, 10])
    ax[0].set_xticks(np.arange(1, 11))
    if ylabel:
        ax[0].set_ylabel("acceptance rate")
    
    ax[1].plot(np.arange(1, 11), av_gt_in_support[0], c="k", alpha=0.3)
    ax[1].plot(np.arange(1, 11), av_gt_in_support[1], c="k", alpha=0.55)
    ax[1].plot(np.arange(1, 11), av_gt_in_support[2], c="k", alpha=1.0)
    ax[1].set_ylim([0.9, 1.0001])
    ax[1].set_xlim([1, 10])
    ax[1].set_xticks(np.arange(1, 11))
    if xlabel == False:
        ax[0].set_xticklabels([])
        ax[1].set_xticklabels([])
    else:
        ax[0].set_xlabel("Round")
        ax[1].set_xlabel("Round")
    if ylabel:
        ax[1].set_ylabel("gt in support")
    if legend:
        ax[1].legend([r"10$^3$", r"10$^4$", r"10$^5$"], ncol=3, loc="upper right", bbox_to_anchor=[1.1, 1.65, 0.0, 0.0], handlelength=0.8, handletextpad=0.4, columnspacing=1.0)
    plt.subplots_adjust(wspace=0.8)
    return fig

def extract_diagnostic(df, key="gt in support"):
    full = []
    for n in [1000, 10000, 100000]:
        a = df.query(f"num_simulations=={n}")[key]
        b = np.asarray(a.to_numpy())
        all_lists = []
        for aa in b:
            aaa = aa[1:-2]
            splitted = list(aaa.split(", "))
            new_ = []
            for s in splitted:
                new_.append(float(s))
            all_lists.append(new_)
        all_lists = np.asarray(all_lists)
        full.append(all_lists)
    return np.asarray(full)