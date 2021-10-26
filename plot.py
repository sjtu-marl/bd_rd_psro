import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import json
import pickle
import os
import seaborn as sns
sns.set()
from scipy.stats import norm
# plt.rcParams['axes.facecolor'] = 'grey'


def plot_error(data, label='', alpha=0.4):
    data = [i[0:150] for i in data]
    # x = list(range(150))
    # data = [i[20:150] for i in data]
    # x = x[20:]
    data_mean = np.mean(np.array(data), axis=0)
    error_bars = stats.sem(np.array(data))
    plt.plot(data_mean, label=label)
    plt.fill_between([i for i in range(data_mean.size)],
                     np.squeeze(data_mean - error_bars),
                     np.squeeze(data_mean + error_bars), alpha=alpha)

def comparison_for_group_game():
    logscale = True
    psro = True
    pipeline_psro = True
    dpp_psro = True
    rectified = False
    self_play = False
    distance_psro = False
    rectified_distance_psro = False
    diverge_psro = True

    psro_exps = []
    psro_cardinality = []
    pipeline_psro_exps = []
    pipeline_psro_cardinality = []
    rectified_exps = []
    rectified_cardinality = []
    self_play_exps = []
    self_play_cardinality = []
    diverge_psro_exps = []
    diverge_psro_cardinality = []
    dpp_psro_exps = []
    dpp_psro_cardinality = []
    rectified_distance_psro_exps = []
    rectified_distance_psro_cardinality = []
    distance_psro_exps = []
    distance_psro_cardinality = []

    with open(os.path.join(r"E:\diverse_psro\results\20210427-171309_AlphaStar_0.5_0.85", 'data.p'), "rb") as f:
        result = pickle.load(f)
    diverge_psro_exps = result['diverge_psro_exps']
    diverge_psro_cardinality = result['diverge_psro_exps']

    # with open(os.path.join(r'E:\diverse_psro\results\20210325-030933_3-move parity game 2_0.5_0.85', 'data.p'), 'rb') as f:
    #     result = pickle.load(f)
    dpp_psro_exps = result['dpp_psro_exps']
    dpp_psro_cardinality = result['dpp_psro_cardinality']
    psro_exps = result['psro_exps']
    psro_cardinality = result['psro_cardinality']

    with open(os.path.join(r"E:\diverse_psro\results\20210427-171309_AlphaStar_0.5_0.85\20210428-113730_AlphaStar_0.5_0.85", 'data.p'), 'rb') as f:
        result = pickle.load(f)
    pipeline_psro_exps = result['pipeline_psro_exps']
    pipeline_psro_cardinality = result['pipeline_psro_cardinality']
    # with open(os.path.join(r'E:\diverse_psro\results\20210313-134804_2000_0.5_0.85', 'data.p'), 'rb') as f:
    #     result = pickle.load(f)
    # rectified_distance_psro_exps = result['rectified_distance_psro_exps']
    # rectified_distance_psro_cardinality = result['rectified_distance_psro_cardinality']
    # with open(os.path.join(r'E:\diverse_psro\results\20210326-033836_3-move parity game 2_0.5_0.85', 'data.p'), 'rb') as f:
    #     result = pickle.load(f)
    # psro_exps = result['psro_exps']
    # psro_cardinality = result['psro_cardinality']
    # rectified_exps = result['rectified_exps']
    # rectified_cardinality = result['rectified_cardinality']
    # self_play_exps = result['self_play_exps']
    # self_play_cardinality = result['self_play_cardinality']

    for j in range(2):
        fig_handle = plt.figure()

        if psro:
            if j == 0:
                plot_error(psro_exps, label='PSRO')
            elif j == 1:
                plot_error(psro_cardinality, label='PSRO')
        if pipeline_psro:
            if j == 0:
                plot_error(pipeline_psro_exps, label='P-PSRO')
            elif j == 1:
                plot_error(pipeline_psro_cardinality, label='P-PSRO')
        if rectified:
            if j == 0:
                length = min([len(l) for l in rectified_exps])
                for i, l in enumerate(rectified_exps):
                    rectified_exps[i] = rectified_exps[i][:length]
                plot_error(rectified_exps, label='PSRO-rN')
            elif j == 1:
                length = min([len(l) for l in rectified_cardinality])
                for i, l in enumerate(rectified_cardinality):
                    rectified_cardinality[i] = rectified_cardinality[i][:length]
                plot_error(rectified_cardinality, label='PSRO-rN')
        if self_play:
            if j == 0:
                plot_error(self_play_exps, label='Self-play')
            elif j == 1:
                plot_error(self_play_cardinality, label='Self-play')
        if dpp_psro:
            if j == 0:
                plot_error(dpp_psro_exps, label='Ours')
            elif j == 1:
                plot_error(dpp_psro_cardinality, label='Ours')
        if distance_psro:
            if j == 0:
                plot_error(distance_psro_exps, label='distance_psro')
            elif j == 1:
                plot_error(distance_psro_cardinality, label='distance_psro')
        if rectified_distance_psro:
            if j == 0:
                plot_error(rectified_distance_psro_exps, label='rectified_distance_psro')
            elif j == 1:
                plot_error(rectified_distance_psro_cardinality, label='rectified_distance_psro')
        if diverge_psro:
            if j == 0:
                plot_error(diverge_psro_exps, label='diverge_psro')
            elif j == 1:
                plot_error(diverge_psro_cardinality, label='diverge_psro')

        plt.legend(loc="upper left")
        # plt.title('Dim {:d}'.format(500))
        plt.title('AlphaStar')

        if logscale and (j == 0):
            plt.yscale('log')
        PATH_RESULTS = r'E:\diverse_psro\results\20210427-171309_AlphaStar_0.5_0.85'
        if not os.path.exists(PATH_RESULTS):
            os.makedirs(PATH_RESULTS)

        plt.savefig(os.path.join(PATH_RESULTS, 'new_' + str(j) + '.pdf'))


def comparison_for_alpha_star():
    logscale = True
    psro = True
    pipeline_psro = True
    dpp_psro = True
    rectified = True
    self_play = True
    distance_psro = True
    rectified_distance_psro = False
    diverge_psro = True
    unified_psro = True

    psro_exps = []
    psro_cardinality = []
    pipeline_psro_exps = []
    pipeline_psro_cardinality = []
    rectified_exps = []
    rectified_cardinality = []
    self_play_exps = []
    self_play_cardinality = []
    diverge_psro_exps = []
    diverge_psro_cardinality = []
    dpp_psro_exps = []
    dpp_psro_cardinality = []
    rectified_distance_psro_exps = []
    rectified_distance_psro_cardinality = []
    distance_psro_exps = []
    distance_psro_cardinality = []
    unified_psro_exps = []
    unified_psro_cardinality = []

    with open(os.path.join(r"E:\diverse_psro\results\20210427-110734_AlphaStar_0.5_0.85", 'data.p'), "rb") as f:
        result = pickle.load(f)
    diverge_psro_exps = result['diverge_psro_exps']
    diverge_psro_cardinality = result['diverge_psro_exps']

    with open(os.path.join(r'E:\diverse_psro\results\alpha_star\20210307-092816_AlphaStar_0.5_0.85', 'data.p'), 'rb') as f:
        result = pickle.load(f)
    dpp_psro_exps = result['dpp_psro_exps']
    dpp_psro_cardinality = result['dpp_psro_cardinality']

    # with open(os.path.join(r'E:\diverse_psro\results\20210323-133539_600_0.5_0.85', 'data.p'), 'rb') as f:
    #     result = pickle.load(f)
    pipeline_psro_exps = result['pipeline_psro_exps']
    pipeline_psro_cardinality = result['pipeline_psro_cardinality']
    rectified_exps = result['rectified_exps']
    rectified_cardinality = result['rectified_cardinality']
    self_play_exps = result['self_play_exps']
    self_play_cardinality = result['self_play_cardinality']
    psro_exps = result['psro_exps']
    psro_cardinality = result['psro_cardinality']
    with open(os.path.join(r'E:\diverse_psro\results\20210528-224841_AlphaStar_0.5_0.85', 'data.p'), 'rb') as f:
        result = pickle.load(f)
    # rectified_distance_psro_exps = result['rectified_distance_psro_exps']
    # rectified_distance_psro_cardinality = result['rectified_distance_psro_cardinality']
    distance_psro_exps = result['distance_psro_exps']
    distance_psro_cardinality = result['distance_psro_cardinality']

    with open(os.path.join(r"E:\diverse_psro\results\20210428-212953_AlphaStar_0.5_0.85", "data.p"), "rb") as f:
        result = pickle.load(f)
    unified_psro_exps = result['diverge_psro_exps']
    unified_psro_cardinality = result['dpp_psro_cardinality']

    for j in range(2):
        fig_handle = plt.figure()

        if psro:
            if j == 0:
                plot_error(psro_exps, label='PSRO')
            elif j == 1:
                plot_error(psro_cardinality, label='PSRO')
        if pipeline_psro:
            if j == 0:
                plot_error(pipeline_psro_exps, label='P-PSRO')
            elif j == 1:
                plot_error(pipeline_psro_cardinality, label='P-PSRO')
        if rectified:
            if j == 0:
                length = min([len(l) for l in rectified_exps])
                for i, l in enumerate(rectified_exps):
                    rectified_exps[i] = rectified_exps[i][:length]
                plot_error(rectified_exps, label='PSRO-rN')
            elif j == 1:
                length = min([len(l) for l in rectified_cardinality])
                for i, l in enumerate(rectified_cardinality):
                    rectified_cardinality[i] = rectified_cardinality[i][:length]
                plot_error(rectified_cardinality, label='PSRO-rN')
        if self_play:
            if j == 0:
                plot_error(self_play_exps, label='Self-play')
            elif j == 1:
                plot_error(self_play_cardinality, label='Self-play')
        if dpp_psro:
            if j == 0:
                plot_error(dpp_psro_exps, label='DPP-PSRO')
            elif j == 1:
                plot_error(dpp_psro_cardinality, label='DPP-PSRO')
        if distance_psro:
            if j == 0:
                plot_error(distance_psro_exps, label='P-PSRO w. RD')
            elif j == 1:
                plot_error(distance_psro_cardinality, label='P-PSRO w. RD')
        if rectified_distance_psro:
            if j == 0:
                plot_error(rectified_distance_psro_exps, label='rectified_distance_psro')
            elif j == 1:
                plot_error(rectified_distance_psro_cardinality, label='rectified_distance_psro')
        if diverge_psro:
            if j == 0:
                plot_error(diverge_psro_exps, label='P-PSRO w. BD')
            elif j == 1:
                plot_error(diverge_psro_cardinality, label='P-PSRO w. BD')
        if unified_psro:
            if j == 0:
                plot_error(unified_psro_exps, label="P-PSRO w. BD&RD")
            elif j == 1:
                plot_error(unified_psro_cardinality, label="P-PSRO w. BD&RD")

        plt.legend(loc="upper right")
        # plt.title('Dim {:d}'.format(500))
        plt.title('AlphaStar')
        plt.xlabel("Training Iterations")
        plt.ylabel("Exploitability")
        if logscale and (j == 0):
            plt.yscale('log')
        PATH_RESULTS = r'E:\diverse_psro\results\alpha_star'
        if not os.path.exists(PATH_RESULTS):
            os.makedirs(PATH_RESULTS)

        plt.savefig(os.path.join(PATH_RESULTS, 'unified_figure_' + str(j) + '.pdf'))


def tune_alpha_star(algo='distance_psro'):
    exp_string = f"{algo}_exps"
    cardinality_string = f"{algo}_cardinality"
    result_folder_list = [r'E:\diverse_psro\results\20210322-130024_AlphaStar_0.5_0.85',
                          r'E:\diverse_psro\results\20210322-131217_AlphaStar_0.5_0.85',
                          r'E:\diverse_psro\results\20210322-133509_AlphaStar_0.5_0.85',
                          r'E:\diverse_psro\results\20210322-133538_AlphaStar_0.5_0.85',
                          r'E:\diverse_psro\results\20210322-135318_AlphaStar_0.5_0.85']
    labels_list = ['100', '10', '50', '500', '500-100-50']
    exps_list = []
    cards_list = []
    for path in result_folder_list:
        with open(os.path.join(path, 'data.p'), 'rb') as f:
            result = pickle.load(f)
        exps_list.append(result[exp_string])
        cards_list.append(result[cardinality_string])

    for i in range(5):
        plot_error(exps_list[i], label=labels_list[i])
    plt.legend(loc="upper left")
    plt.title('diverge psro')
    plt.yscale('log')
    PATH_RESULTS = r'E:\diverse_psro\results\alpha_star'
    if not os.path.exists(PATH_RESULTS):
        os.makedirs(PATH_RESULTS)
    plt.savefig(os.path.join(PATH_RESULTS, f'figure_0_diverge_psro.pdf'))

    fig_handle = plt.figure()
    for i in range(5):
        plot_error(cards_list[i], label=labels_list[i])
    plt.legend(loc="upper left")
    plt.title('rectified distance psro')
    PATH_RESULTS = r'E:\diverse_psro\results\alpha_star'
    if not os.path.exists(PATH_RESULTS):
        os.makedirs(PATH_RESULTS)
    plt.savefig(os.path.join(PATH_RESULTS, f'figure_1_diverge_psro.pdf'))


def plot():
    logscale = True
    psro = False
    pipeline_psro = False
    dpp_psro = True
    rectified = False
    self_play = False
    distance_psro = True

    with open(os.path.join(r"E:\diverse_psro\results\20210322-131217_AlphaStar_0.5_0.85", 'data.p'), "rb") as f:
        result = pickle.load(f)
    distance_psro_exps = result['distance_psro_exps']
    distance_psro_cardinality = result['distance_psro_cardinality']

    with open(os.path.join(r"E:\diverse_psro\results\20210308-134554_AlphaStar_0.5_0.875", 'data.p'), "rb") as f:
        result = pickle.load(f)
    dpp_psro_exps = result['dpp_psro_exps']
    dpp_psro_cardinality = result['dpp_psro_cardinality']
    rectified_exps = result['rectified_exps']
    rectified_cardinality = result['rectified_cardinality']
    self_play_exps = result['self_play_exps']
    self_play_cardinality = result['self_play_cardinality']

    # with open(os.path.join(r"E:\diverse_psro\results\plot_500_0.5\20210307-172053_500_0.5", 'data.p'), "rb") as f:
    #     result = pickle.load(f)
    psro_exps = result['psro_exps']
    psro_cardinality = result['psro_cardinality']
    pipeline_psro_exps = result['pipeline_psro_exps']
    pipeline_psro_cardinality = result['pipeline_psro_cardinality']

    for j in range(2):
        fig_handle = plt.figure()

        if psro:
            if j == 0:
                plot_error(psro_exps, label='PSRO')
            elif j == 1:
                plot_error(psro_cardinality, label='PSRO')
        if pipeline_psro:
            if j == 0:
                plot_error(pipeline_psro_exps, label='P-PSRO')
            elif j == 1:
                plot_error(pipeline_psro_cardinality, label='P-PSRO')
        if rectified:
            if j == 0:
                length = min([len(l) for l in rectified_exps])
                for i, l in enumerate(rectified_exps):
                    rectified_exps[i] = rectified_exps[i][:length]
                plot_error(rectified_exps, label='PSRO-rN')
            elif j == 1:
                length = min([len(l) for l in rectified_cardinality])
                for i, l in enumerate(rectified_cardinality):
                    rectified_cardinality[i] = rectified_cardinality[i][:length]
                plot_error(rectified_cardinality, label='PSRO-rN')
        if self_play:
            if j == 0:
                plot_error(self_play_exps, label='Self-play')
            elif j == 1:
                plot_error(self_play_cardinality, label='Self-play')
        if dpp_psro:
            if j == 0:
                plot_error(dpp_psro_exps, label='Ours')
            elif j == 1:
                plot_error(dpp_psro_cardinality, label='Ours')
        if distance_psro:
            if j == 0:
                plot_error(distance_psro_exps, label='distance_psro')
            elif j == 1:
                plot_error(distance_psro_cardinality, label='distance_psro')

        plt.legend(loc="upper left")
        # plt.title('Dim {:d}'.format(500))
        plt.title('AlphaStar')

        if logscale and (j == 0):
            plt.yscale('log')
        PATH_RESULTS = r'E:\diverse_psro\results\alpha_star'
        if not os.path.exists(PATH_RESULTS):
            os.makedirs(PATH_RESULTS)

        plt.savefig(os.path.join(PATH_RESULTS, 'compare_figure_' + str(j) + '.pdf'))

def comparision_for_pe():
    logscale = True
    psro = True
    pipeline_psro = True
    dpp_psro = True
    rectified = True
    self_play = True
    distance_psro = True
    rectified_distance_psro = False
    diverge_psro = True
    unified_psro = True

    psro_exps = []
    psro_cardinality = []
    pipeline_psro_exps = []
    pipeline_psro_cardinality = []
    rectified_exps = []
    rectified_cardinality = []
    self_play_exps = []
    self_play_cardinality = []
    diverge_psro_exps = []
    diverge_psro_cardinality = []
    dpp_psro_exps = []
    dpp_psro_cardinality = []
    rectified_distance_psro_exps = []
    rectified_distance_psro_cardinality = []
    distance_psro_exps = []
    distance_psro_cardinality = []
    unified_psro_exps = []
    unified_psro_cardinality = []
    with open(os.path.join(r"E:\diverse_psro\results\20210528-221149_AlphaStar_0.5_0.85", 'data.p'), "rb") as f:
        result = pickle.load(f)
    distance_psro_exps = result['distance_psro_exps']
    distance_psro_cardinality = result['distance_psro_cardinality']

    with open(os.path.join(r"E:\diverse_psro\results\20210527-195259_AlphaStar_0.5_0.85", 'data.p'), "rb") as f:
        result = pickle.load(f)
    diverge_psro_exps = result['diverge_psro_exps']
    diverge_psro_cardinality = result['diverge_psro_exps']

    dpp_psro_exps = result['dpp_psro_exps']
    dpp_psro_cardinality = result['dpp_psro_cardinality']

    pipeline_psro_exps = result['pipeline_psro_exps']
    pipeline_psro_cardinality = result['pipeline_psro_cardinality']
    rectified_exps = result['rectified_exps']
    rectified_cardinality = result['rectified_cardinality']
    self_play_exps = result['self_play_exps']
    self_play_cardinality = result['self_play_cardinality']
    psro_exps = result['psro_exps']
    psro_cardinality = result['psro_cardinality']

    with open(os.path.join(r"E:\diverse_psro\results\20210528-213639_AlphaStar_0.5_0.85", "data.p"), "rb") as f:
        result = pickle.load(f)
    unified_psro_exps = result['diverge_psro_exps']
    unified_psro_cardinality = result['diverge_psro_cardinality']

    for j in range(2):
        fig_handle = plt.figure()

        if psro:
            if j == 0:
                plot_error(psro_exps, label='PSRO')
            elif j == 1:
                plot_error(psro_cardinality, label='PSRO')
        if pipeline_psro:
            if j == 0:
                plot_error(pipeline_psro_exps, label='P-PSRO')
            elif j == 1:
                plot_error(pipeline_psro_cardinality, label='P-PSRO')
        if rectified:
            if j == 0:
                length = min([len(l) for l in rectified_exps])
                for i, l in enumerate(rectified_exps):
                    rectified_exps[i] = rectified_exps[i][:length]
                plot_error(rectified_exps, label='PSRO-rN')
            elif j == 1:
                length = min([len(l) for l in rectified_cardinality])
                for i, l in enumerate(rectified_cardinality):
                    rectified_cardinality[i] = rectified_cardinality[i][:length]
                plot_error(rectified_cardinality, label='PSRO-rN')
        if self_play:
            if j == 0:
                plot_error(self_play_exps, label='Self-play')
            elif j == 1:
                plot_error(self_play_cardinality, label='Self-play')
        if dpp_psro:
            if j == 0:
                plot_error(dpp_psro_exps, label='DPP-PSRO')
            elif j == 1:
                plot_error(dpp_psro_cardinality, label='DPP-PSRO')
        if distance_psro:
            if j == 0:
                plot_error(distance_psro_exps, label='P-PSRO w. RD')
            elif j == 1:
                plot_error(distance_psro_cardinality, label='P-PSRO w. RD')
        if rectified_distance_psro:
            if j == 0:
                plot_error(rectified_distance_psro_exps, label='rectified_distance_psro')
            elif j == 1:
                plot_error(rectified_distance_psro_cardinality, label='rectified_distance_psro')
        if diverge_psro:
            if j == 0:
                plot_error(diverge_psro_exps, label='P-PSRO w. BD')
            elif j == 1:
                plot_error(diverge_psro_cardinality, label='P-PSRO w. BD')
        if unified_psro:
            if j == 0:
                plot_error(unified_psro_exps, label="P-PSRO w. BD&RD")
            elif j == 1:
                plot_error(unified_psro_cardinality, label="P-PSRO w. BD&RD")

        plt.legend(loc="upper right")
        # plt.title('Dim {:d}'.format(500))
        plt.title('AlphaStar')
        plt.xlabel("Training Iterations")
        plt.ylabel("Negative Population Effectivity")
        if logscale and (j == 0):
            plt.yscale('log')
        PATH_RESULTS = r'E:\diverse_psro\results\alpha_star'
        if not os.path.exists(PATH_RESULTS):
            os.makedirs(PATH_RESULTS)
        # plt.grid()
        plt.savefig(os.path.join(PATH_RESULTS, 'unified_pe_figure_' + str(j) + '.pdf'))

def ablation_study():
    stat_name = "distance_psro_pe"
    directory_list = [r"E:\diverse_psro\results\20210603-112936_AlphaStar_0.5_0.2",
                      r"E:\diverse_psro\results\20210603-113002_AlphaStar_0.5_0.4",
                      r"E:\diverse_psro\results\20210603-113026_AlphaStar_0.5_0.6",
                      r"E:\diverse_psro\results\20210603-113047_AlphaStar_0.5_0.8",
                      r"E:\diverse_psro\results\20210603-113101_AlphaStar_0.5_1.0",
                      r"E:\diverse_psro\results\20210603-113121_AlphaStar_0.5_1.2",
                      r"E:\diverse_psro\results\20210603-225528_AlphaStar_0.5_1.5",
                      r"E:\diverse_psro\results\20210603-225540_AlphaStar_0.5_2.0",
                      r"E:\diverse_psro\results\20210603-224708_AlphaStar_0.5_10.0",
                      r"E:\diverse_psro\results\20210603-224739_AlphaStar_0.5_5.0",
                      r"E:\diverse_psro\results\20210603-224810_AlphaStar_0.5_20.0"]

    # directory_list = [r"E:\diverse_psro\results\20210603-113240_AlphaStar_0.5_0.2",
    #                   r"E:\diverse_psro\results\20210603-113258_AlphaStar_0.5_0.4",
    #                   r"E:\diverse_psro\results\20210603-113314_AlphaStar_0.5_0.6",
    #                   r"E:\diverse_psro\results\20210603-113335_AlphaStar_0.5_0.8",
    #                   r"E:\diverse_psro\results\20210603-113350_AlphaStar_0.5_1.0",
    #                   r"E:\diverse_psro\results\20210603-113408_AlphaStar_0.5_1.2",
    #                   r"E:\diverse_psro\results\20210603-225732_AlphaStar_0.5_1.5",
    #                   r"E:\diverse_psro\results\20210603-225746_AlphaStar_0.5_2.0",
    #                   r"E:\diverse_psro\results\20210603-224906_AlphaStar_0.5_10.0",
    #                   r"E:\diverse_psro\results\20210603-224850_AlphaStar_0.5_5.0",
    #                   r"E:\diverse_psro\results\20210603-224919_AlphaStar_0.5_20.0"]

    label_list = ["0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.5", "2", "10", "5", "20"]
    stat_list = []
    fig_handle = plt.figure()
    PATH_RESULTS = r'E:\diverse_psro\results\alpha_star'

    for i in range(len(label_list)):
        with open(os.path.join(directory_list[i], "data.p"), "rb") as f:
            result = pickle.load(f)
        if (label_list[i] == "10" or label_list[i] == "20"):
            continue
        stat_list.append(result[stat_name])
        plot_error(stat_list[-1], label=f"$\lambda_2$=cdf({-float(label_list[i])})")
        print(f"label is {label_list[i]} cdf is {1-norm.cdf(float(label_list[i]))}")

    plt.legend(loc="upper right")
    # plt.title('Dim {:d}'.format(500))
    plt.title('AlphaStar')
    plt.xlabel("Training Iterations")
    plt.ylabel("Negative Population Effectivity")
    plt.yscale('log')

    plt.savefig(os.path.join(PATH_RESULTS, 'ablation_distance_pe' + '.pdf'))


if __name__ == '__main__':
    # tune_alpha_star(algo='diverge_psro')
    # plot()
    # comparison_for_alpha_star()
    # comparison_for_group_game()
    # comparision_for_pe()
    # with open(os.path.join(r"/home/diverse_psro/results/20210603-112008_AlphaStar_0.5_0.85", 'data.p'), "rb") as f:
    #     result = pickle.load(f)
    # print(result)
    ablation_study()
