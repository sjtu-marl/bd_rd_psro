import numpy as np
import torch
from torch import nn, optim
import pickle
import os
import multiprocessing as mp
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy import stats

LR = 0.1

device = "cpu"
INV_GAUSS_VAR = 0.54
NUMBER_GAUSS = 9
RADIUS = 4

FILE_TRAJ = {
    'rectified': 'rectified.p',
    'psro': 'psro.p',
    'p-psro': 'p_psro.p',
    'dpp': 'dpp.p',
    'distance': 'distance.p',
    'diverge': 'diverge.p',
    'unify': 'unify.p'
}


def fictitious_play(iters=2000, payoffs=None, verbose=False):
    dim = payoffs.shape[0]
    pop = np.random.uniform(0, 1, (1, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    averages = pop
    exps = []
    for i in range(iters):
        average = np.average(pop, axis=0)
        br = get_br_to_strat(average, payoffs=payoffs)
        exp1 = average @ payoffs @ br.T
        exp2 = br @ payoffs @ average.T
        exps.append(exp2 - exp1)
        # if verbose:
        #     print(exp, "exploitability")
        averages = np.vstack((averages, average))
        pop = np.vstack((pop, br))
    return averages, exps


class MyGaussianPDF(nn.Module):
    def __init__(self, mu):
        super(MyGaussianPDF, self).__init__()
        self.mu = mu
        self.cov = INV_GAUSS_VAR * torch.eye(2)
        # self.c = (1./(2*np.pi))
        self.c = 1.

    def forward(self, x):
        return self.c * torch.exp(-0.5 * torch.diagonal((x - self.mu) @ self.cov @ (x - self.mu).t()))


class GMMAgent(nn.Module):
    def __init__(self, mu):
        super(GMMAgent, self).__init__()
        self.gauss = MyGaussianPDF(mu).to(device)
        self.x = nn.Parameter(0.01 * torch.randn(2, dtype=torch.float), requires_grad=False)

    def forward(self):
        return self.gauss(self.x)


class TorchPop:
    def __init__(self, num_learners, seed=0):
        torch.manual_seed(seed)
        self.pop_size = num_learners + 1
        mus_array = []
        for i in range(NUMBER_GAUSS):
            mus_array.append(
                [RADIUS * np.cos(i * 2 * np.pi / NUMBER_GAUSS), RADIUS * np.sin(i * 2 * np.pi / NUMBER_GAUSS)])
        mus = np.array(mus_array)
        # mus = np.array([[2.8722, -0.025255],
        #                 [1.8105, 2.2298],
        #                 [1.8105, -2.2298],
        #                 [-0.61450, 2.8058],
        #                 [-0.61450, -2.8058],
        #                 [-2.5768, 1.2690],
        #                 [-2.5768, -1.2690]]
        #                )
        mus = torch.from_numpy(mus).float().to(device)
        self.mus = mus
        game_array_first_row = []
        game_array_first_row.append(0)
        for _ in range((NUMBER_GAUSS - 1) // 2):
            game_array_first_row.append(1)
        for _ in range((NUMBER_GAUSS - 1) // 2):
            game_array_first_row.append(-1)
        game_array = [game_array_first_row]
        for _ in range(NUMBER_GAUSS - 1):
            game_array.append([game_array[-1][-1]] + game_array[-1][0:-1])
        self.game = torch.from_numpy(np.array(game_array)).float().to(device)
        # self.game = torch.from_numpy(np.array([
        #     [0., 1., 1., 1, -1, -1, -1],
        #     [-1., 0., 1., 1., 1., -1., -1.],
        #     [-1., -1., 0., 1., 1., 1., -1],
        #     [-1., -1., -1., 0, 1., 1., 1.],
        #     [1., -1., -1., -1., 0., 1., 1.],
        #     [1., 1., -1., -1, -1, 0., 1.],
        #     [1., 1., 1., -1., -1., -1., 0.]
        # ])).float().to(device)

        self.pop = [GMMAgent(mus) for _ in range(self.pop_size)]
        self.pop_hist = [[self.pop[i].x.detach().cpu().clone().numpy()] for i in range(self.pop_size)]

    def visualise_pop(self, br=None, ax=None, color=None, ignore=False):

        def multivariate_gaussian(pos, mu, Sigma):
            """Return the multivariate Gaussian distribution on array pos."""

            n = mu.shape[0]
            Sigma_det = np.linalg.det(Sigma)
            Sigma_inv = np.linalg.inv(Sigma)
            N = np.sqrt((2 * np.pi) ** n * Sigma_det)
            # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
            # way across all the input variables.
            fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
            return np.exp(-fac / 2) / N

        metagame = self.get_metagame(numpy=True)
        metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1]
        agents = [agent.x.detach().cpu().numpy() for agent in self.pop]
        agents = list(zip(*agents))

        # Colors
        if color is None:
            colors = cm.rainbow(np.linspace(0, 1, len(agents[0])))
        else:
            colors = [color] * len(agents[0])

        # fig = plt.figure(figsize=(6, 6))
        ax.scatter(agents[0], agents[1], alpha=1., marker='.', color=colors,
                   s=8 * plt.rcParams['lines.markersize'] ** 2)
        if br is not None:
            ax.scatter(br[0], br[1], marker='.', c='k')
        for i, hist in enumerate(self.pop_hist):
            if metanash[i] < 1e-3:
                if ignore:
                    continue
            if hist:
                hist = list(zip(*hist))
                ax.plot(hist[0], hist[1], alpha=0.8, color=colors[i], linewidth=4)

        # ax = plt.gca()
        for i in range(NUMBER_GAUSS):
            ax.scatter(self.mus[i, 0].item(), self.mus[i, 1].item(), marker='x', c='k')
            for j in range(4):
                delta = 0.025
                range_value = 4.5 * NUMBER_GAUSS / 7
                x = np.arange(-range_value, range_value, delta)
                y = np.arange(-range_value, range_value, delta)
                X, Y = np.meshgrid(x, y)
                pos = np.empty(X.shape + (2,))
                pos[:, :, 0] = X
                pos[:, :, 1] = Y
                Z = multivariate_gaussian(pos, self.mus[i, :].numpy(), INV_GAUSS_VAR * np.eye(2))
                levels = 10
                # levels = np.logspace(0.01, 1, 10, endpoint=True)
                CS = ax.contour(X, Y, Z, levels, colors='k', linewidths=0.5, alpha=0.2)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # ax.clabel(CS, fontsize=9, inline=1)
        # circle = plt.Circle((0, 0), 0.2, color='r')
        # ax.add_artist(circle)
        ax.set_xlim([-4.5, 4.5])
        ax.set_ylim([-4.5, 4.5])

    def get_payoff(self, agent1, agent2):
        p = agent1()
        q = agent2()
        return p @ self.game @ q + 0.5 * (p - q).sum()

    def get_payoff_aggregate(self, agent1, metanash, K):
        # Computes the payoff of agent1 against the aggregated first :K agents using metanash as weights
        agg_agent = metanash[0] * self.pop[0]()
        for k in range(1, K):
            agg_agent += metanash[k] * self.pop[k]()
        return agent1() @ self.game @ agg_agent + 0.5 * (agent1() - agg_agent).sum()

    def get_js_divergence(self, agent1, metanash, K):
        def entropy(p_k):
            p_k = p_k + 1e-8
            p_k = p_k / torch.sum(p_k)
            return -torch.sum(p_k * torch.log(p_k))

        agg_agent = metanash[0] * self.pop[0]()
        for k in range(1, K):
            agg_agent += metanash[k] * self.pop[k]()
        agent1_values = agent1()
        agent1_values = agent1_values / torch.sum(agent1_values)
        agg_agent = agg_agent / torch.sum(agg_agent)
        return 2 * entropy((agent1_values + agg_agent) / 2) - entropy(agent1_values) - entropy(agg_agent)

    def get_payoff_aggregate_weights(self, agent1, weights, K):
        # Computes the payoff of agent1 against the aggregated first :K agents using metanash as weights
        agg_agent = weights[0] * self.pop[0]()
        for k in range(1, len(weights)):
            agg_agent += weights[k] * self.pop[k]()
        return agent1() @ self.game @ agg_agent + 0.5 * (agent1() - agg_agent).sum()

    def get_br_to_strat(self, metanash, lr, nb_iters=20):
        br = GMMAgent(self.mus)
        br.x = nn.Parameter(0.1 * torch.randn(2, dtype=torch.float), requires_grad=False)
        br.x.requires_grad = True
        optimiser = optim.Adam(br.parameters(), lr=lr)
        for _ in range(int(nb_iters * 10)):
            loss = -self.get_payoff_aggregate(br, metanash, self.pop_size, )
            # Optimise !
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        return br

    def get_metagame(self, k=None, numpy=False):
        if k == None:
            k = self.pop_size
        if numpy:
            with torch.no_grad():
                metagame = torch.zeros(k, k)
                for i in range(k):
                    for j in range(k):
                        metagame[i, j] = self.get_payoff(self.pop[i], self.pop[j])
                return metagame.detach().cpu().clone().numpy()
        else:
            metagame = torch.zeros(k, k)
            for i in range(k):
                for j in range(k):
                    metagame[i, j] = self.get_payoff(self.pop[i], self.pop[j])
            return metagame

    def add_new(self):
        with torch.no_grad():
            self.pop.append(GMMAgent(self.mus))
            self.pop_hist.append([self.pop[-1].x.detach().cpu().clone().numpy()])
            self.pop_size += 1

    def get_exploitability(self, metanash, lr, nb_iters=20):
        br = self.get_br_to_strat(metanash, lr, nb_iters=nb_iters)
        with torch.no_grad():
            exp = self.get_payoff_aggregate(br, metanash, self.pop_size).item()
        return exp


def get_br_to_strat(strat, payoffs, verbose=False):
    row_weighted_payouts = strat @ payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br


def fp_for_non_symmetric_game(emp_game_matrix, iters):
    row_player_dim = emp_game_matrix.shape[0]
    column_player_dim = emp_game_matrix.shape[1]
    row_pop = np.random.uniform(0, 1, (1, row_player_dim))
    row_pop = row_pop / row_pop.sum()
    column_pop = np.random.uniform(0, 1, (1, column_player_dim))
    column_pop = column_pop / column_pop.sum()
    for i in range(iters):
        row_avg = np.average(row_pop, axis=0)
        column_avg = np.average(column_pop, axis=0)
        br_column = get_br_to_strat(row_avg, emp_game_matrix)
        br_row = get_br_to_strat(column_avg, -emp_game_matrix.T)
        row_pop = np.vstack((row_pop, br_row))
        column_pop = np.vstack((column_pop, br_column))
    row_avg = np.average(row_pop, axis=0)
    column_avg = np.average(column_pop, axis=0)
    # print(f"Nash is {row_avg}")
    return row_avg, column_avg, row_avg @ emp_game_matrix @ column_avg.T


def get_pop_effectivity(self, iters=100, br_iter=2, verbose=True):
    pop_size = len(self.pop)
    curr_pop_metagame = self.get_metagame(numpy=True)
    # metanash = fictitious_play(iters=2000, payoffs=curr_pop_metagame)[0][-1]
    metanash = np.random.random(size=(pop_size,))
    opponent_pop = [self.get_br_to_strat(metanash / sum(metanash), lr=LR, nb_iters=br_iter)]

    meta_game = []
    nash_value_list = []
    with torch.no_grad():
        row_vec = []
        for i in range(pop_size):
            row_vec.append(self.get_payoff(self.pop[i], opponent_pop[0]).item())
    meta_game.append(row_vec)
    for i in range(iters):
        # solve
        nash_row, _, nash_value = fp_for_non_symmetric_game(emp_game_matrix=np.array(meta_game).T, iters=1000)
        # scale for column player
        br_column = self.get_br_to_strat(nash_row, lr=0.1, nb_iters=br_iter)
        opponent_pop.append(br_column)
        row_vec = []
        for j in range(pop_size):
            row_vec.append(self.get_payoff(self.pop[j], br_column).item())
        meta_game.append(row_vec)
        if verbose:
            print(f"{i}th iterations: nash value is {nash_value}")
        nash_value_list.append(nash_value)
    return meta_game, nash_value_list


def compute_pop_effectivity(args):
    num_experiment = args[1]
    np.random.seed(num_experiment)
    torch.random.manual_seed(num_experiment)
    print(f"this is experiment {num_experiment}")
    args = args[0]
    path_result = args['path_result']
    seed = args['seed']
    br_iter = args['br_iter']
    total_iter = args['total_iter']
    nash_value_dict = {}
    for key in FILE_TRAJ.keys():
        nash_value_dict[key] = []
    for i, key in enumerate(FILE_TRAJ.keys()):
        # if not os.path.exists(os.path.join(path_result, f"{seed}_" + FILE_TRAJ[key]) + '.p'):
        # if not os.path.exists(os.path.join(path_result, FILE_TRAJ[key]) + '.p'):
        #     continue
        if (key == "psro" or key == "rectified"):
            d = pickle.load(
                open(os.path.join(" diverse_psro/results/gauss/20210514-105734", FILE_TRAJ[key]) + '.p',
                     'rb'))
        else:
            d = pickle.load(open(os.path.join(path_result, FILE_TRAJ[key]) + '.p', 'rb'))
        torch_pop = d['pop']
        print(f"key is {key}")
        meta_game, nash_value_dict[key] = get_pop_effectivity(torch_pop, total_iter, br_iter)
    return nash_value_dict


def compute_pop_effectivity_mp(path_result, seed, br_iter, number_experiment, total_iter):
    args = {'path_result': path_result, 'seed': seed, 'br_iter': br_iter, 'total_iter': total_iter}
    pool = mp.Pool()
    result = pool.map(compute_pop_effectivity, [(args, i) for i in range(number_experiment)])

    psro_exps = []
    pipeline_psro_exps = []
    dpp_psro_exps = []
    rectified_exps = []
    distance_psro_exps = []
    diverge_psro_exps = []
    unify_psro_exps = []
    psro = True
    pipeline_psro = True
    rectified = True
    dpp_psro = True,
    distance_psro = True
    diverge_psro = True
    unify_psro = True

    for r in result:
        psro_exps.append(r['rectified'])
        pipeline_psro_exps.append(r['p-psro'])
        dpp_psro_exps.append(r['dpp'])
        rectified_exps.append(r['rectified'])
        distance_psro_exps.append(r['distance'])
        diverge_psro_exps.append(r['diverge'])
        unify_psro_exps.append(r['unify'])
    d = {
        'psro_exps': psro_exps,
        'pipeline_psro_exps': pipeline_psro_exps,
        'dpp_psro_exps': dpp_psro_exps,
        'rectified_exps': rectified_exps,
        'distance_psro_exps': distance_psro_exps,
        'diverge_psro_exps': diverge_psro_exps,
        'unify_psro_exps': unify_psro_exps
    }
    pickle.dump(d, open(os.path.join(path_result, 'pop_eff_data.p'), 'wb'))

    def plot_error(data, label=''):
        min_len = min([len(i) for i in data])
        data = [i[2:min_len] for i in data]
        avg = np.mean(np.array(data), axis=0)
        error_bars = stats.sem(np.array(data))
        plt.plot(avg, label=label)
        plt.fill_between([i for i in range(avg.shape[0])],
                         (avg - error_bars).reshape(-1),
                         (avg + error_bars).reshape(-1), alpha=alpha)
        print(f"For label {label}: {avg[-1] * 100} + {error_bars[-1] * 100}")

    alpha = .4
    logscale = False
    for j in range(1):
        fig_handle = plt.figure()
        if psro:
            if j == 0:
                plot_error(psro_exps, label='PSRO')
        if pipeline_psro:
            if j == 0:
                plot_error(pipeline_psro_exps, label='P-PSRO')
        if rectified:
            if j == 0:
                length = min([len(l) for l in rectified_exps])
                for i, l in enumerate(rectified_exps):
                    rectified_exps[i] = rectified_exps[i][:length]
                plot_error(rectified_exps, label='PSRO-rN')
        if dpp_psro:
            if j == 0:
                plot_error(dpp_psro_exps, label='DPP-PSRO')
        if distance_psro:
            if j == 0:
                plot_error(distance_psro_exps, label='P-PSRO w. RD')
        if diverge_psro:
            if j == 0:
                plot_error(diverge_psro_exps, label='P-PSRO w. BD')
        if unify_psro:
            if j == 0:
                plot_error(unify_psro_exps, label='P-PSRO w. BD&RD')

        plt.legend(loc="upper left")
        plt.title('Population Effectivity')

        if logscale and (j == 0):
            plt.yscale('log')
        if j == 0:
            plt.savefig(os.path.join(path_result, f'pop_effec_{br_iter}_figure_{seed}_' + str(j) + '.pdf'))
    plt.show()


def run_traj(path_result):
    titles = {
        'rectified': 'PSRO-rN',
        'dpp': 'DPP-PSRO',
        'p-psro': 'P-PSRO',
        'psro': 'PSRO',
        'distance': 'P-PSRO w. RD',
        'diverge': 'P-PSRO w. BD',
        'unify': 'P-PSRO w. BD&RD'
    }
    pops = {}
    fig1, axs1 = plt.subplots(1, 7, figsize=(5 * 4, 5 * 1), dpi=500)
    axs1 = axs1.flatten()
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:blue', 'tab:blue', 'tab:blue']
    d_psro = pickle.load(open(os.path.join(" diverse_psro/results/gauss/20210514-105734", FILE_TRAJ['psro']) + '.p', 'rb'))['pop']
    d_rectified = pickle.load(open(os.path.join(" diverse_psro/results/gauss/20210514-105734", FILE_TRAJ['rectified']) + '.p','rb'))['pop']
    # d_psro.visualise_pop(ax=axs1[0], color=colors[0])
    # axs1[0].set_title(titles["psro"], fontsize=23)
    # d_rectified.visualise_pop(ax=axs1[1], color=colors[1])
    # axs1[1].set_title(titles["rectified"], fontsize=23)

    print(f"expl is {expl_from_pop(d_psro)} for PSRO")
    print(f"expl is {expl_from_pop(d_rectified)} for Rectified")

    for i, key in enumerate(FILE_TRAJ.keys()):
        ax = axs1[i]
        if not os.path.exists(os.path.join(path_result, FILE_TRAJ[key]) + '.p'):
            continue
        d = pickle.load(open(os.path.join(path_result, FILE_TRAJ[key]) + '.p', 'rb'))
        pops[FILE_TRAJ[key]] = d['pop']
        print(f"expl is {expl_from_pop(pops[FILE_TRAJ[key]])} for {key}")
        # pops[FILE_TRAJ[key]].visualise_pop(ax=ax, color=colors[i])
        # ax.set_title(titles[key], fontsize=23)

    # fig1.tight_layout()
    # fig1.savefig(os.path.join(path_result, 'trajectories.pdf'))
    #
    # pops = {}
    # fig2, axs2 = plt.subplots(1, 7, figsize=(5 * 4, 5 * 1), dpi=200)
    # axs2 = axs2.flatten()
    # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:blue', 'tab:blue', 'tab:blue']
    #
    # for i, key in enumerate(FILE_TRAJ.keys()):
    #     ax = axs2[i]
    #     if not os.path.exists(os.path.join(path_result, FILE_TRAJ[key]) + '.p'):
    #         continue
    #     d = pickle.load(open(os.path.join(path_result, FILE_TRAJ[key]) + '.p', 'rb'))
    #     pops[FILE_TRAJ[key]] = d['pop']
    #     pops[FILE_TRAJ[key]].visualise_pop(ax=ax, color=colors[i], ignore=True)
    #     ax.set_title(titles[key])
    #
    # fig2.tight_layout()
    # fig2.savefig(os.path.join(path_result, 'trajectories_ignore.pdf'))


def pop_eff():
    path_result = " diverse_psro/results/gauss/20210503-175335"
    seed = 0
    br_iter = 2.5
    number_experiment = 10
    total_iter = 30
    compute_pop_effectivity_mp(path_result, seed, br_iter, number_experiment, total_iter)
    print(f"opponent strength is {br_iter * 10}")

def run_expl(path_result):
    path_result, num_experiment = path_result[0], path_result[1]
    np.random.seed(num_experiment)
    torch.random.manual_seed(num_experiment)

    pops = {}
    d_psro = pickle.load(open(os.path.join(" diverse_psro/results/gauss/20210514-105734", FILE_TRAJ['psro']) + '.p', 'rb'))['pop']
    d_rectified = pickle.load(open(os.path.join(" diverse_psro/results/gauss/20210514-105734", FILE_TRAJ['rectified']) + '.p','rb'))['pop']
    result = {}
    result["psro"] = expl_from_pop(d_psro)
    result["rectified"] = expl_from_pop(d_rectified)

    for i, key in enumerate(FILE_TRAJ.keys()):
        if not os.path.exists(os.path.join(path_result, FILE_TRAJ[key]) + '.p'):
            continue
        d = pickle.load(open(os.path.join(path_result, FILE_TRAJ[key]) + '.p', 'rb'))
        pops[FILE_TRAJ[key]] = d['pop']
        result[key] = expl_from_pop(pops[FILE_TRAJ[key]])
    return result

def run_expl_mp(path_result, number_experiment):
    pool = mp.Pool()
    results = pool.map(run_expl, [(path_result, i*10) for i in range(number_experiment)])
    result_dict = {}
    for key in FILE_TRAJ.keys():
        result_dict[key] = []
    for r in results:
        for key in FILE_TRAJ.keys():
            result_dict[key].append(r[key])
    for key in FILE_TRAJ.keys():
        print(f"key is {key} mean is {np.mean(np.array(result_dict[key]))} std is {stats.sem(np.array(result_dict[key]))}")



def expl_from_pop(pop):
    meta_game = pop.get_metagame(numpy=True)
    metanash = fictitious_play(iters=2000, payoffs=meta_game)[0][-1]
    br = pop.get_br_to_strat(metanash, 0.01, nb_iters=50)
    expl = pop.get_payoff_aggregate(br, metanash, pop.pop_size)
    return expl.item()

def look_for_best():
    exp_name_pop = "diverge_psro_pop"
    exp_name_expl = "diverge_psro_exps"
    # directory_list = [r"E:\diverse_psro\results\gauss\20210603-152742",
    #                   r"E:\diverse_psro\results\gauss\20210603-152808",
    #                   r"E:\diverse_psro\results\gauss\20210603-152827",
    #                   r"E:\diverse_psro\results\gauss\20210603-152848",
    #                   r"E:\diverse_psro\results\gauss\20210603-152923",
    #                   r"E:\diverse_psro\results\gauss\20210603-153002"]
    directory_list = [r"E:\diverse_psro\results\gauss\20210604-084532",
                      r"E:\diverse_psro\results\gauss\20210604-084502",
                      r"E:\diverse_psro\results\gauss\20210604-084342",
                      r"E:\diverse_psro\results\gauss\20210603-153121",
                      r"E:\diverse_psro\results\gauss\20210603-153138",
                      r"E:\diverse_psro\results\gauss\20210603-153200",
                      r"E:\diverse_psro\results\gauss\20210603-153238",
                      r"E:\diverse_psro\results\gauss\20210603-153307",
                      r"E:\diverse_psro\results\gauss\20210603-153324"]
    lmb = [0.001, 0.005, 0.01, 0.05, 0.5, 1, 5, 10, 50]
    for j, path in enumerate(directory_list):
        print("###########")
        exp_this_iter = []
        for i in range(5):
            # if i == 1:
            #     continue
            d = pickle.load(open(os.path.join(path, f'checkpoint_{i}'), 'rb'))
            d_pop = d[exp_name_pop]
            # d_exp = d[exp_name_expl]
            d_exp = get_pop_effectivity(d_pop, iters=30, br_iter=2, verbose=False)[1]
            # re - calculating is {expl_from_pop(d_pop)}
            print(f"expl is {d_exp[-1]} for checkpoint {i} exps ")
            # exp_this_seed = expl_from_pop(d_pop)
            exp_this_iter.append(d_exp[-1])
        print(f"lmb is {lmb[j]*1500} avg exp is {sum(exp_this_iter)/len(exp_this_iter)} std is {stats.sem(np.array(exp_this_iter))}")


if __name__ == '__main__':
    # for _ in range(1):
    #     run_traj(" diverse_psro/results/gauss/20210503-175335")
    # plt.show()
    # run_expl_mp(" diverse_psro/results/gauss/20210513-185910", 20)
    # pop_eff()

    look_for_best()
#  diverse_psro/results/gauss/20210513-185910 +  diverse_psro/results/gauss/20210514-105734
