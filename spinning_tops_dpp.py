import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy import stats
import pickle
import json
from numpy.random import RandomState
import argparse
import multiprocessing as mp
from shutil import copyfile
from utils import js_divergence, pop_effective_diversity, generate_group_transitive_game

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

np.random.seed(0)

parser = argparse.ArgumentParser(description='All Spinning Top Payoffs DPP')
parser.add_argument('--nb_iters', type=int, default=150)
parser.add_argument('--nb_exps', type=int, default=5)
parser.add_argument('--mp', default=True, action='store_false', help='Set --mp for False, otherwise leave it for True')
parser.add_argument('--game_name', type=str, default='AlphaStar')
parser.add_argument('--lambda_weight', type=float, default='0.85')
args = parser.parse_args()

LR = 0.5
TH = 0.03
LAMBDA = args.lambda_weight
expected_card = []
sizes = []

time_string = time.strftime("%Y%m%d-%H%M%S")
PATH_RESULTS = os.path.join('results',
                            time_string + '_' + str(args.game_name) + '_' + str(LR) + '_' + str(LAMBDA))
if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)
print(f'The directory is {PATH_RESULTS}')
dst = os.path.join(PATH_RESULTS, os.path.basename(__file__).split('.')[0] + time_string + '.py')
if not os.path.isfile(dst):
    copyfile(__file__, dst)

# Search over the pure strategies to find the BR to a strategy
def get_br_to_strat(strat, payoffs=None, verbose=False):
    row_weighted_payouts = strat @ payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br


# Fictituous play as a nash equilibrium solver
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


# Solve exploitability of a nash equilibrium over a fixed population
def get_exploitability(pop, payoffs, iters=1000):
    emp_game_matrix = pop @ payoffs @ pop.T
    averages, _ = fictitious_play(payoffs=emp_game_matrix, iters=iters)
    strat = averages[-1] @ pop  # Aggregate
    test_br = get_br_to_strat(strat, payoffs=payoffs)
    exp1 = strat @ payoffs @ test_br.T
    exp2 = test_br @ payoffs @ strat
    return exp2 - exp1


def joint_loss(pop, payoffs, meta_nash, k, lambda_weight, lr):
    dim = payoffs.shape[0]

    br = np.zeros((dim,))
    cards = []

    if np.random.randn() < lambda_weight:
        aggregated_enemy = meta_nash @ pop[:k]
        values = payoffs @ aggregated_enemy.T
        br[np.argmax(values)] = 1
        # print('\nbr')
    else:
        for i in range(dim):
            br_tmp = np.zeros((dim,))
            br_tmp[i] = 1.

            pop_k = lr * br_tmp + (1 - lr) * pop[k]
            pop_tmp = np.vstack((pop[:k], pop_k))
            M = pop_tmp @ payoffs @ pop_tmp.T
            metanash_tmp, _ = fictitious_play(payoffs=M, iters=1000)
            L = np.diag(metanash_tmp[-1]) @ M @ M.T @ np.diag(metanash_tmp[-1])
            # L = M @ M.T
            l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
            cards.append(l_card)
        br[np.argmax(cards)] = 1
        # print('\nDiverse')
    return br


def divergence_loss(pop, payoffs, meta_nash, k, lambda_weight, lr, i):
    dim = payoffs.shape[0]
    br = np.zeros((dim,))
    if i <= 75:
        alpha = 500
    elif i <= 150:
        alpha = 100
    else:
        alpha = 50
    if np.random.randn() < lambda_weight:
        aggregated_enemy = meta_nash @ pop[:k]
        values = payoffs @ aggregated_enemy.T
        br[np.argmax(values)] = 1
        # print(f'Best Response {np.argmax(values)}')
    else:
        aggregated_enemy = meta_nash @ pop[:k]
        values = payoffs @ aggregated_enemy.T

        aggregated_enemy = aggregated_enemy.reshape(-1)
        # min_index = [i for i in range(len(aggregated_enemy)) if aggregated_enemy[i] == np.min(aggregated_enemy)]
        diverse_response = [values[i] + alpha * js_divergence(i, aggregated_enemy) for i in
                            range(len(aggregated_enemy))]
        selected_index = np.argmax(diverse_response)
        br[selected_index] = 1
        # print(f'Diverse: value[{np.argmax(values)}]={np.max(values)} diverse[{selected_index}]={np.max(diverse_response)}')

    return br


def distance_loss(pop, payoffs, meta_nash, k, lambda_weight, lr):
    dim = payoffs.shape[0]

    br = np.zeros((dim,))
    cards = []

    if np.random.randn() < lambda_weight:
        aggregated_enemy = meta_nash @ pop[:k]
        values = payoffs @ aggregated_enemy.T
        br[np.argmax(values)] = 1
    else:
        for i in range(dim):
            br_tmp = np.zeros((dim,))
            br_tmp[i] = 1.

            pop_k = lr * br_tmp + (1 - lr) * pop[k]
            pop_tmp = np.vstack((pop[:k], pop_k))
            M = pop_tmp @ payoffs @ pop[:k].T
            old_payoff = M[0:-1].T
            new_vec = M[-1].reshape(-1, 1)
            distance = distance_solver(old_payoff, new_vec)
            cards.append(distance)
        br[np.argmax(cards)] = 1

    return br


def rectified_distance_loss(pop, payoffs, meta_nash, k, lambda_weight, lr):
    dim = payoffs.shape[0]

    br = np.zeros((dim,))
    cards = []

    if np.random.randn() < lambda_weight:
        aggregated_enemy = meta_nash @ pop[:k]
        values = payoffs @ aggregated_enemy.T
        br[np.argmax(values)] = 1
    else:
        for i in range(dim):
            br_tmp = np.zeros((dim,))
            br_tmp[i] = 1.

            pop_k = lr * br_tmp + (1 - lr) * pop[k]
            pop_tmp = np.vstack((pop[:k], pop_k))
            M = pop_tmp @ payoffs @ pop[:k].T
            old_payoff = M[0:-1].T
            new_vec = M[-1].reshape(-1, 1)
            new_vec[new_vec < 0] = 0
            distance = distance_solver(old_payoff, new_vec)
            cards.append(distance)
        br[np.argmax(cards)] = 1

    return br


def distance_solver(A, b):
    One = np.ones(shape=(A.shape[1], 1))
    I = np.identity(A.shape[0])
    A_pinv = np.linalg.pinv(A)
    I_minus_AA_pinv = I - A @ A_pinv
    Sigma_min = min(np.linalg.svd(A.T, full_matrices=True)[1])
    distance = ((Sigma_min ** 2) / A.shape[1]) * ((1 - (One.T @ A_pinv @ b)[0, 0]) ** 2) + np.square(
        I_minus_AA_pinv @ b).sum()
    return distance


def psro_steps(iters=5, payoffs=None, verbose=False, seed=0,
               num_learners=4, improvement_pct_threshold=.03, lr=.2, loss_func='dpp', full=False):
    dim = payoffs.shape[0]

    r = np.random.RandomState(seed)
    pop = r.uniform(0, 1, (1 + num_learners, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    exp = get_exploitability(pop, payoffs, iters=1000)
    pop_effectivity = pop_effective_diversity(pop, payoffs, iters=2000)

    exps = [exp]
    pop_eff = [pop_effectivity]

    M = pop @ payoffs @ pop.T
    L = M @ M.T
    l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
    l_cards = [l_card]

    learner_performances = [[.1] for i in range(num_learners + 1)]
    population_strategy_list = []
    for i in range(iters):
        # Define the weighting towards diversity as a function of the fixed population size, this is currently a hyperparameter
        lambda_weight = LAMBDA
        print('\niteration: ', i, ' exp full: ', exps[-1])
        print('size of pop: ', pop.shape[0])

        for j in range(num_learners):
            # first learner (when j=num_learners-1) plays against normal meta Nash
            # second learner plays against meta Nash with first learner included, etc.
            k = pop.shape[0] - j - 1
            emp_game_matrix = pop[:k] @ payoffs @ pop[:k].T
            meta_nash, _ = fictitious_play(payoffs=emp_game_matrix, iters=1000)
            population_strategy = meta_nash[-1] @ pop[:k]  # aggregated enemy according to nash
            population_strategy_list.append(population_strategy)
            if loss_func == 'br':
                # standard PSRO
                br = get_br_to_strat(population_strategy, payoffs=payoffs)
            else:
                # Diverse PSRO
                if loss_func == 'dpp':
                    br = joint_loss(pop, payoffs, meta_nash[-1], k, lambda_weight, lr)
                    br_orig = get_br_to_strat(population_strategy, payoffs=payoffs)
                elif loss_func == 'distance':
                    br = distance_loss(pop, payoffs, meta_nash[-1], k, lambda_weight, lr)
                elif loss_func == 'rectified_distance':
                    br = rectified_distance_loss(pop, payoffs, meta_nash[-1], k, lambda_weight, lr)
                else:
                    br = divergence_loss(pop, payoffs, meta_nash[-1], k, lambda_weight, lr, i)

            # Update the mixed strategy towards the pure strategy which is returned as the best response to the
            # nash equilibrium that is being trained against.
            pop[k] = lr * br + (1 - lr) * pop[k]
            performance = pop[k] @ payoffs @ population_strategy.T + 1  # make it positive for pct calculation
            # selected_policy = [population_strategy[i] for i in [76]]
            learner_performances[k].append(performance)
            # print(f"iteration {i} learner {j + 1} br is {np.argmax(br)} performance is {performance}")
            # print(f"population is {selected_policy}")
            # values = population_strategy @ payoffs
            # print(f'best_value[{np.argmin(values)}] = {np.min(values)} selected_value[76] = {[values[i] for i in [76]]}')
            # print(f"iteration {i} learner {j + 1} br is {np.argmax(br)} "
            #       f"performance is {performance} ratio is {performance / learner_performances[k][-2]}")

            # if the first learner plateaus, add a new policy to the population
            if j == num_learners - 1 and performance / learner_performances[k][-2] - 1 < improvement_pct_threshold:
                learner = np.random.uniform(0, 1, (1, dim))
                learner = learner / learner.sum(axis=1)[:, None]
                pop = np.vstack((pop, learner))
                learner_performances.append([0.1])
                # one step distance loss update
                # for i in range(1):
                    # exp = pop_effective_diversity(pop, payoffs, iters=2000)
                    # print(f'expl before distance update is {exp}')
                    # exps.append(exp)
                    # if loss_func == "br":
                    #     br = get_br_to_strat(population_strategy, payoffs=payoffs)
                    # else:
                    # br = distance_loss(pop, payoffs, meta_nash[-1], k, lambda_weight, lr)
                    # pop[k] = lr * br + (1 - lr) * pop[k]
                # k = pop.shape[0] - j - 1
                # emp_game_matrix = pop[:k] @ payoffs @ pop[:k].T
                # meta_nash, _ = fictitious_play(payoffs=emp_game_matrix, iters=1000)
                # print(f'new policy added nash is {meta_nash[-1][-1]} for the new policy')

        # calculate exploitability for meta Nash of whole population
        exp = get_exploitability(pop, payoffs, iters=1000)
        pop_effectivity = pop_effective_diversity(pop, payoffs, iters=2000)
        print(f'expl is {exp}')
        # print(f"pop eff is {pop_effectivity}")
        exps.append(exp)
        pop_eff.append(pop_effectivity)
        # exps.append(pop_effectivity)

        M = pop @ payoffs @ pop.T
        L = M @ M.T
        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
        l_cards.append(l_card)

    return pop, exps, l_cards, pop_eff


# Define the self-play algorithm
def self_play_steps(iters=10, payoffs=None, verbose=False, improvement_pct_threshold=.03, lr=.2, seed=0):
    dim = payoffs.shape[0]
    r = np.random.RandomState(seed)
    pop = r.uniform(0, 1, (2, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    exp = get_exploitability(pop, payoffs, iters=1000)
    exps = [exp]
    performances = [.01]

    M = pop @ payoffs @ pop.T
    L = M @ M.T
    l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
    l_cards = [l_card]

    for i in range(iters):
        # if i % 10 == 0:
        #    print('iteration: ', i, 'exploitability: ', exps[-1])
        br = get_br_to_strat(pop[-2], payoffs=payoffs)
        pop[-1] = lr * br + (1 - lr) * pop[-1]
        performance = pop[-1] @ payoffs @ pop[-2].T + 1
        performances.append(performance)
        if performance / performances[-2] - 1 < improvement_pct_threshold:
            learner = np.random.uniform(0, 1, (1, dim))
            learner = learner / learner.sum(axis=1)[:, None]
            pop = np.vstack((pop, learner))
        exp = get_exploitability(pop, payoffs, iters=1000)
        exps.append(exp)

        M = pop @ payoffs @ pop.T
        L = M @ M.T
        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
        l_cards.append(l_card)

    return pop, exps, l_cards


# Define the PSRO rectified nash algorithm
def psro_rectified_steps(iters=10, payoffs=None, verbose=False, eps=1e-2, seed=0,
                         num_start_strats=1, num_pseudo_learners=4, lr=0.3, threshold=0.001):
    dim = payoffs.shape[0]
    r = np.random.RandomState(seed)
    pop = r.uniform(0, 1, (num_start_strats, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    exp = get_exploitability(pop, payoffs, iters=1000)
    exps = [exp]
    counter = 0

    M = pop @ payoffs @ pop.T
    L = M @ M.T
    l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
    l_cards = [l_card]

    while counter < iters * num_pseudo_learners:
        # if counter % (5 * num_pseudo_learners) == 0:
        #    print('iteration: ', int(counter / num_pseudo_learners), ' exp: ', exps[-1])
        #    print('size of population: ', pop.shape[0])

        new_pop = np.copy(pop)
        emp_game_matrix = pop @ payoffs @ pop.T
        averages, _ = fictitious_play(payoffs=emp_game_matrix, iters=iters)

        # go through all policies. If the policy has positive meta Nash mass,
        # find policies it wins against, and play against meta Nash weighted mixture of those policies
        for j in range(pop.shape[0]):
            if counter > iters * num_pseudo_learners:
                return pop, exps, l_cards
            # if positive mass, add a new learner to pop and update it with steps, submit if over thresh
            # keep track of counter
            if averages[-1][j] > eps:
                # create learner
                learner = np.random.uniform(0, 1, (1, dim))
                learner = learner / learner.sum(axis=1)[:, None]
                new_pop = np.vstack((new_pop, learner))
                idx = new_pop.shape[0] - 1

                current_performance = 0.02
                last_performance = 0.01
                while current_performance / last_performance - 1 > threshold:
                    counter += 1
                    mask = emp_game_matrix[j, :]
                    mask[mask >= 0] = 1
                    mask[mask < 0] = 0
                    weights = np.multiply(mask, averages[-1])
                    weights /= weights.sum()
                    strat = weights @ pop
                    br = get_br_to_strat(strat, payoffs=payoffs)
                    new_pop[idx] = lr * br + (1 - lr) * new_pop[idx]
                    last_performance = current_performance
                    current_performance = new_pop[idx] @ payoffs @ strat + 1

                    if counter % num_pseudo_learners == 0:
                        # count this as an 'iteration'

                        # exploitability
                        exp = get_exploitability(new_pop, payoffs, iters=1000)
                        exps.append(exp)

                        M = pop @ payoffs @ pop.T
                        L = M @ M.T
                        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
                        l_cards.append(l_card)

        pop = np.copy(new_pop)

    return pop, exps, l_cards


def run_experiment(param_seed):
    params, seed = param_seed
    iters = params['iters']
    num_threads = params['num_threads']
    lr = params['lr']
    thresh = params['thresh']
    psro = params['psro']
    pipeline_psro = params['pipeline_psro']
    dpp_psro = params['dpp_psro']
    rectified = params['rectified']
    self_play = params['self_play']
    distance_psro = params['distance_psro']
    rectified_distance_psro = params['rectified_distance_psro']
    diverge_psro = params['diverge_psro']

    psro_exps = []
    psro_cardinality = []
    pipeline_psro_exps = []
    pipeline_psro_cardinality = []
    dpp_psro_exps = []
    dpp_psro_cardinality = []
    rectified_exps = []
    rectified_cardinality = []
    self_play_exps = []
    self_play_cardinality = []
    distance_psro_exps = []
    distance_psro_cardinality = []
    rectified_distance_psro_exps = []
    rectified_distance_psro_cardinality = []
    diverge_psro_exps = []
    diverge_psro_cardinality = []

    distance_psro_pe = []
    diverge_psro_pe = []

    print('Experiment: ', seed + 1)
    np.random.seed(seed)
    with open("payoffs_data/" + str(args.game_name) + ".pkl", "rb") as fh:
        payoffs = pickle.load(fh)
    # payoffs = generate_group_transitive_game(1000 // 10, 10)
    if psro:
        print('PSRO')
        pop, exps, cards, pe = psro_steps(iters=iters, num_learners=1, seed=seed + 1,
                                      improvement_pct_threshold=thresh, lr=lr,
                                      payoffs=payoffs, loss_func='br')
        psro_exps = exps
        psro_cardinality = cards
    if pipeline_psro:
        print('Pipeline PSRO')
        pop, exps, cards, pe = psro_steps(iters=iters, num_learners=num_threads, seed=seed + 1,
                                      improvement_pct_threshold=thresh, lr=lr,
                                      payoffs=payoffs, loss_func='br')
        pipeline_psro_exps = exps
        pipeline_psro_cardinality = cards
    if dpp_psro:
        print('Pipeline DPP')
        pop, exps, cards, pe = psro_steps(iters=iters, num_learners=num_threads, seed=seed + 1,
                                      improvement_pct_threshold=thresh, lr=lr,
                                      payoffs=payoffs, loss_func='dpp')
        dpp_psro_exps = exps
        dpp_psro_cardinality = cards
    if rectified:
        print('Rectified')
        pop, exps, cards = psro_rectified_steps(iters=iters, num_pseudo_learners=num_threads, payoffs=payoffs,
                                                seed=seed + 1,
                                                lr=lr, threshold=thresh)
        rectified_exps = exps
        rectified_cardinality = cards
    if self_play:
        print('Self-play')
        pop, exps, cards = self_play_steps(iters=iters, payoffs=payoffs, improvement_pct_threshold=thresh, lr=lr,
                                           seed=seed + 1)
        self_play_exps = exps
        self_play_cardinality = cards
    if distance_psro:
        print('Distance PSRO')
        pop, exps, cards, pe = psro_steps(iters=iters, num_learners=num_threads, seed=seed + 1,
                                      improvement_pct_threshold=thresh, lr=lr,
                                      payoffs=payoffs, loss_func='distance')
        distance_psro_exps = exps
        distance_psro_cardinality = cards
        distance_psro_pe = pe
    if rectified_distance_psro:
        print('Rectified Distance PSRO')
        pop, exps, cards, pe = psro_steps(iters=iters, num_learners=num_threads, seed=seed + 1,
                                      improvement_pct_threshold=thresh, lr=lr,
                                      payoffs=payoffs, loss_func='rectified_distance')
        rectified_distance_psro_exps = exps
        rectified_distance_psro_cardinality = cards
    if diverge_psro:
        print('Diverge PSRO')
        pop, exps, cards, pe = psro_steps(iters=iters, num_learners=num_threads, seed=seed + 1,
                                      improvement_pct_threshold=thresh, lr=lr,
                                      payoffs=payoffs, loss_func='diverge_psro')
        diverge_psro_exps = exps
        diverge_psro_cardinality = cards
        diverge_psro_pe = pe

    return {
        'psro_exps': psro_exps,
        'psro_cardinality': psro_cardinality,
        'pipeline_psro_exps': pipeline_psro_exps,
        'pipeline_psro_cardinality': pipeline_psro_cardinality,
        'dpp_psro_exps': dpp_psro_exps,
        'dpp_psro_cardinality': dpp_psro_cardinality,
        'rectified_exps': rectified_exps,
        'rectified_cardinality': rectified_cardinality,
        'self_play_exps': self_play_exps,
        'self_play_cardinality': self_play_cardinality,
        'distance_psro_exps': distance_psro_exps,
        'distance_psro_cardinality': distance_psro_cardinality,
        'rectified_distance_psro_exps': rectified_distance_psro_exps,
        'rectified_distance_psro_cardinality': rectified_distance_psro_cardinality,
        'diverge_psro_exps': diverge_psro_exps,
        'diverge_psro_cardinality': diverge_psro_cardinality,
        "distance_psro_pe": distance_psro_pe,
        "diverge_psro_pe": diverge_psro_pe
    }


def run_experiments(num_experiments=2, iters=40, num_threads=20, lr=0.6, thresh=0.001, logscale=True,
                    psro=False,
                    pipeline_psro=False,
                    rectified=False,
                    self_play=False,
                    dpp_psro=False,
                    distance_psro=False,
                    rectified_distance_psro=False,
                    diverge_psro=False):
    params = {
        'num_experiments': num_experiments,
        'iters': iters,
        'num_threads': num_threads,
        'lr': lr,
        'thresh': thresh,
        'psro': psro,
        'pipeline_psro': pipeline_psro,
        'dpp_psro': dpp_psro,
        'rectified': rectified,
        'self_play': self_play,
        'distance_psro': distance_psro,
        'rectified_distance_psro': rectified_distance_psro,
        'diverge_psro': diverge_psro
    }

    psro_exps = []
    psro_cardinality = []
    pipeline_psro_exps = []
    pipeline_psro_cardinality = []
    dpp_psro_exps = []
    dpp_psro_cardinality = []
    rectified_exps = []
    rectified_cardinality = []
    self_play_exps = []
    self_play_cardinality = []
    distance_psro_exps = []
    distance_psro_cardinality = []
    rectified_distance_psro_exps = []
    rectified_distance_psro_cardinality = []
    diverge_psro_exps = []
    diverge_psro_cardinality = []

    distance_psro_pe = []
    diverge_psro_pe = []

    with open(os.path.join(PATH_RESULTS, 'params.json'), 'w', encoding='utf-8') as json_file:
        json.dump(params, json_file, indent=4)

    result = []

    # print(args.mp)
    if args.mp == False:
        for i in range(num_experiments):
            result.append(run_experiment((params, i)))

    else:
        pool = mp.Pool()
        result = pool.map(run_experiment, [(params, i) for i in range(num_experiments)])

    for r in result:
        psro_exps.append(r['psro_exps'])
        psro_cardinality.append(r['psro_cardinality'])
        pipeline_psro_exps.append(r['pipeline_psro_exps'])
        pipeline_psro_cardinality.append(r['pipeline_psro_cardinality'])
        dpp_psro_exps.append(r['dpp_psro_exps'])
        dpp_psro_cardinality.append(r['dpp_psro_cardinality'])
        rectified_exps.append(r['rectified_exps'])
        rectified_cardinality.append(r['rectified_cardinality'])
        self_play_exps.append(r['self_play_exps'])
        self_play_cardinality.append(r['self_play_cardinality'])
        distance_psro_exps.append(r['distance_psro_exps'])
        distance_psro_cardinality.append(r['distance_psro_cardinality'])
        rectified_distance_psro_exps.append(r['rectified_distance_psro_exps'])
        rectified_distance_psro_cardinality.append(r['rectified_distance_psro_cardinality'])
        diverge_psro_exps.append(r['diverge_psro_exps'])
        diverge_psro_cardinality.append(r['diverge_psro_cardinality'])
        distance_psro_pe.append(r["distance_psro_pe"])
        diverge_psro_pe.append(r["diverge_psro_pe"])

    d = {
        'psro_exps': psro_exps,
        'psro_cardinality': psro_cardinality,
        'pipeline_psro_exps': pipeline_psro_exps,
        'pipeline_psro_cardinality': pipeline_psro_cardinality,
        'dpp_psro_exps': dpp_psro_exps,
        'dpp_psro_cardinality': dpp_psro_cardinality,
        'rectified_exps': rectified_exps,
        'rectified_cardinality': rectified_cardinality,
        'self_play_exps': self_play_exps,
        'self_play_cardinality': self_play_cardinality,
        'distance_psro_exps': distance_psro_exps,
        'distance_psro_cardinality': distance_psro_cardinality,
        'rectified_distance_psro_exps': rectified_distance_psro_exps,
        'rectified_distance_psro_cardinality': rectified_distance_psro_cardinality,
        'diverge_psro_exps': diverge_psro_exps,
        'diverge_psro_cardinality': diverge_psro_cardinality,
        'distance_psro_pe': distance_psro_pe,
        'diverge_psro_pe': diverge_psro_pe
    }
    pickle.dump(d, open(os.path.join(PATH_RESULTS, 'data.p'), 'wb'))

    def plot_error(data, label=''):
        min_len = min([len(i) for i in data])
        data = [i[0:min_len] for i in data]
        data_mean = np.mean(np.array(data), axis=0)
        error_bars = stats.sem(np.array(data))
        plt.plot(data_mean, label=label)
        plt.fill_between([i for i in range(data_mean.size)],
                         np.squeeze(data_mean - error_bars),
                         np.squeeze(data_mean + error_bars), alpha=alpha)

    alpha = .4
    for j in range(3):
        fig_handle = plt.figure()

        if psro:
            if j == 0:
                plot_error(psro_exps, label='PSRO')
            elif j == 1:
                plot_error(psro_cardinality, label='PSRO')
            elif j == 2:
                plot_error(psro_exps, label='PSRO')
        if pipeline_psro:
            if j == 0:
                plot_error(pipeline_psro_exps, label='P-PSRO')
            elif j == 1:
                plot_error(pipeline_psro_cardinality, label='P-PSRO')
            elif j == 2:
                plot_error(pipeline_psro_exps, label='P-PSRO')
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
            elif j == 2:
                length = min([len(l) for l in rectified_exps])
                for i, l in enumerate(rectified_exps):
                    rectified_exps[i] = rectified_exps[i][:length]
                plot_error(rectified_exps, label='PSRO-rN')
        if self_play:
            if j == 0:
                plot_error(self_play_exps, label='Self-play')
            elif j == 1:
                plot_error(self_play_cardinality, label='Self-play')
            elif j == 2:
                plot_error(self_play_exps, label='Self-play')
        if dpp_psro:
            if j == 0:
                plot_error(dpp_psro_exps, label='Ours')
            elif j == 1:
                plot_error(dpp_psro_cardinality, label='Ours')
            elif j == 2:
                plot_error(dpp_psro_exps, label='Ours')
        if distance_psro:
            if j == 0:
                plot_error(distance_psro_exps, label='distance_psro')
            elif j == 1:
                plot_error(distance_psro_cardinality, label='distance_psro')
            elif j == 2:
                plot_error(distance_psro_exps, label='distance_psro')
        if rectified_distance_psro:
            if j == 0:
                plot_error(rectified_distance_psro_exps, label='rectified_distance_psro')
            elif j == 1:
                plot_error(rectified_distance_psro_cardinality, label='rectified_distance_psro')
            elif j == 2:
                plot_error(rectified_distance_psro_exps, label='rectified_distance_psro')
        if diverge_psro:
            if j == 0:
                plot_error(diverge_psro_exps, label='diverge_psro')
            elif j == 1:
                plot_error(diverge_psro_cardinality, label='diverge_psro')
            elif j == 2:
                plot_error(diverge_psro_exps, label='diverge_psro')

        plt.legend(loc="upper left")
        plt.title(args.game_name)

        if logscale and (j == 0):
            plt.yscale('log')

        if j == 0:
            string = 'Exploitability Log'
        elif j == 1:
            string = 'Cardinality'
        elif j == 2:
            string = 'Exploitability Standard'

        plt.savefig(os.path.join(PATH_RESULTS, 'figure_' + string + '.pdf'))


if __name__ == "__main__":
    start_time = time.time()
    run_experiments(num_experiments=args.nb_exps, num_threads=2, iters=args.nb_iters, lr=.5, thresh=TH,
                    psro=False,
                    pipeline_psro=False,
                    rectified=False,
                    self_play=False,
                    dpp_psro=False,
                    distance_psro=False,
                    rectified_distance_psro=False,
                    diverge_psro=True)
    end_time = time.time()
    # print('Total time for {}'.format(args.nb_exps) + ' experiments was {}'.format(end_time - start_time) + ' seconds when multiprocessing was: {}'.format(args.mp))
    print(f'The directory is {PATH_RESULTS}')
