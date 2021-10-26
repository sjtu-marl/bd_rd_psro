import numpy as np




def get_br_to_strat(strat, payoffs=None, verbose=False):
    row_weighted_payouts = strat @ payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br


def js_divergence(n, target_dist):
    def entropy(p_k):
        p_k = p_k + 1e-8
        p_k = p_k / sum(p_k)
        return -(p_k * np.log(p_k)).sum()

    original_dist = np.zeros(shape=target_dist.shape)
    original_dist[n] = 1
    return 2 * entropy(original_dist + target_dist) - entropy(original_dist) - entropy(target_dist)

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

def pop_effective_diversity(pop, payoff, iters):
    emp_game_matrix = pop @ payoff
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
    print(f"Nash is {row_avg}")
    return -row_avg @ emp_game_matrix @ column_avg.T


def generate_group_transitive_game(n, m):
    group_transitive_game = np.random.random((m * n, m * n))
    group_transitive_game = np.triu(group_transitive_game)
    group_transitive_game = (group_transitive_game - group_transitive_game.T)
    group_transitive_game = group_transitive_game
    sub_games = generate_non_transitive_game(n)
    for i in range(m):
        group_transitive_game[i * n:(i + 1) * n, i * n:(i + 1) * n] = sub_games
    return group_transitive_game / np.abs(group_transitive_game).max()


def generate_non_transitive_game(n):
    a = np.random.random((n, n))
    b = np.zeros((n, n))
    a = a - a.T
    for i in range(n):
        for j in range(n):
            b[i, j] = sum([a[i, j] + a[j, k] - a[i, k] for k in range(n)]) / n
    return b


if __name__ == '__main__':
    pass
