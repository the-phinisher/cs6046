import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2.colors)

deltas = np.linspace(0, 1, 11)
n = 1000
n_iters = 100
m = np.array([25, 50, 75, 100])
var = 1


def ucb_regret(u1, u2, n, var):
    rewards_1 = np.random.normal(u1, np.sqrt(var), n)
    rewards_2 = np.random.normal(u2, np.sqrt(var), n)

    T_i = [1, 1]

    def alpha(t, i):
        val = 4 * np.log(n) / T_i[i]
        return np.sqrt(val)

    ucb_1 = rewards_1[0] + alpha(1, 0)
    ucb_2 = rewards_2[1] + alpha(1, 1)
    rew_1 = [rewards_1[0]]
    rew_2 = [rewards_2[1]]

    for i in range(2, n):
        if ucb_1 > ucb_2:
            T_i[0] += 1
            rew_1 += [rewards_1[i]]
            ucb_1 = np.array(rew_1).mean() + alpha(i, 0)
        else:
            T_i[1] += 1
            rew_2 += [rewards_2[i]]
            ucb_2 = np.array(rew_2).mean() + alpha(i, 1)

    return -(np.sum(rew_1) + np.sum(rew_2))


def etc_regret(u1, u2, n, m, var):
    rewards_1 = np.random.normal(u1, np.sqrt(var), n)
    rewards_2 = np.random.normal(u2, np.sqrt(var), n)

    empirical_mean_1 = np.mean(rewards_1[:m])
    empirical_mean_2 = np.mean(rewards_2[:m])

    if empirical_mean_1 >= empirical_mean_2:
        return -(
            np.sum(rewards_1[:m]) + np.sum(rewards_2[:m]) + np.sum(rewards_1[2 * m :])
        )
    else:
        return -(
            np.sum(rewards_1[:m]) + np.sum(rewards_2[:m]) + np.sum(rewards_2[2 * m :])
        )


regrets_theory = []
regrets_sim = []
regrets_optimal_m_sim = []


def optimal_m(delta, n):
    if delta <= 0:
        return 1
    return max(1, int(np.ceil((4 / delta**2) * np.log(n * delta**2 / 4))))


for delta in deltas:
    prob = 1 - norm.cdf(delta / np.sqrt(2 * var / m))
    regret_theory = delta * (m + (n - 2 * m) * prob)
    regrets_theory.append(regret_theory)

    u1 = 0
    u2 = -delta
    regret_sim = np.array(
        [[etc_regret(u1, u2, n, mi, var) for mi in m] for _ in range(n_iters)]
    ).mean(axis=0)
    regrets_sim.append(regret_sim)

    u1 = 0
    u2 = -delta
    mx = optimal_m(delta, n)
    regret_optimal_m_sim = np.array(
        [etc_regret(u1, u2, n, mx, var) for _ in range(n_iters)]
    ).mean(axis=0)
    regrets_optimal_m_sim.append(regret_optimal_m_sim)

regrets_sim = np.array(regrets_sim)
regrets_theory = np.array(regrets_theory)

ucb_regrets = np.array(
    [[ucb_regret(0, -delta, n, var) for delta in deltas] for _ in range(n_iters)]
).mean(axis=0)

plt.figure(figsize=(16, 10))
for i, mi in enumerate(m):
    plt.plot(deltas, regrets_sim[:, i], label=f"ETC Regret ($m$ = {mi})", linewidth=0.75, linestyle="--")
plt.plot(deltas, regrets_optimal_m_sim, label="Optimal m Regret", linewidth=2)
plt.plot(deltas, ucb_regrets, label="UCB Regret", linewidth=2)
plt.xlabel("Δ")
plt.ylabel("Regret")
plt.legend()
plt.savefig("regret.png")
plt.show()
