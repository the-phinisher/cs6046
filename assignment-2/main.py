import numpy as np
import matplotlib.pyplot as plt

n = 1000
deltas = np.linspace(0, 1, 11)
num_experiments = 100


def theoretical_upper_bound(n, delta):
    if delta == 0:
        return 0
    term_1 = n * delta
    term_2 = delta + (4 / delta) * max(0, 1 + np.log(n * delta**2 / 4))
    return min(term_1, term_2)


def simulate_ETC(delta, n, m):
    if 2 * m > n:
        print("Warn: m too large, clipping to n // 2")
        m = n // 2
    mu1 = 0
    mu2 = -delta

    rewards_1 = np.random.normal(mu1, 1, n)
    rewards_2 = np.random.normal(mu2, 1, n)

    empirical_mean_1 = np.mean(rewards_1[:m])
    empirical_mean_2 = np.mean(rewards_2[:m])

    if empirical_mean_1 >= empirical_mean_2:
        return (
            np.sum(rewards_1[2 * m :]) + np.sum(rewards_2[:m]) + np.sum(rewards_1[:m])
        )
    else:
        return (
            np.sum(rewards_2[2 * m :]) + np.sum(rewards_1[:m]) + np.sum(rewards_2[:m])
        )


average_regrets = []
theoretical_bounds = []

for delta in deltas:
    rewards = []
    for _ in range(num_experiments):
        m = (
            1
            if (delta == 0)
            else max(1, int(np.ceil((4 / delta**2) * np.log(n * delta**2 / 4))))
        )
        reward = simulate_ETC(delta, n, m)
        rewards.append(reward)

    average_regrets.append(-np.mean(rewards))
    theoretical_bounds.append(theoretical_upper_bound(n, delta))


deltas_highres = np.linspace(0, 1, 101)
plt.plot(deltas, average_regrets, label="Empirical Regret")
plt.plot(deltas, theoretical_bounds, label="Theoretical Upper Bound", linestyle="--")
plt.xlabel("Δ")
plt.ylabel("Regret")
plt.legend()
plt.title("Performance of ETC Algorithm")
plt.show()
