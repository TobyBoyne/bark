import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

STEP_SIZE = 0.1


def softplus(x):
    """Used to transform from unconstrained space to constrained space"""
    return np.exp(x)
    return np.log(np.exp(x) + 1)


def softplus_inv(y):
    return np.log(y)
    return np.log(np.exp(y) - 1)


def log_q_ratio_lognorm(cur_val, new_val):
    """Compute the log transition ratio for a lognormal proposal"""
    log_q_star = stats.lognorm.logpdf(cur_val, s=STEP_SIZE, scale=new_val)
    log_q = stats.lognorm.logpdf(new_val, s=STEP_SIZE, scale=cur_val)
    return log_q_star - log_q


def propose_noise_transition(cur_noise: float):
    # take a proposal in the unconstrained space
    cur_raw_noise = softplus_inv(cur_noise)
    new_raw_noise = cur_raw_noise + np.random.randn() * STEP_SIZE
    new_noise = softplus(new_raw_noise)
    return new_noise


def noise_acceptance_probability(cur_noise: float, new_noise: float):
    log_q_ratio = log_q_ratio_lognorm(cur_noise, new_noise)

    prior = stats.halfnorm(scale=1.0)
    prior_ratio = prior.logpdf(new_noise) - prior.logpdf(cur_noise)

    return min(log_q_ratio + prior_ratio, 0.0)


def step(cur_noise):
    new_noise = propose_noise_transition(cur_noise)
    log_alpha = noise_acceptance_probability(cur_noise, new_noise)
    if np.log(np.random.rand()) <= log_alpha:
        return new_noise, True
    else:
        return cur_noise, False


def run(steps=500, initial_noise=1.0):
    cur_noise = initial_noise
    accepted = []
    noises = []
    for _ in range(steps):
        cur_noise, acc = step(cur_noise)
        accepted.append(acc)
        noises.append(cur_noise)

    return noises


noises = run(10_000)
fig, axs = plt.subplots(ncols=2)
axs[0].plot(noises)
axs[1].hist(noises, density=True)
xs = np.linspace(0, 2)
axs[1].plot(xs, stats.halfnorm(scale=1).pdf(xs))
plt.show()
