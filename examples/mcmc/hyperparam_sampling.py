import matplotlib.pyplot as plt
import numpy as np

from bark.fitting.bark_sampler import BARKTrainParams, _bark_params_to_jitclass
from bark.fitting.noise_scale_proposals import (
    PROPOSAL_STEP_SIZE,
    gamma_logpdf,
    get_noise_proposal_softplus,
    half_normal_logpdf,
    propose_positive_transition_softplus,
)


def halfnormal_pdf(x, scale):
    return 2 / np.sqrt(2 * np.pi) * np.exp(half_normal_logpdf(x, scale))


def main():
    noise = 0.1
    scale = 1.0

    N_SAMPLES = 10_000

    noise_samples = np.zeros(N_SAMPLES)
    scale_samples = np.zeros(N_SAMPLES)
    accepted_count = 0
    params = _bark_params_to_jitclass(BARKTrainParams())
    for i in range(N_SAMPLES):
        new_noise, log_q_prior = get_noise_proposal_softplus(noise, params)
        if np.log(np.random.rand()) <= log_q_prior:
            noise = new_noise
            accepted_count += 1

        noise_samples[i] = noise
        scale_samples[i] = scale

    print(f"{accepted_count/N_SAMPLES:.2f} acceptance rate")

    fig, axs = plt.subplots(nrows=2, figsize=(5, 5))
    axs[0].plot(noise_samples)
    axs[0].set_title("Noise samples")
    axs[1].plot(scale_samples)
    axs[1].set_title("Scale samples")

    fig, axs = plt.subplots(ncols=3, figsize=(10, 5))
    axs[0].hist(noise_samples, bins=100, density=True)
    axs[1].hist(scale_samples, bins=100, density=True)
    t = np.linspace(0.1, 5, 100)
    for i in (0, 1):
        axs[i].plot(
            t,
            np.exp(gamma_logpdf(t, params.gamma_prior_shape, params.gamma_prior_rate)),
            "--",
        )
        axs[i].plot(t, halfnormal_pdf(t, 5.0))
        axs[i].set_title(["Noise", "Scale"][i])

    x = np.abs(np.random.normal(0, np.sqrt(5), 10000))
    axs[2].hist(x, bins=100, density=True)
    axs[2].plot(t, halfnormal_pdf(t, 5.0))

    # axs[0].set_xlim(0, 0.5)
    plt.show()


def check_pdf():
    noise = 0.1
    scale = 1.0

    N_SAMPLES = 50_000

    noise_samples = np.zeros(N_SAMPLES)
    scale_samples = np.zeros(N_SAMPLES)
    cur_value = np.array([noise, scale])

    for i in range(N_SAMPLES):
        new_value = propose_positive_transition_softplus(cur_value)

        noise_samples[i] = new_value[0]
        scale_samples[i] = new_value[1]

    fig, axs = plt.subplots(nrows=2, figsize=(5, 5))
    axs[0].hist(noise_samples, bins=100, density=True)
    axs[1].hist(scale_samples, bins=100, density=True)

    noise_step, scale_step = PROPOSAL_STEP_SIZE**2

    t_noise = np.linspace(0.05, 0.15, 100)
    pdf = np.exp(
        -((np.log(np.exp(t_noise) - 1) - np.log(np.exp(noise) - 1)) ** 2)
        / (2 * noise_step)
    )
    pdf /= np.sqrt(2 * np.pi * noise_step)
    pdf *= 1 / (1 - np.exp(-t_noise))
    axs[0].plot(t_noise, pdf)

    t_scale = np.linspace(0.5, 1.5, 100)
    pdf = np.exp(
        -((np.log(np.exp(t_scale) - 1) - np.log(np.exp(scale) - 1)) ** 2)
        / (2 * scale_step)
    )
    pdf /= np.sqrt(2 * np.pi * scale_step)
    pdf *= 1 / (1 - np.exp(-t_scale))
    axs[1].plot(t_scale, pdf)

    plt.show()


if __name__ == "__main__":
    # check_pdf()
    main()
