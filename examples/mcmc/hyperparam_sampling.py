import matplotlib.pyplot as plt
import numpy as np

from bark.fitting.noise_scale_proposals import (
    get_noise_scale_proposal,
    half_normal_logpdf,
)


def halfnormal_pdf(x, scale):
    return 2 / np.sqrt(2 * np.pi) * np.exp(half_normal_logpdf(x, scale))


def main():
    noise = 0.1
    scale = 1.0

    N_SAMPLES = 10_000

    noise_samples = np.zeros(N_SAMPLES)
    scale_samples = np.zeros(N_SAMPLES)

    for i in range(N_SAMPLES):
        (new_noise, new_scale), log_q_prior = get_noise_scale_proposal(noise, scale)
        if np.log(np.random.rand()) <= log_q_prior:
            noise = new_noise
            scale = new_scale

        noise_samples[i] = noise
        scale_samples[i] = scale

    fig, axs = plt.subplots(nrows=2, figsize=(5, 5))
    axs[0].plot(noise_samples)
    axs[0].set_title("Noise samples")
    axs[1].plot(scale_samples)
    axs[1].set_title("Scale samples")

    fig, axs = plt.subplots(ncols=3, figsize=(10, 5))
    axs[0].hist(noise_samples, bins=100, density=True)
    axs[1].hist(scale_samples, bins=100, density=True)
    t = np.linspace(0, 5, 100)
    for i in (0, 1):
        axs[i].plot(t, halfnormal_pdf(t, 1.0))
        axs[i].plot(t, halfnormal_pdf(t, 5.0))
        axs[i].set_title(["Noise", "Scale"][i])

    x = np.abs(np.random.normal(0, np.sqrt(5), 10000))
    axs[2].hist(x, bins=100, density=True)
    axs[2].plot(t, halfnormal_pdf(t, 5.0))
    plt.show()


if __name__ == "__main__":
    main()
