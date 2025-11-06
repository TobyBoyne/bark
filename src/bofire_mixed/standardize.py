from jaxtyping import Array, Float


class Standardize:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def __call__(self, y: Float[Array, "N 1"], train: bool) -> Float[Array, "N 1"]:
        if train:
            self.mean = y.mean().item()
            self.std = max(y.std().item(), 1e-6)
        return (y - self.mean) / self.std

    def untransform(self, y: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
        return y * self.std + self.mean

    def untransform_mu_var(
        self, mu: Float[Array, "*batch N"], var: Float[Array, "*batch N"]
    ) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N"]]:
        return self.untransform(mu), var * self.std**2
