import gpytorch as gpy

class RBFGP(gpy.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpy.means.ZeroMean()

        self.covar_module = gpy.kernels.ScaleKernel(gpy.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)