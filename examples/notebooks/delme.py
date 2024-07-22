from jaxtyping import install_import_hook

with install_import_hook("alfalfa", "beartype.beartype"):
    from alfalfa.benchmarks import G3
    from alfalfa.bofire_utils.sampling import sample_projected


benchmark = G3()
samples = sample_projected(benchmark.domain, n=10, seed=42)
evals = benchmark.f(samples)  # noqa: F841

assert benchmark.domain.constraints.is_fulfilled(samples, tol=1e-3).all()
