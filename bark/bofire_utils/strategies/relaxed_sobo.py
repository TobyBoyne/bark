import pandas as pd
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)
from bofire.data_models.strategies.predictives.sobo import (
    SoboStrategy as SoboStrategyDataModel,
)
from bofire.strategies.predictives.sobo import SoboStrategy

from bark.benchmarks import XGBoostMNIST
from bark.bofire_utils.data_models.strategies.relaxed_sobo import (
    RelaxedSoboStrategy as RelaxedSoboStrategyDataModel,
)


def get_relaxed_domain(domain: Domain) -> Domain:
    inputs = domain.inputs.get(includes=ContinuousInput)
    for feat in domain.inputs.get(includes=DiscreteInput):
        lb, ub = feat.lower_bound, feat.upper_bound
        cont_relax = ContinuousInput(key=feat.key, bounds=(lb - 0.5, ub + 0.5))
        inputs += (cont_relax,)

    for feat in domain.inputs.get(includes=CategoricalInput):
        one_hot = [
            ContinuousInput(key=f"{feat.key}_{cat}", bounds=(0, 1))
            for cat in feat.categories
        ]
        inputs += one_hot

    return Domain(inputs=inputs, outputs=domain.outputs, constraints=domain.constraints)


class RelaxedSoboStrategy:
    def __init__(self, data_model: RelaxedSoboStrategyDataModel, **kwargs):
        self.domain = data_model.domain
        relaxed_domain = get_relaxed_domain(data_model.domain)
        sobo_dm = SoboStrategyDataModel(domain=relaxed_domain, seed=data_model.seed)
        self.sobo = SoboStrategy(sobo_dm)
        self.input_preprocessing_specs = {
            k: CategoricalEncodingEnum.ONE_HOT
            for k in data_model.domain.inputs.get_keys(includes=CategoricalInput)
        }

    def ask(self, candidate_count):
        candidates = self.sobo.ask(candidate_count)
        return self.domain.inputs.inverse_transform(
            candidates, specs=self.input_preprocessing_specs
        )

    def tell(self, experiments: pd.DataFrame):
        experiments_transformed = self.domain.inputs.transform(
            experiments, specs=self.input_preprocessing_specs
        )
        experiments_transformed[self.domain.outputs.get_keys()] = experiments[
            self.domain.outputs.get_keys()
        ]
        self.sobo.tell(experiments_transformed)

    @property
    def experiments(self):
        experiments = self.domain.inputs.inverse_transform(
            self.sobo.experiments, specs=self.input_preprocessing_specs
        )
        experiments[self.domain.outputs.get_keys()] = self.sobo.experiments[
            self.domain.outputs.get_keys()
        ]
        return experiments


if __name__ == "__main__":
    domain = XGBoostMNIST(seed=0).domain
    dm = RelaxedSoboStrategyDataModel(
        domain=domain,
        seed=0,
    )
    relaxed = RelaxedSoboStrategy(dm)
    print(relaxed.domain)
