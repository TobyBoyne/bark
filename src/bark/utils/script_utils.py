from os import PathLike
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class BenchmarkConfig(BaseModel):
    benchmark: str
    benchmark_save_name: str | None = None
    benchmark_params: dict[str, Any] = Field(default_factory=dict)
    num_train: int
    num_test: int

    def get_save_name(self) -> str:
        return self.benchmark_save_name or self.benchmark


class ModelConfig(BaseModel):
    model: str
    model_save_name: str | None = None
    model_params: dict[str, Any] = Field(default_factory=dict)

    def get_save_name(self) -> str:
        return self.model_save_name or self.model


def read_benchmark_config(path_to_benchmark_config: PathLike) -> BenchmarkConfig:
    with open(path_to_benchmark_config, "r") as f:
        cfg = yaml.safe_load(f)
        return BenchmarkConfig.model_validate(cfg)


def read_model_config(path_to_model_config: PathLike) -> ModelConfig:
    with open(path_to_model_config, "r") as f:
        cfg = yaml.safe_load(f)
        return ModelConfig.model_validate(cfg)


def get_output_path(
    output_dir: PathLike,
    benchmark_config: BenchmarkConfig,
    model_config: ModelConfig,
    write_config: bool = True,
) -> Path:
    result_output_dir = (
        Path(output_dir)
        / benchmark_config.get_save_name()
        / model_config.get_save_name()
    )
    result_output_dir.mkdir(parents=True, exist_ok=True)

    if write_config:
        yaml.dump(
            {**benchmark_config.model_dump(), **model_config.model_dump()},
            open(result_output_dir / "config.yaml", "w"),
        )

    return result_output_dir
