import json
from pathlib import Path

import pytest

from loan_risk.config import TrainConfig
from loan_risk.data import synthetic_data
from loan_risk.training import train


@pytest.fixture(scope="session")
def dataset():
    return synthetic_data(1800, 42)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, dataset):
    output = tmp_path_factory.mktemp("runs") / "smoke"
    result = train(
        dataset,
        output,
        TrainConfig(trees=20, search_iterations=1, folds=2, bootstrap_iterations=10),
    )
    return output, result


@pytest.fixture
def request_record():
    return json.loads((Path(__file__).parents[1] / "configs/request.json").read_text())
