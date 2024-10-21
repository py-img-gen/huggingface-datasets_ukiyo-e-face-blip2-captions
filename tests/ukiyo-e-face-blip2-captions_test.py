import os

import datasets as ds
import pytest


@pytest.fixture
def org_name() -> str:
    return "py-img-gen"


@pytest.fixture
def dataset_name() -> str:
    return "ukiyo-e-face-blip2-captions"


@pytest.fixture
def dataset_path(dataset_name: str) -> str:
    return f"{dataset_name}.py"


@pytest.fixture
def repo_id(org_name: str, dataset_name: str) -> str:
    return f"{org_name}/{dataset_name}"


@pytest.fixture
def ukiyo_e_face_dataset_path() -> str:
    return "./ukiyoe-1024-v2.tar"


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
def test_load_dataset(
    dataset_path: str,
    ukiyo_e_face_dataset_path: str,
    repo_id: str,
    num_images: int = 5209,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        data_dir=ukiyo_e_face_dataset_path,
        trust_remote_code=True,
    )
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["train"].num_rows == num_images

    dataset.push_to_hub(repo_id=repo_id, private=True)
