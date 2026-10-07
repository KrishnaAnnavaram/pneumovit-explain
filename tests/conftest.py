import pytest

from pneumovit_explain import synthetic
from pneumovit_explain.config import Settings
from pneumovit_explain.dataset import patient_split, scan


@pytest.fixture(scope="session")
def xray_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("xr") / "chest_xray"
    synthetic.generate(root, n_patients=90, size=64, seed=4)
    return root


@pytest.fixture(scope="session")
def split_index(xray_root):
    return patient_split(scan(xray_root), 0.15, 0.2, seed=1)


@pytest.fixture()
def settings():
    return Settings(seed=1, n_bootstrap=100, image_size=64)
