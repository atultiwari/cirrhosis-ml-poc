import importlib

import pytest

from cirrhosis_ml import config, io

SUBPACKAGES = ["survival", "models", "evaluation", "pbc", "mimic", "eicu", "reporting"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name):
    assert importlib.import_module(f"cirrhosis_ml.{name}") is not None


def test_configs_load():
    assert "cirrhosis" in config.load_yaml("icd_codes.yaml")
    assert "variables" in config.load_yaml("canonical_schema.yaml")


def test_restricted_guard_blocks(monkeypatch):
    monkeypatch.delenv(io.ALLOW_RESTRICTED_ENV, raising=False)
    target = config.RESTRICTED_DATA_DIR / "mimic_iv" / "anything.csv"
    with pytest.raises(io.RestrictedDataError):
        io.assert_readable(target)


def test_open_data_allowed():
    assert io.assert_readable(config.PBC_CACHE_DIR / "x.csv").name == "x.csv"


def test_data_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert config.data_root() == tmp_path.resolve()
