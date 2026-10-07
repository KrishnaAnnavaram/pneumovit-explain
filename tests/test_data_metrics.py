import numpy as np
import pytest

from pneumovit_explain import metrics
from pneumovit_explain.config import ConfigError, Settings
from pneumovit_explain.dataset import DatasetError, parse_patient, patient_split, scan


@pytest.mark.parametrize("name, label, patient, subtype", [
    ("person1_bacteria_1.jpeg", "PNEUMONIA", "P:person1", "bacteria"),
    ("person1_virus_6.jpeg", "PNEUMONIA", "P:person1", "virus"),
    ("IM-0115-0001.jpeg", "NORMAL", "N:IM-0115", "normal"),
    ("NORMAL2-IM-1427-0001.jpeg", "NORMAL", "N:NORMAL2-IM-1427", "normal"),
    ("odd_name.png", "NORMAL", "U:NORMAL:odd_name", "normal"),
])
def test_patient_ids_from_kermany_file_names(name, label, patient, subtype):
    assert parse_patient(name, label) == (patient, subtype)


def test_scan_pools_all_official_parts_and_warns(xray_root):
    idx = scan(xray_root)
    assert set(idx.frame["official_split"]) <= {"train", "val", "test"}
    assert idx.frame["patient"].str.startswith(("P:", "N:")).all()
    assert any("more than one official split" in w for w in idx.warnings)


def test_patient_split_has_no_shared_patients(split_index):
    # Problem 1: a real validation part, and no patient in two parts.
    f = split_index.frame
    assert set(f["split"]) == {"train", "val", "test"}
    assert (f.groupby("patient")["split"].nunique() == 1).all()
    for part in ("val", "test"):
        assert 0.4 < f.loc[f["split"] == part, "y"].mean() < 0.95


def test_scan_errors(tmp_path):
    with pytest.raises(DatasetError):
        scan(tmp_path / "none")
    (tmp_path / "train" / "NORMAL").mkdir(parents=True)
    (tmp_path / "train" / "NORMAL" / "bad.jpeg").write_bytes(b"no")
    with pytest.raises(DatasetError):
        scan(tmp_path)


def test_too_few_patients(xray_root, tmp_path):
    from pneumovit_explain import synthetic
    synthetic.generate(tmp_path / "small", n_patients=4, seed=1)
    with pytest.raises(DatasetError):
        patient_split(scan(tmp_path / "small"), 0.15, 0.2)


def test_settings_keep_keys_out_of_repr():
    s = Settings.from_env({"GOOGLE_API_KEY": "secret-value", "PNEUMOVIT_LLM_PROVIDER": "gemini"})
    assert s.api_keys["GOOGLE_API_KEY"] == "secret-value"
    assert "secret-value" not in repr(s)
    with pytest.raises(ConfigError):
        Settings.from_env({"PNEUMOVIT_LLM_PROVIDER": "magic"})
    with pytest.raises(ConfigError):
        Settings.from_env({"PNEUMOVIT_TARGET_SENSITIVITY": "1.5"})


def test_threshold_reaches_the_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    t = metrics.threshold_for_sensitivity(y, p, 0.95)
    sens = metrics.summary(y, p, t)["sensitivity"]
    assert sens >= 0.95
    assert metrics.summary(y, p, t + 1e-3)["sensitivity"] < sens or t == p[y == 1].max()


def test_screening_metrics_from_a_known_confusion():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.1, 0.1])
    s = metrics.summary(y, p, 0.5)
    assert s["confusion"] == {"tp": 3, "fn": 1, "tn": 3, "fp": 1}
    assert s["sensitivity"] == 0.75 and s["specificity"] == 0.75 and s["ppv"] == 0.75


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(1)
    z_true = rng.normal(0, 1.5, 4000)
    y = (rng.random(4000) < metrics.sigmoid(z_true)).astype(int)
    T = metrics.fit_temperature(z_true * 3, y)  # logits that are 3 times too large
    assert 2.4 < T < 3.6
    assert metrics.ece(y, metrics.sigmoid(z_true * 3 / T)) < metrics.ece(y, metrics.sigmoid(z_true * 3))


def test_patient_bootstrap_ci_contains_the_point():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(60), 3)
    y = np.repeat(rng.integers(0, 2, 60), 3)
    p = np.clip(y * 0.4 + rng.random(180) * 0.6, 0, 1)
    ci = metrics.bootstrap(y, p, 0.5, groups, n_boot=200, seed=0)
    for k in ("auc", "sensitivity", "specificity"):
        assert ci[k]["low"] <= ci[k]["value"] <= ci[k]["high"]
