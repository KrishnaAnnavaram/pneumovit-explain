import json

import numpy as np
import pytest

from pneumovit_explain.cli import main


def test_cli_end_to_end(tmp_path, capsys):
    root = tmp_path / "chest_xray"
    assert main(["synth", "--out", str(root), "--patients", "80"]) == 0
    assert (root / "metadata.csv").exists()
    assert main(["index", "--data-dir", str(root), "--out", str(tmp_path / "split.csv")]) == 0
    bundle = tmp_path / "b.joblib"
    assert main(["train", "--data-dir", str(root), "--out", str(bundle)]) == 0
    out = tmp_path / "rep"
    assert main(["evaluate", "--data-dir", str(root), "--bundle", str(bundle), "--bootstrap", "50",
                 "--out", str(out)]) == 0
    record = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert {"auc", "sensitivity", "specificity"} <= set(record["test"]["ci"])
    assert "zone_agreement" in record
    assert "NOT A DIAGNOSTIC DEVICE" in (out / "report.md").read_text(encoding="utf-8")
    img = next((root / "train" / "PNEUMONIA").glob("*.jpeg"))
    capsys.readouterr()
    assert main(["explain", "--bundle", str(bundle), "--image", str(img), "--overlay", str(tmp_path / "o.png")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["summary"]["source"] == "template" and (tmp_path / "o.png").exists()


def test_cli_errors(tmp_path, capsys):
    assert main(["index", "--data-dir", str(tmp_path / "nothing")]) == 2
    import joblib
    joblib.dump([1, 2], tmp_path / "x.joblib")
    assert main(["explain", "--bundle", str(tmp_path / "x.joblib"), "--image", "x.png"]) == 2


def test_vit_forward_rollout_and_device_fallback():
    torch = pytest.importorskip("torch")
    from pneumovit_explain.deep import ViTConfig, attention_rollout, build_vit, pick_device
    assert pick_device("cuda").type == ("cuda" if torch.cuda.is_available() else "cpu")  # Problem 7
    model = build_vit(ViTConfig(image_size=32, patch=8, dim=32, depth=2, heads=2))
    model.eval()
    out = model(torch.randn(3, 1, 32, 32))
    assert out.shape == (3,)
    roll = attention_rollout(model)
    assert roll.shape == (3, 4, 4) and np.allclose(roll.max(axis=(1, 2)), 1)


def test_vit_training_keeps_the_best_validation_epoch(split_index, tmp_path):
    # Problems 1-3: early stopping on val AUC, the best epoch is saved, and the saved file is what loads.
    pytest.importorskip("torch")
    from pneumovit_explain.dataset import load_gray
    from pneumovit_explain.deep import TorchClassifier, TrainConfig, ViTConfig
    tr, va = split_index.part("train"), split_index.part("val")
    X = [load_gray(p, 32) for p in tr["path"]]
    Xv = [load_gray(p, 32) for p in va["path"]]
    clf = TorchClassifier("vit_scratch", seed=0, device="cpu",
                          train_config=TrainConfig(epochs=3, patience=2, batch_size=32, image_size=32),
                          vit_config=ViTConfig(image_size=32, patch=8, dim=32, depth=2, heads=2))
    ckpt = tmp_path / "vit.pt"
    clf.fit(X, tr["y"].to_numpy(), Xv, va["y"].to_numpy(), checkpoint=ckpt)
    hist = clf.log_.history
    assert clf.log_.best_val_auc == max(h["val_auc"] for h in hist)
    assert hist[0]["lr"] < 3e-4  # the cosine schedule moves every epoch
    loaded = TorchClassifier.load(ckpt, device="cpu")
    assert np.allclose(loaded.logits(Xv[:4]), clf.logits(Xv[:4]), atol=1e-5)
    assert clf.rollout(Xv[0]).shape == (4, 4)
    import pickle
    again = pickle.loads(pickle.dumps(clf))
    assert np.allclose(again.logits(Xv[:4]), clf.logits(Xv[:4]), atol=1e-5)
