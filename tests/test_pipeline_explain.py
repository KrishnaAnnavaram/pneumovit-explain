import json

import numpy as np
import pytest

from pneumovit_explain import BANNER, pipeline
from pneumovit_explain.dataset import load_gray
from pneumovit_explain.explain import knn, narrative, saliency
from pneumovit_explain.models import LightClassifier, xray_features


@pytest.fixture(scope="module")
def trained(split_index):
    from pneumovit_explain.config import Settings
    settings = Settings(seed=1, n_bootstrap=100, image_size=64)
    return pipeline.train_and_select(split_index, LightClassifier(seed=1), settings)


def test_selection_never_reads_the_test_part(split_index, monkeypatch):
    # Problem 1: model selection, calibration and the threshold use train and val only.
    from pneumovit_explain.config import Settings
    seen = []
    real = pipeline.load_gray

    def spy(path, size=None):
        seen.append(path)
        return real(path, size)

    monkeypatch.setattr(pipeline, "load_gray", spy)
    pipeline.train_and_select(split_index, LightClassifier(seed=1), Settings(seed=1, image_size=64))
    test_paths = set(split_index.part("test")["path"])
    assert seen and test_paths.isdisjoint(seen)


def test_bundle_values(trained):
    bundle, info = trained
    assert 0 < bundle.temperature < 20 and 0 < bundle.threshold < 1
    assert bundle.val_summary["sensitivity"] >= 0.95
    assert info["selection"] and all("val_auc" in r for r in info["selection"])


def test_light_model_beats_chance_on_test(trained, split_index, settings):
    bundle, _ = trained
    res = pipeline.evaluate_test(bundle, split_index, settings)
    assert res["at_operating_threshold"]["auc"] > 0.7
    assert {"sensitivity", "specificity", "ppv", "npv", "brier", "ece"} <= set(res["at_operating_threshold"])


def test_saved_bundle_is_the_selected_model(trained, tmp_path, split_index):
    # Problem 2: the served model is the same object that was selected on val.
    bundle, _ = trained
    path = pipeline.save_bundle(bundle, tmp_path / "b.joblib")
    loaded = pipeline.load_bundle(path)
    imgs = [load_gray(p, 64) for p in split_index.part("val")["path"][:5]]
    assert np.allclose(loaded.probability(imgs), bundle.probability(imgs))
    assert loaded.threshold == bundle.threshold


def test_neighbours_come_from_training_images_of_both_classes(trained, split_index):
    # Problem 5: the neighbour search is not limited to the predicted class or to test images.
    bundle, _ = trained
    train_files = set(split_index.part("train")["file"])
    assert set(bundle.index.files) == train_files
    labels = set()
    for p in split_index.part("test")["path"][:15]:
        img = load_gray(p, 64)
        labels |= {nb.label for nb in bundle.index.query(bundle.classifier.embed([img])[0], 10)}
    assert labels == {"NORMAL", "PNEUMONIA"}


def test_analyze_runs_in_memory_and_has_all_parts(trained, split_index, tmp_path, monkeypatch):
    # Problems 7 and 8: CPU only, no file writes per request, no crash on an odd input shape.
    bundle, _ = trained
    monkeypatch.chdir(tmp_path)
    img = load_gray(split_index.part("test")["path"].iloc[0])  # full size, not resized
    rgb = np.stack([img * 255] * 3, axis=-1)
    result = bundle.analyze(rgb, k=5)
    assert list(tmp_path.iterdir()) == []
    assert 0 <= result["probability"] <= 1 and result["neighbours_summary"]["k"] == 5
    assert result["heat"].shape == (64, 64) and result["saliency_zone"] in saliency.zone_scores(result["heat"])
    assert result["summary"]["text"].endswith(BANNER)


def test_index_save_and_load(tmp_path):
    rng = np.random.default_rng(0)
    idx = knn.TrainIndex(rng.normal(size=(20, 4)), rng.integers(0, 2, 20), [f"f{i}" for i in range(20)])
    loaded = knn.TrainIndex.load(idx.save(tmp_path / "i.npz"))
    q = rng.normal(size=4)
    assert [n.file for n in idx.query(q, 3)] == [n.file for n in loaded.query(q, 3)]
    assert idx.query(idx.vectors[7], 1)[0].file == "f7"


def _fake_logits(images):
    """Logit rises with the brightness of a bright square in the image centre."""
    return np.array([im[28:36, 28:36].mean() * 10 for im in images])


def test_occlusion_finds_the_region_that_drives_the_score():
    img = np.full((64, 64), 0.3, dtype=np.float32)
    img[28:36, 28:36] = 0.9
    heat = saliency.occlusion(_fake_logits, img, patch=8, stride=4)
    assert heat[30, 30] > 0 and abs(heat[5, 5]) < 1e-6


def test_randomisation_check_and_overlay(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.random((16, 16))
    assert saliency.randomisation_check(a, a) == pytest.approx(1.0)
    assert abs(saliency.randomisation_check(a, rng.random((16, 16)))) < 0.3
    img = np.zeros((16, 16))
    out = saliency.save_overlay(img, a, tmp_path / "o.png")
    from PIL import Image
    arr = np.asarray(Image.open(out))
    assert arr[0, 0].tolist() == [0, 0, 0]  # Problem 10: the raw image is shown without re-normalisation


def test_sanity_check_shows_low_correlation(trained, split_index, settings):
    res = pipeline.sanity_check(trained[0], split_index, settings, n_images=3)
    assert res["images"] == 3 and res["mean_spearman"] < 0.6


def _result(p=0.8, threshold=0.3, pos=7, k=10):
    return {"probability": p, "threshold": threshold, "neighbours_summary": {"k": k, "pneumonia": pos,
            "normal": k - pos}, "saliency_zone": "right lower", "saliency_zone_share": 0.4}


def test_the_llm_never_receives_the_predicted_label():
    # Problem 5: the LLM gets facts, not "explain why it is PNEUMONIA".
    prompts = []

    class Recorder:
        name = "rec"

        def generate(self, prompt, image_png=None):
            prompts.append((prompt, image_png))
            return "The probability is 0.80 and the threshold is 0.30. A clinician must review the image."

    out = narrative.summarise(_result(), Recorder(), np.zeros((8, 8)), send_image=False)
    facts_line = prompts[0][0].split("FACTS: ", 1)[1].split("\n", 1)[0]
    assert "PNEUMONIA" not in facts_line and "predicted" not in facts_line.lower()
    assert prompts[0][1] is None  # Problem 9: no image leaves the machine by default
    assert out["source"] == "rec" and out["text"].endswith(BANNER)


@pytest.mark.parametrize("text, reason", [
    ("This X-ray definitely shows pneumonia.", "certainty"),
    ("The probability is 0.99, so treat it.", "number 0.99"),
])
def test_unsafe_llm_text_falls_back_to_the_template(text, reason):
    class Bad:
        name = "bad"

        def generate(self, prompt, image_png=None):
            return text

    out = narrative.summarise(_result(), Bad())
    assert out["source"] == "template" and any(reason in r for r in out["rejected_because"])


def test_llm_failure_keeps_the_template():
    class Broken:
        name = "broken"

        def generate(self, prompt, image_png=None):
            raise ConnectionError("down")

    out = narrative.summarise(_result(), Broken())
    assert out["source"] == "template" and "ConnectionError" in out["rejected_because"][0]


def test_template_flags_disagreement():
    text = narrative.template(narrative.facts_from(_result(p=0.9, pos=2)))
    assert "do not agree" in text


def test_gemini_and_openai_adapters_with_fake_transport():
    calls = []

    def gemini_transport(url, headers, body):
        calls.append((url, headers, json.loads(body)))
        return json.dumps({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}).encode()

    g = narrative.GeminiClient("k1", "gemini-2.5-flash", transport=gemini_transport)
    assert g.generate("hi") == "ok"
    assert "gemini-2.5-flash:generateContent" in calls[0][0] and calls[0][1]["x-goog-api-key"] == "k1"

    def openai_transport(url, headers, body):
        calls.append((url, headers, json.loads(body)))
        return json.dumps({"choices": [{"message": {"content": "fine"}}]}).encode()

    o = narrative.OpenAICompatibleClient("k2", "m", "http://localhost:11434/v1", transport=openai_transport)
    assert o.generate("hi") == "fine" and calls[1][0] == "http://localhost:11434/v1/chat/completions"
    with pytest.raises(ValueError):
        narrative.GeminiClient("")


def test_features_shape():
    assert xray_features(np.random.default_rng(0).random((64, 64))).shape == (16 + 18 + 8 + 64,)
