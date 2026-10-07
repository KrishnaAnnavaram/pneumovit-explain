"""Command line interface: `pneumovit-explain <command>`."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import BANNER, __version__, report, synthetic
from .config import ConfigError, Settings
from .dataset import DatasetError, load_gray, patient_split, scan
from .explain import narrative, saliency
from .models import LightClassifier
from .pipeline import evaluate_test, load_bundle, sanity_check, save_bundle, train_and_select, zone_agreement

MODELS = ("light_logreg", "vit_scratch", "vit_small_dino", "densenet121", "convnext_tiny")


def _settings(args) -> Settings:
    return Settings.from_env().with_overrides(
        seed=getattr(args, "seed", None), image_size=getattr(args, "image_size", None),
        target_sensitivity=getattr(args, "target_sensitivity", None), n_bootstrap=getattr(args, "bootstrap", None),
        device=getattr(args, "device", None))


def _index(args, settings: Settings):
    root = args.data_dir or settings.data_dir
    if root is None:
        raise ConfigError("give --data-dir or set PNEUMOVIT_DATA_DIR (or run `pneumovit-explain demo`)")
    idx = scan(root)
    for w in idx.warnings:
        print(f"[pneumovit] warning: {w}", file=sys.stderr)
    return patient_split(idx, settings.val_fraction, settings.test_fraction, settings.seed)


def _split_counts(idx) -> dict:
    f = idx.frame
    return {s: {"images": int((f["split"] == s).sum()), "patients": int(f.loc[f["split"] == s, "patient"].nunique()),
                "pneumonia_share": float(f.loc[f["split"] == s, "y"].mean())} for s in ("train", "val", "test")}


def _classifier(name: str, settings: Settings, epochs: int | None):
    if name == "light_logreg":
        return LightClassifier(seed=settings.seed)
    from .deep import TorchClassifier, TrainConfig, ViTConfig  # extra 'deep'
    tc = TrainConfig(image_size=settings.image_size if name == "vit_scratch" else 224)
    if epochs:
        tc.epochs = epochs
    vit = ViTConfig(image_size=tc.image_size, patch=max(4, tc.image_size // 14), dim=192, depth=6, heads=6)
    return TorchClassifier(name, settings.seed, settings.device, tc, vit)


def cmd_synth(args) -> int:
    meta = synthetic.generate(args.out, args.patients, args.size, args.pneumonia_share, args.seed)
    print(f"wrote {len(meta)} synthetic images of {args.patients} patients to {args.out}")
    return 0


def cmd_index(args) -> int:
    settings = _settings(args)
    idx = _index(args, settings)
    for part, c in _split_counts(idx).items():
        print(f"{part:<6} {c['images']:>6} images {c['patients']:>5} patients  pneumonia share {c['pneumonia_share']:.3f}")
    if args.out:
        idx.frame.drop(columns=["path"]).to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


def cmd_train(args) -> int:
    settings = _settings(args)
    idx = _index(args, settings)
    clf = _classifier(args.model, settings, args.epochs)
    bundle, info = train_and_select(idx, clf, settings)
    out = Path(args.out or settings.model_dir / "bundle.joblib")
    save_bundle(bundle, out)
    print(f"val AUC {bundle.val_summary['auc']:.3f}, temperature {bundle.temperature:.3f}, "
          f"threshold {bundle.threshold:.3f} (val sensitivity {bundle.val_summary['sensitivity']:.3f})")
    print(f"saved {out}. The test part was not used.")
    return 0


def cmd_evaluate(args) -> int:
    settings = _settings(args)
    idx = _index(args, settings)
    if args.bundle:
        bundle = load_bundle(args.bundle)
        info = {"temperature": bundle.temperature, "threshold": bundle.threshold}
    else:
        bundle, info = train_and_select(idx, _classifier(args.model, settings, args.epochs), settings)
        if args.save_bundle:
            save_bundle(bundle, args.save_bundle)
    test = evaluate_test(bundle, idx, settings)
    record = {"model": bundle.classifier.name, "seed": settings.seed, "target_sensitivity": settings.target_sensitivity,
              "split": _split_counts(idx), "train_info": info, "val": bundle.val_summary, "test": test,
              "sanity_check": sanity_check(bundle, idx, settings) if args.sanity_check else {}}
    meta_csv = Path(args.data_dir or settings.data_dir) / "metadata.csv"
    if meta_csv.is_file():  # synthetic data: the true finding zones are known
        import pandas as pd
        zones = dict(pd.read_csv(meta_csv)[["file", "zone"]].itertuples(index=False, name=None))
        record["zone_agreement"] = zone_agreement(bundle, idx, zones)
    paths = report.write(record, Path(args.out) if args.out else settings.output_dir)
    ci = test["ci"]
    print(f"test AUC {ci['auc']['value']:.3f} [{ci['auc']['low']:.3f}, {ci['auc']['high']:.3f}]  "
          f"sensitivity {ci['sensitivity']['value']:.3f} [{ci['sensitivity']['low']:.3f}, {ci['sensitivity']['high']:.3f}]  "
          f"specificity {ci['specificity']['value']:.3f} [{ci['specificity']['low']:.3f}, {ci['specificity']['high']:.3f}]")
    if "zone_agreement" in record:
        za = record["zone_agreement"]
        print(f"saliency zone agreement (synthetic): {za['agreement']:.3f} over {za['images']} images (chance {za['chance']:.3f})")
    if record["sanity_check"]:
        print(f"saliency sanity check: mean Spearman {record['sanity_check']['mean_spearman']:.3f}")
    print(f"wrote {paths['report']}")
    print(BANNER)
    return 0


def cmd_explain(args) -> int:
    settings = _settings(args)
    bundle = load_bundle(args.bundle)
    img = load_gray(args.image)
    client = narrative.make_client(settings)
    result = bundle.analyze(img, settings.k_neighbours, client, settings.llm_send_image)
    if args.overlay:
        saliency.save_overlay(result["image"], result["heat"], args.overlay)
    public = {k: v for k, v in result.items() if k not in {"heat", "image"}}
    print(json.dumps(public, indent=2, default=float))
    return 0


def cmd_app(args) -> int:  # pragma: no cover - starts a server
    env = dict(os.environ, PNEUMOVIT_BUNDLE=str(args.bundle))
    script = Path(__file__).with_name("app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(script)], env=env)


def cmd_demo(args) -> int:
    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="pneumovit_demo_"))
    data = work / "chest_xray"
    synthetic.generate(data, args.patients, 64, 0.7, args.seed or 42)
    print(f"[pneumovit] synthetic images in {data}", file=sys.stderr)
    args.data_dir, args.model, args.bundle, args.epochs = str(data), "light_logreg", None, None
    args.save_bundle = str(work / "bundle.joblib")
    args.out = args.out or str(work / "report")
    args.sanity_check = True
    return cmd_evaluate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pneumovit-explain",
                                     description="Pneumonia classification with prediction-independent explanations.")
    parser.add_argument("--version", action="version", version=f"pneumovit-explain {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--data-dir")
        p.add_argument("--seed", type=int)
        p.add_argument("--image-size", type=int)
        p.add_argument("--target-sensitivity", type=float)
        p.add_argument("--bootstrap", type=int)
        p.add_argument("--device")

    p = sub.add_parser("synth", help="write synthetic X-ray-like images in the Kermany layout")
    p.add_argument("--out", default="data/synthetic/chest_xray")
    p.add_argument("--patients", type=int, default=120)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--pneumonia-share", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("index", help="scan the folder and make the patient-aware split")
    common(p)
    p.add_argument("--out", help="CSV with patient and split per file")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("train", help="fit on train, select and calibrate on val, save the bundle")
    common(p)
    p.add_argument("--model", default="light_logreg", choices=MODELS)
    p.add_argument("--epochs", type=int)
    p.add_argument("--out")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="one test pass with CIs (trains first unless --bundle is given)")
    common(p)
    p.add_argument("--model", default="light_logreg", choices=MODELS)
    p.add_argument("--epochs", type=int)
    p.add_argument("--bundle")
    p.add_argument("--save-bundle")
    p.add_argument("--sanity-check", action="store_true", help="saliency model-randomisation test")
    p.add_argument("--out")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("explain", help="probability, similar training cases, saliency and a summary for one image")
    common(p)
    p.add_argument("--bundle", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--overlay", help="PNG path for the saliency overlay")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("app", help="start the Streamlit app (extra 'app')")
    p.add_argument("--bundle", default="models/bundle.joblib")
    p.set_defaults(func=cmd_app)

    p = sub.add_parser("demo", help="offline demo: synthetic images, light model, report, bundle")
    p.add_argument("--patients", type=int, default=300)
    p.add_argument("--workdir")
    p.add_argument("--out")
    p.add_argument("--seed", type=int)
    p.add_argument("--image-size", type=int)
    p.add_argument("--target-sensitivity", type=float)
    p.add_argument("--bootstrap", type=int)
    p.add_argument("--device")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ConfigError, DatasetError, FileNotFoundError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
