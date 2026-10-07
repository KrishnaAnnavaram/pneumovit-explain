"""Evaluation record (JSON) and report (Markdown)."""

from __future__ import annotations

import json
from pathlib import Path

from . import BANNER, __version__


def _ci(c: dict) -> str:
    return f"{c['value']:.3f} [{c['low']:.3f}, {c['high']:.3f}]"


def render(record: dict) -> str:
    t, info, split = record["test"], record["train_info"], record["split"]
    op, half = t["at_operating_threshold"], t["at_0_5"]
    lines = [
        "# pneumovit-explain evaluation report", "", f"> {BANNER}", "",
        f"- Model: `{record['model']}`, pneumovit-explain {__version__}, seed {record['seed']}",
        f"- Patient-aware split: train {split['train']['images']} images / {split['train']['patients']} patients, "
        f"val {split['val']['images']} / {split['val']['patients']}, test {split['test']['images']} / "
        f"{split['test']['patients']}",
        f"- Temperature (fit on val): {info['temperature']:.3f}. Operating threshold (val sensitivity >= "
        f"{record['target_sensitivity']}): {info['threshold']:.3f}",
        "", "## Validation (used for every choice)", "",
        f"- AUC {record['val']['auc']:.3f}, sensitivity {record['val']['sensitivity']:.3f}, "
        f"specificity {record['val']['specificity']:.3f} at the operating threshold", "",
        "## Test (one pass, 95% patient bootstrap CI)", "",
        "| Metric | Operating threshold | Threshold 0.5 |", "|---|---|---|",
        f"| AUC | {_ci(t['ci']['auc'])} | {half['auc']:.3f} |",
        f"| Sensitivity | {_ci(t['ci']['sensitivity'])} | {half['sensitivity']:.3f} |",
        f"| Specificity | {_ci(t['ci']['specificity'])} | {half['specificity']:.3f} |",
        f"| PPV | {op['ppv']:.3f} | {half['ppv']:.3f} |",
        f"| NPV | {op['npv']:.3f} | {half['npv']:.3f} |",
        f"| Accuracy | {op['accuracy']:.3f} | {half['accuracy']:.3f} |",
        f"| Brier | {op['brier']:.3f} | {half['brier']:.3f} |",
        f"| ECE | {op['ece']:.3f} | {half['ece']:.3f} |",
        "", f"Confusion at the operating threshold: {op['confusion']} (prevalence {t['prevalence']:.3f})", "",
        "| Subtype | Test images | Share flagged |", "|---|---|---|",
        *[f"| {k} | {v['images']} | {v['flagged_share']:.3f} |" for k, v in t["by_subtype"].items()],
    ]
    sc = record.get("sanity_check", {})
    if "mean_spearman" in sc:
        lines += ["", "## Saliency sanity check (model randomisation)", "",
                  f"Spearman correlation between the maps of the trained model and of a model fit on shuffled "
                  f"labels: mean {sc['mean_spearman']:.3f}, max {sc['max_spearman']:.3f} over {sc['images']} images. "
                  "Low values mean that the map depends on what the model learned."]
    za = record.get("zone_agreement")
    if za:
        lines += ["", "## Saliency zone agreement (synthetic data only)", "",
                  f"The top saliency zone equals the true finding zone in {za['agreement']:.3f} of "
                  f"{za['images']} test images with a focal finding (chance {za['chance']:.3f})."]
    lines.append("")
    return "\n".join(lines)


def write(record: dict, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"report": out / "report.md", "results": out / "results.json"}
    paths["report"].write_text(render(record), encoding="utf-8")
    paths["results"].write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
    return paths
