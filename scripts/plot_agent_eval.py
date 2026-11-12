"""Render the Error Analyst output (reports/agent_loop_result.json) as a figure."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/agent_loop_result.json")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/screenshots/07_agent_evaluation.png")
r = json.loads(src.read_text())
ea = r["error_analysis"]
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.1, 1, 1.6]})

steps = [s for s in r["steps"] if s["node"] == "ml_engineer"]
a1.plot(range(1, len(steps) + 1), [s["f1"] for s in steps], marker="o")
for i, s in enumerate(steps, 1):
    a1.annotate(s["model_family"].replace("_", " "), (i, s["f1"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
a1.axhline(0.9, ls="--", c="gray", lw=1, label="target 0.90")
a1.set(title="Validation F1 per agent cycle", xlabel="cycle", ylabel="F1", ylim=(0.75, 1.0), xticks=range(1, len(steps) + 1))
a1.legend(fontsize=8)

cm = ea["confusion_matrix"]["matrix"]
a2.imshow(cm, cmap="Greens")
for i in range(2):
    for j in range(2):
        a2.text(j, i, cm[i][j], ha="center", va="center", fontsize=14)
a2.set(title=f"Holdout confusion matrix (n={ea['n_holdout']})", xticks=[0, 1], yticks=[0, 1],
       xticklabels=["pred <75", "pred ≥75"], yticklabels=["true <75", "true ≥75"])

sl = sorted(ea["slices"], key=lambda s: s["accuracy"])
a3.barh([s["region"][:28] for s in sl], [s["accuracy"] for s in sl])
for k, s in enumerate(sl):
    a3.text(s["accuracy"] + 0.005, k, f"{s['accuracy']:.2f} (n={s['n']})", va="center", fontsize=8)
a3.set(title="Holdout accuracy by region (Error Analyst)", xlim=(0.8, 1.06))
fig.suptitle(
    f"Best model: {r['best_spec']['model_family']} {r['best_spec']['hyperparameters']} · holdout "
    f"F1 {ea['f1']:.3f} · accuracy {ea['accuracy']:.3f} · precision {ea['precision']:.3f} · "
    f"recall {ea['recall']:.3f} · ROC AUC {ea.get('roc_auc', float('nan')):.3f}",
    fontsize=10,
)
fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=140)
print(out)
