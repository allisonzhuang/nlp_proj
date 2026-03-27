"""Plot BLEU-over-steps training curves from eval checkpoint data."""

import json
import matplotlib.pyplot as plt

curves = {
    "IoFT": "./checkpoints_v1/ioft/training_curve.json",
    "CoTFT (CoT)": "./checkpoints_v1/cotft_cot/training_curve.json",
    "CoTFT (MAPS)": "./checkpoints_v1/cotft_maps/training_curve.json",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for label, path in curves.items():
    with open(path) as f:
        data = json.load(f)
    results = data["results"]
    # Deduplicate by step (final checkpoint may duplicate last step)
    seen = {}
    for r in results:
        seen[r["step"]] = r
    results = sorted(seen.values(), key=lambda x: x["step"])

    steps = [r["step"] for r in results]
    bleus = [r["bleu"] for r in results]
    chrfs = [r["chrf"] for r in results]

    ax1.plot(steps, bleus, label=label, linewidth=2, marker='o', markersize=4)
    ax2.plot(steps, chrfs, label=label, linewidth=2, marker='o', markersize=4)

ax1.set_xlabel("Training Steps", fontsize=13)
ax1.set_ylabel("BLEU", fontsize=13)
ax1.set_title("BLEU over Training Steps", fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 5200)

ax2.set_xlabel("Training Steps", fontsize=13)
ax2.set_ylabel("chrF++", fontsize=13)
ax2.set_title("chrF++ over Training Steps", fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 5200)

plt.tight_layout()
plt.savefig("bleu_curves.png", dpi=150)
print("Saved to bleu_curves.png")
