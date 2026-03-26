"""Plot training loss curves from Slurm log files."""

import re
import matplotlib.pyplot as plt

def extract_losses(log_path):
    """Extract (step, loss) pairs from a training log."""
    losses = []
    step = 0
    with open(log_path) as f:
        for line in f:
            match = re.search(r"'loss': ([\d.]+)", line)
            if match:
                step += 100  # logging_steps=100
                losses.append((step, float(match.group(1))))
    return losses

logs = {
    "IoFT": "logs/ft_ioft_5205.log",
    "CoTFT (CoT)": "logs/ft_cotft_cot_5206.log",
    "CoTFT (MAPS)": "logs/ft_cotft_maps_5207.log",
}

fig, ax = plt.subplots(figsize=(10, 6))

for label, path in logs.items():
    try:
        losses = extract_losses(path)
        if losses:
            steps, vals = zip(*losses)
            ax.plot(steps, vals, label=label, linewidth=2)
    except FileNotFoundError:
        print(f"Skipping {label}: log not found")

ax.set_xlabel("Training Steps", fontsize=13)
ax.set_ylabel("Loss", fontsize=13)
ax.set_title("Fine-Tuning Loss Curves: IoFT vs CoTFT", fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150)
print("Saved to loss_curves.png")
