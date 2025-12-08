import matplotlib.pyplot as plt
import numpy as np

def plot_note_tuning(note_stats, save_path=None):
    if not note_stats:
        print("No notes detected.")
        return

    labels = [
        f"{n['note_name']} ({n['start_time']:.2f}-{n['end_time']:.2f}s)"
        for n in note_stats
    ]
    errors = [n["mean_cents_error"] for n in note_stats]

    xs = np.arange(len(note_stats))
    plt.figure(figsize=(10, 4))
    plt.bar(xs, errors)
    plt.axhline(0, color="k", linestyle="--")
    plt.axhline(25, color="gray", linestyle=":", alpha=0.7)
    plt.axhline(-25, color="gray", linestyle=":", alpha=0.7)
    plt.ylabel("Mean cents error (sharp + / flat −)")
    plt.xticks(xs, labels, rotation=90)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()
    plt.close()
