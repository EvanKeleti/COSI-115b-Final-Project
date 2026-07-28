import numpy as np
import torch
from matplotlib import pyplot as plt


def ema(x, alpha=0.95):
    """
    exponential moving average
    """
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)): # todo improve - use blur missing nan values
        y[i] = alpha * y[i - 1] + (1 - alpha) * x[i]
    return y

def make_filled_copy(x: list[float]):
    x = x.copy()
    for i, f in enumerate(torch.isfinite(torch.tensor(x))):
        if not f:
            if i == 0:
                x[0] = 0
            else:
                x[i] = x[i - 1]
    return x

# TODO title or file name with model type (heads, layers, etc)
def plot_results(
        checkpoint,
        save_path: str,
) -> None:
    # Replace non-finite grad norms
    # Temp fix until figure out cause of bad gradients and fix root cause
    losses = make_filled_copy(checkpoint["losses"])
    grad_norms = make_filled_copy(checkpoint["grad_norms"]) # todo - don't do at beginning of norms, just plot starting at first valid
    accuracies = make_filled_copy(checkpoint["accuracies"])

    smoothed_loss = ema(losses, alpha=0.98)
    smoothed_grad = ema(grad_norms, alpha=0.98)
    smoothed_acc = ema(accuracies, alpha=0.98)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle("Training Stats Log")

    x_values = range(1, len(losses) + 1)

    # Left panel - Loss curve
    ax1.plot(x_values, losses, alpha=0.2, label='Raw')
    ax1.plot(x_values, smoothed_loss, alpha=0.95, label='Smoothed')
    ax1.set_ylim(0, max(smoothed_loss) * 1.2) # No need to plot outliers in raw losses - they make important part of graph flattened visually
    ax1.legend(loc='best')
    ax1.set_title('Loss')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.grid(True) # todo - shorter interval between y gridlines

    # Mid panel - Accuracy
    ax2.plot(x_values, accuracies, alpha=0.2, label='Raw')
    ax2.plot(x_values, smoothed_acc, alpha=0.95, label='Smoothed')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='best')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    # Right panel - Gradient norm
    ax3.plot(x_values, grad_norms, alpha=0.2, label='Raw')
    ax3.plot(x_values, smoothed_grad, alpha=0.95, label='Smoothed')
    ax3.set_ylim(0, max(smoothed_grad) * 1.2)
    ax3.legend(loc='best')
    ax3.set_title('Gradient Norm')
    ax3.set_xlabel('Batch')
    ax3.set_ylabel('Grad Norm')
    ax3.grid(True)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(save_path)
    plt.close()
