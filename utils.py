import numpy as np
import torch
from matplotlib import pyplot as plt


def ema(x, alpha=0.95):
    """
    exponential moving average
    """
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i - 1] + (1 - alpha) * x[i]
    return y

# TODO title or file name with model type (heads, layers, etc)
def plot_results(
        losses: list[float],
        grad_norms: list[float],
        save_path: str,
) -> None:
    # Replace non-finite grad norms
    # Temp fix until figure out cause of bad gradients and fix root cause
    grads_copy = grad_norms.copy() # todo - don't do at beginning of norms, just plot starting at first valid
    for i, f in enumerate(torch.isfinite(torch.tensor(grads_copy))):
        if not f:
            if i == 0:
                grads_copy[0] = 0
            else:
                grads_copy[i] = grads_copy[i - 1]

    smoothed_loss = ema(losses, alpha=0.98)
    smoothed_grad = ema(grads_copy, alpha=0.98)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x_values = range(1, len(losses) + 1)

    # Left panel - Loss curve
    ax1.plot(x_values, losses, alpha=0.2, label='Raw')
    ax1.plot(x_values, smoothed_loss, alpha=0.95, label='Smoothed')
    ax1.set_ylim(0, max(smoothed_loss) * 1.2) # No need to plot outliers in raw losses - they make important part of graph flattened visually
    ax1.legend(loc='best')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.grid(True) # todo - shorter interval between y gridlines

    # Right panel - Gradient norm
    ax2.plot(x_values, grad_norms, alpha=0.2, label='Raw')
    ax2.plot(x_values, smoothed_grad, alpha=0.95, label='Smoothed')
    ax2.set_ylim(0, max(smoothed_grad) * 1.2)
    ax2.legend(loc='best')
    ax2.set_title('Gradient Norm')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Grad Norm')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
