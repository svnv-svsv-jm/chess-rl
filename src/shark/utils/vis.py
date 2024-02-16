__all__ = ["plot_metrics"]

import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_metrics(df: pd.DataFrame) -> None:
    """Cool plot.

    Args:
        df (pd.DataFrame):
            Metrics.
    """
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.plot(df["reward/train"].to_numpy())
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(df["step_count/train"].to_numpy())
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(df["reward_sum/eval"].to_numpy())
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(df["step_count/eval"].to_numpy())
    plt.title("Max step count (test)")
    plt.savefig(os.path.join("pytest_artifacts", "metrics.png"))
    # plt.show()
