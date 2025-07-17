import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)

def simulate_gbm(S0, mu, sigma, T, dt=1/252, paths=1, seed=None, log_returns=False):
    """
    Simulates Geometric Brownian Motion (GBM) paths.
    Parameters:
        S0 (float): Initial price
        mu (float): Drift (annual return)
        sigma (float): Volatility
        T (float): Time horizon (in years)
        dt (float): Time step (default: 1 trading day)
        paths (int): Number of paths
        seed (int): Random seed
        log_returns (bool): Return log returns instead of prices
    Returns:
        (ndarray, ndarray): Tuple of (price/log-return matrix, time array)
    """
    if seed is not None:
        np.random.seed(seed)
        logging.info(f"Seed set to {seed}")

    N = int(T / dt)
    t = np.linspace(0, T, N + 1)
    logging.info(f"Simulating {paths} GBM path(s) over {N} steps")

    W = np.random.standard_normal((N, paths))
    W = np.vstack([np.zeros((1, paths)), np.cumsum(W * np.sqrt(dt), axis=0)])

    drift = (mu - 0.5 * sigma**2) * t[:, None]
    diffusion = sigma * W
    S = S0 * np.exp(drift + diffusion)

    if log_returns:
        log_ret = np.diff(np.log(S), axis=0)
        return log_ret, t[1:]
    return S, t

def plot_paths(S, t, title="GBM Simulation"):
    plt.figure(figsize=(10, 5))
    for i in range(S.shape[1]):
        plt.plot(t, S[:, i], lw=1.2, label=f'Path {i+1}')
    plt.title(title)
    plt.xlabel("Time (Years)")
    plt.ylabel("Log Returns" if title.startswith("Log") else "Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_to_csv(S, t, filename="gbm_output.csv"):
    df = pd.DataFrame(S, index=t)
    df.index.name = "Time"
    df.to_csv(filename)
    logging.info(f"Saved output to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Geometric Brownian Motion paths.")
    parser.add_argument("--S0", type=float, default=100.0, help="Initial price")
    parser.add_argument("--mu", type=float, default=0.1, help="Expected return")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument("--T", type=float, default=1.0, help="Time horizon in years")
    parser.add_argument("--paths", type=int, default=3, help="Number of paths to simulate")
    parser.add_argument("--log", action="store_true", help="Plot log returns instead of prices")
    parser.add_argument("--export", action="store_true", help="Export simulation to CSV")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    output, t = simulate_gbm(
        S0=args.S0,
        mu=args.mu,
        sigma=args.sigma,
        T=args.T,
        paths=args.paths,
        seed=args.seed,
        log_returns=args.log
    )

    title = "Log Return Paths (GBM)" if args.log else "Simulated GBM Price Paths"
    plot_paths(output, t, title=title)

    if args.export:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"gbm_sim_{now}.csv"
        save_to_csv(output, t, fname)
