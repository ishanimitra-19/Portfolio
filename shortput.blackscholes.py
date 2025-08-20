# Short Put (Sell) 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

# ----------------------------
# Math / pricing utilities
# ----------------------------

def black_scholes_put(S: float, K: float, sigma: float, r: float, t: float) -> float:
    if sigma <= 0 or t <= 0 or S <= 0 or K <= 0:
        raise ValueError("S, K, sigma, and t must be positive.")
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)


def gbm_path(S0: float, mu: float, sigma: float, steps: int, T: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    dt = T / steps
    Z = rng.standard_normal(steps)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    path = np.empty(steps + 1)
    path[0] = S0
    path[1:] = S0 * np.exp(np.cumsum(log_returns))
    return path


def gbm_terminal_samples(S0: float, mu: float, sigma: float, T: float, n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    Z = rng.standard_normal(n)
    ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return ST


# ----------------------------
# Core computations
# ----------------------------

def short_put_pl_per_share(premium_received: float, K: float, ST: float) -> float:
    payoff = max(K - ST, 0.0)
    return premium_received - payoff


def run_once(S, K, sigma, r, t, steps=252, mu_for_path=None, seed=None):
    theo = black_scholes_put(S, K, sigma, r, t)
    mu = r if mu_for_path is None else mu_for_path
    path = gbm_path(S, mu, sigma, steps, t, seed=seed)
    ST = float(path[-1])
    pl_per_share = short_put_pl_per_share(theo, K, ST)
    return theo, path, ST, pl_per_share


def mc_pl_per_contract_vectorized(S, K, sigma, r, t, market_bid, n_paths=1_000_000, mu_for_mc=None, seed=None):
    mu = r if mu_for_mc is None else mu_for_mc
    ST = gbm_terminal_samples(S, mu, sigma, t, n_paths, seed=seed)
    payoff = np.maximum(K - ST, 0.0)
    pl_per_share = market_bid - payoff
    return (pl_per_share * 100.0)  # returns array length n_paths


def mc_mean_pl_streaming(S, K, sigma, r, t, market_bid, n_paths=1_000_000, batch_size=200_000, mu_for_mc=None, seed=None):
    """
    Memory-safe mean using batches. Handles very large n_paths.
    """
    mu = r if mu_for_mc is None else mu_for_mc
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    remaining = n_paths
    sum_pl = 0.0
    while remaining > 0:
        b = min(batch_size, remaining)
        Z = rng.standard_normal(b)
        ST = S * np.exp((mu - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * Z)
        payoff = np.maximum(K - ST, 0.0)
        pl_per_share = market_bid - payoff
        sum_pl += float(pl_per_share.sum()) * 100.0
        remaining -= b
    return sum_pl / n_paths  # expected P/L per contract


# ----------------------------
# Plotting helpers
# ----------------------------

def plot_single_path(path: np.ndarray, K: float, pl_per_share: float):
    steps = len(path) - 1
    ST = path[-1]
    plt.figure()
    plt.title("Terminal Value of a Short Put Option")
    plt.plot(path, label="Price Path")
    plt.hlines(K, 0, steps, label="Strike")
    plt.vlines(steps, min(ST, K), max(ST, K), label="P/L segment")
    plt.xlabel("Time step")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cumulative_equity(sampled_contract_pls: np.ndarray, max_points: int = 50_000):
    n = sampled_contract_pls.size
    if n > max_points:
        stride = n // max_points
        y = sampled_contract_pls[::stride]
    else:
        y = sampled_contract_pls
    plt.figure()
    plt.title("Trading the Short Put Edge Over Time")
    plt.plot(np.cumsum(y), label="Account Equity")
    plt.xlabel("Option Trade")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------
# Simple Tkinter UI
# ----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Short Put Simulator")

        self.vars = {
            "S": tk.StringVar(value="100"),
            "K": tk.StringVar(value="100"),
            "sigma": tk.StringVar(value="0.30"),
            "r": tk.StringVar(value="0.05"),
            "t_years": tk.StringVar(value="1.0"),
            "market_bid": tk.StringVar(value="8.90"),
            "market_ask": tk.StringVar(value="9.50"),
            "steps": tk.StringVar(value="252"),
            "mc_paths": tk.StringVar(value="1000000"),      # 1,000,000 default
            "mc_batch": tk.StringVar(value="200000"),       # streaming batch size
            "mu_override": tk.StringVar(value=""),
            "seed": tk.StringVar(value=""),
        }

        form = ttk.Frame(self, padding=10)
        form.grid(row=0, column=0, sticky="nsew")

        def add_row(r, label, key):
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="e", padx=4, pady=2)
            ttk.Entry(form, textvariable=self.vars[key], width=16).grid(row=r, column=1, sticky="w", padx=4, pady=2)

        rows = [
            ("Spot S", "S"),
            ("Strike K", "K"),
            ("Vol sigma (dec)", "sigma"),
            ("Rate r (dec, cont.)", "r"),
            ("Time t (years)", "t_years"),
            ("Market bid", "market_bid"),
            ("Market ask", "market_ask"),
            ("Path steps", "steps"),
            ("MC paths", "mc_paths"),
            ("MC batch size", "mc_batch"),
            ("Drift μ override (blank=r)", "mu_override"),
            ("Seed (blank=random)", "seed"),
        ]
        for i, (lbl, key) in enumerate(rows):
            add_row(i, lbl, key)

        btns = ttk.Frame(self, padding=(10, 0))
        btns.grid(row=1, column=0, sticky="ew")
        ttk.Button(btns, text="Run single path + price", command=self.on_run_once).grid(row=0, column=0, padx=4, pady=6)
        ttk.Button(btns, text="Run Monte Carlo (mean only, streaming)", command=self.on_run_mc_stream).grid(row=0, column=1, padx=4, pady=6)
        ttk.Button(btns, text="Run Monte Carlo (full, plots)", command=self.on_run_mc_full).grid(row=0, column=2, padx=4, pady=6)

        self.out = tk.Text(self, height=16, width=96)
        self.out.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def read_floats(self):
        f = {}
        f["S"] = float(self.vars["S"].get())
        f["K"] = float(self.vars["K"].get())
        f["sigma"] = float(self.vars["sigma"].get())
        f["r"] = float(self.vars["r"].get())
        f["t"] = float(self.vars["t_years"].get())
        f["market_bid"] = float(self.vars["market_bid"].get())
        f["market_ask"] = float(self.vars["market_ask"].get())
        f["steps"] = int(float(self.vars["steps"].get()))
        f["mc_paths"] = int(float(self.vars["mc_paths"].get()))
        f["mc_batch"] = int(float(self.vars["mc_batch"].get()))
        mu_text = self.vars["mu_override"].get().strip()
        f["mu"] = None if mu_text == "" else float(mu_text)
        seed_text = self.vars["seed"].get().strip()
        f["seed"] = None if seed_text == "" else int(seed_text)
        return f

    def write(self, s: str):
        self.out.insert("end", s + "\n")
        self.out.see("end")

    def on_run_once(self):
        try:
            p = self.read_floats()
            theo, path, ST, pl_share = run_once(
                p["S"], p["K"], p["sigma"], p["r"], p["t"],
                steps=p["steps"], mu_for_path=p["mu"], seed=p["seed"]
            )
            self.write(f"Theoretical put price: {theo:.6f}")
            self.write(f"Market maker quote: {p['market_bid']:.2f} @ {p['market_ask']:.2f}")
            self.write(f"Terminal price S_T: {ST:.6f}")
            self.write(f"Short put P/L per share at S_T (using theoretical premium): {pl_share:.6f}")
            plot_single_path(path, p["K"], pl_share)
        except Exception as e:
            self.write(f"Error: {e}")

    def on_run_mc_stream(self):
        try:
            p = self.read_floats()
            exp_pl_contract = mc_mean_pl_streaming(
                p["S"], p["K"], p["sigma"], p["r"], p["t"], p["market_bid"],
                n_paths=p["mc_paths"], batch_size=p["mc_batch"], mu_for_mc=p["mu"], seed=p["seed"]
            )
            self.write(f"Expected P/L per contract at bid (streaming mean, n={p['mc_paths']:,}): {exp_pl_contract:.6f}")
        except Exception as e:
            self.write(f"Error: {e}")

    def on_run_mc_full(self):
        try:
            p = self.read_floats()
            pls_contract = mc_pl_per_contract_vectorized(
                p["S"], p["K"], p["sigma"], p["r"], p["t"], p["market_bid"],
                n_paths=p["mc_paths"], mu_for_mc=p["mu"], seed=p["seed"]
            )
            self.write(f"MC sample size: {p['mc_paths']:,}")
            self.write(f"Sample mean P/L per contract: {pls_contract.mean():.6f}")
            self.write(f"Sample std P/L per contract:  {pls_contract.std(ddof=1):.6f}")
            q = np.quantile(pls_contract, [0.01, 0.05, 0.50, 0.95, 0.99])
            self.write(f"Quantiles (1%,5%,50%,95%,99%): {q.round(4)}")
            plot_cumulative_equity(pls_contract)
        except MemoryError:
            self.write("MemoryError: Use the streaming button or reduce n_paths.")
        except Exception as e:
            self.write(f"Error: {e}")


if __name__ == "__main__":
    App().mainloop()
