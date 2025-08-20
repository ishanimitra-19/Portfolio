# Short Call; (Sell) 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Short call option simulator under the Black–Scholes framework.

This simulator is for educational purposes only and does not constitute financial advice.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

# --- Pricing utilities ---

def black_scholes_call(S: float, K: float, sigma: float, r: float, t: float) -> float:
    """Return the theoretical Black–Scholes price of a European call."""
    if sigma <= 0 or t <= 0 or S <= 0 or K <= 0:
        raise ValueError("S, K, sigma, and t must be positive.")
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

def gbm_path(S0: float, mu: float, sigma: float, steps: int, T: float, seed: int = None) -> np.ndarray:
    """Simulate one geometric Brownian motion price path."""
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    dt = T / steps
    Z = rng.standard_normal(steps)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    path = np.empty(steps + 1)
    path[0] = S0
    path[1:] = S0 * np.exp(np.cumsum(log_returns))
    return path

def gbm_terminal_samples(S0: float, mu: float, sigma: float, T: float, n: int, seed: int = None) -> np.ndarray:
    """Generate terminal prices under GBM."""
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    Z = rng.standard_normal(n)
    return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

# --- Core computations (SHORT call) ---

def short_call_pl_per_share(premium_received: float, K: float, ST: float) -> float:
    """Return profit/loss per share for a short call: premium minus intrinsic value."""
    payoff = max(ST - K, 0.0)
    return premium_received - payoff

def run_once(S, K, sigma, r, t, steps=252, mu_for_path=None, seed=None):
    """Simulate a single short call trade; return theoretical premium, path, terminal price, and P/L."""
    theo = black_scholes_call(S, K, sigma, r, t)
    mu = r if mu_for_path is None else mu_for_path
    path = gbm_path(S, mu, sigma, steps, t, seed=seed)
    ST = float(path[-1])
    pl_per_share = short_call_pl_per_share(theo, K, ST)
    return theo, path, ST, pl_per_share

def mc_pl_per_contract_vectorized(S, K, sigma, r, t, market_bid, n_paths=1_000_000, mu_for_mc=None, seed=None):
    """Vectorized Monte Carlo estimate of short-call P/L per contract (premium from bid price)."""
    mu = r if mu_for_mc is None else mu_for_mc
    ST = gbm_terminal_samples(S, mu, sigma, t, n_paths, seed=seed)
    payoff = np.maximum(ST - K, 0.0)
    pl_per_share = market_bid - payoff
    return (pl_per_share * 100.0)

def mc_mean_pl_streaming(S, K, sigma, r, t, market_bid, n_paths=1_000_000, batch_size=200_000, mu_for_mc=None, seed=None):
    """Compute the expected P/L per contract via streaming Monte Carlo."""
    mu = r if mu_for_mc is None else mu_for_mc
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    remaining = n_paths
    sum_pl = 0.0
    while remaining > 0:
        b = min(batch_size, remaining)
        Z = rng.standard_normal(b)
        ST = S * np.exp((mu - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * Z)
        payoff = np.maximum(ST - K, 0.0)
        pl_per_share = market_bid - payoff
        sum_pl += float(pl_per_share.sum()) * 100.0
        remaining -= b
    return sum_pl / n_paths

# --- Plotting helpers ---

def plot_single_path(path: np.ndarray, K: float, pl_per_share: float):
    """Plot a simulated price path for a short call option."""
    steps = len(path) - 1
    ST = path[-1]
    plt.figure()
    plt.title("Terminal Value of a Short Call Option")
    plt.plot(path, label="Price Path")
    plt.hlines(K, 0, steps, label="Strike")
    plt.vlines(steps, min(ST, K), max(ST, K), label="P/L segment")
    plt.xlabel("Time step")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cumulative_equity(sampled_contract_pls: np.ndarray, max_points: int = 50_000):
    """Plot cumulative equity across sequential short call trades."""
    n = sampled_contract_pls.size
    y = sampled_contract_pls[::n // max_points] if n > max_points else sampled_contract_pls
    plt.figure()
    plt.title("Trading the Short Call Over Time")
    plt.plot(np.cumsum(y), label="Account Equity")
    plt.xlabel("Option Trade")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Tkinter UI ---

class App(tk.Tk):
    """GUI for simulating short call option trades."""
    def __init__(self):
        super().__init__()
        self.title("Short Call Simulator")

        # Default entry values
        self.vars = {
            "S": tk.StringVar(value="100"),
            "K": tk.StringVar(value="100"),
            "sigma": tk.StringVar(value="0.30"),
            "r": tk.StringVar(value="0.05"),
            "t_years": tk.StringVar(value="1.0"),
            "market_bid": tk.StringVar(value="9.50"),
            "market_ask": tk.StringVar(value="10.10"),
            "steps": tk.StringVar(value="252"),
            "mc_paths": tk.StringVar(value="1000000"),
            "mc_batch": tk.StringVar(value="200000"),
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
            ("Market bid (premium received)", "market_bid"),
            ("Market ask (for reference)", "market_ask"),
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
        """Convert entry fields to floats/ints, handling optional overrides."""
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
        """Append a line to the text output widget."""
        self.out.insert("end", s + "\n")
        self.out.see("end")

    def on_run_once(self):
        """Run a single-path simulation using theoretical premium."""
        try:
            p = self.read_floats()
            theo, path, ST, pl_share = run_once(
                p["S"], p["K"], p["sigma"], p["r"], p["t"],
                steps=p["steps"], mu_for_path=p["mu"], seed=p["seed"]
            )
            bid = p["market_bid"]
            ask = p["market_ask"]
            self.write(f"Theoretical call price (premium received): {theo:.6f}")
            self.write(f"Market maker quote: {bid:.2f} (bid) / {ask:.2f} (ask)")
            self.write(f"Terminal price S_T: {ST:.6f}")
            self.write(f"Short call P/L per share at S_T (using theoretical premium): {pl_share:.6f}")
            plot_single_path(path, p["K"], pl_share)
        except Exception as e:
            self.write(f"Error: {e}")

    def on_run_mc_stream(self):
        """Run a streaming Monte Carlo estimate using the market bid price."""
        try:
            p = self.read_floats()
            exp_pl_contract = mc_mean_pl_streaming(
                p["S"], p["K"], p["sigma"], p["r"], p["t"], p["market_bid"],
                n_paths=p["mc_paths"], batch_size=p["mc_batch"], mu_for_mc=p["mu"], seed=p["seed"]
            )
            n_paths_val = p["mc_paths"]
            self.write(f"Expected P/L per contract at bid (streaming mean, n={n_paths_val:,}): {exp_pl_contract:.6f}")
        except Exception as e:
            self.write(f"Error: {e}")

    def on_run_mc_full(self):
        """Run a full Monte Carlo simulation and plot the results."""
        try:
            p = self.read_floats()
            pls_contract = mc_pl_per_contract_vectorized(
                p["S"], p["K"], p["sigma"], p["r"], p["t"], p["market_bid"],
                n_paths=p["mc_paths"], mu_for_mc=p["mu"], seed=p["seed"]
            )
            n_paths_val = p["mc_paths"]
            self.write(f"MC sample size: {n_paths_val:,}")
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

