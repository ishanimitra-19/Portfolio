import os
import json
import time
import threading
import random
import tkinter as tk
from tkinter import ttk, messagebox

import alpaca_trade_api as tradeapi
import openai

DATA_FILE = "equities.json"

# Read API keys from environment variables
ALPACA_KEY = os.environ.get("ALPACA_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets/"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialise Alpaca API client
api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version="v2")


def fetch_portfolio():
    positions = api.list_positions()
    portfolio = []
    for pos in positions:
        portfolio.append({
            "symbol": pos.symbol,
            "qty": pos.qty,
            "entry_price": pos.avg_entry_price,
            "current_price": pos.current_price,
            "unrealized_pl": pos.unrealized_pl,
            "side": "buy",
        })
    return portfolio


def fetch_open_orders():
    orders = api.list_orders(status="open")
    open_orders = []
    for order in orders:
        open_orders.append({
            "symbol": order.symbol,
            "qty": order.qty,
            "limit_price": order.limit_price,
            "side": "buy",
        })
    return open_orders


def fetch_mock_api(symbol: str) -> dict:
    """Mock price fetcher; replace with real market data if desired."""
    return {"price": 100}


def chatgpt_response(message: str) -> str:
    """Generate a response using OpenAI’s API based on portfolio context."""
    portfolio_data = fetch_portfolio()
    open_orders = fetch_open_orders()

    pre_prompt = f"""
You are an AI Portfolio Manager responsible for analyzing my portfolio.
Tasks:
1. Evaluate risk exposures of current holdings.
2. Analyze my open limit orders and their potential impact.
3. Provide insights into portfolio health, diversification and trade adjustments.
4. Speculate on the market outlook based on current conditions.
5. Identify potential market risks and suggest risk management strategies.
6. Stay on top of earnings dates for my positions.


Portfolio: {portfolio_data}
Open orders: {open_orders}

Answer the following question with this background in mind: {message}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": pre_prompt}],
        api_key=OPENAI_API_KEY,
    )
    return response["choices"][0]["message"]["content"]


class TradingBotGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Trading Bot")
        self.equities = self.load_equities()
        self.system_running = False

        # Build form for adding equities
        form_frame = tk.Frame(root)
        form_frame.pack(pady=10)

        tk.Label(form_frame, text="Symbol:").grid(row=0, column=0)
        self.symbol_entry = tk.Entry(form_frame)
        self.symbol_entry.grid(row=0, column=1)

        tk.Label(form_frame, text="Levels:").grid(row=0, column=2)
        self.levels_entry = tk.Entry(form_frame)
        self.levels_entry.grid(row=0, column=3)

        tk.Label(form_frame, text="Drawdown%:").grid(row=0, column=4)
        self.drawdown_entry = tk.Entry(form_frame)
        self.drawdown_entry.grid(row=0, column=5)

        add_button = tk.Button(form_frame, text="Add Equity", command=self.add_equity)
        add_button.grid(row=0, column=6)

        # Table for displaying equities
        self.tree = ttk.Treeview(
            root,
            columns=("Symbol", "Position", "Entry Price", "Levels", "Status"),
            show="headings",
        )
        for col in ["Symbol", "Position", "Entry Price", "Levels", "Status"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120)
        self.tree.pack(pady=10)

        toggle_button = tk.Button(
            root, text="Toggle Selected System", command=self.toggle_selected_system
        )
        toggle_button.pack(pady=5)

        remove_button = tk.Button(
            root, text="Remove Selected Equity", command=self.remove_selected_equity
        )
        remove_button.pack(pady=5)

        # Chat interface
        chat_frame = tk.Frame(root)
        chat_frame.pack(pady=10)

        self.chat_input = tk.Entry(chat_frame, width=50)
        self.chat_input.grid(row=0, column=0, padx=5)

        send_button = tk.Button(chat_frame, text="Send", command=self.send_message)
        send_button.grid(row=0, column=1)

        self.chat_output = tk.Text(root, height=5, width=60, state=tk.DISABLED)
        self.chat_output.pack()

        self.refresh_table()

        self.running = True
        self.auto_update_thread = threading.Thread(target=self.auto_update, daemon=True)
        self.auto_update_thread.start()

    def add_equity(self) -> None:
        """Add a new equity to the portfolio with drawdown levels."""
        symbol = self.symbol_entry.get().upper()
        levels_str = self.levels_entry.get()
        drawdown_str = self.drawdown_entry.get()

        if not symbol or not levels_str.isdigit() or not drawdown_str.replace(".", "", 1).isdigit():
            messagebox.showerror("Error", "Invalid input")
            return

        levels = int(levels_str)
        drawdown = float(drawdown_str) / 100
        entry_price = fetch_mock_api(symbol)["price"]

        # Compute target prices for each level
        level_prices = {
            i + 1: round(entry_price * (1 - drawdown * (i + 1)), 2)
            for i in range(levels)
        }

        self.equities[symbol] = {
            "position": 0,
            "entry_price": entry_price,
            "levels": level_prices,
            "drawdown": drawdown,
            "status": "Off",
        }
        self.save_equities()
        self.refresh_table()

    def toggle_selected_system(self) -> None:
        """Toggle the trading system on/off for selected equities."""
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "No equity selected")
            return

        for item in selected_items:
            symbol = self.tree.item(item)["values"][0]
            status = self.equities[symbol]["status"]
            self.equities[symbol]["status"] = "On" if status == "Off" else "Off"

        self.save_equities()
        self.refresh_table()

    def remove_selected_equity(self) -> None:
        """Remove selected equities from the portfolio."""
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "No equity selected")
            return

        for item in selected_items:
            symbol = self.tree.item(item)["values"][0]
            self.equities.pop(symbol, None)

        self.save_equities()
        self.refresh_table()

    def send_message(self) -> None:
        """Send a query to the ChatGPT portfolio assistant."""
        message = self.chat_input.get().strip()
        if not message:
            return

        response = chatgpt_response(message)
        self.chat_output.config(state=tk.NORMAL)
        self.chat_output.insert(tk.END, f"You: {message}\n{response}\n\n")
        self.chat_output.config(state=tk.DISABLED)
        self.chat_input.delete(0, tk.END)

    def fetch_alpaca_data(self, symbol: str) -> dict:
        """Fetch the latest trade price from Alpaca; returns -1 on error."""
        try:
            trade = api.get_latest_trade(symbol)
            return {"price": trade.price}
        except Exception:
            return {"price": -1}

    def check_existing_orders(self, symbol: str, price: float) -> bool:
        """Check whether there is already an open order at a given price."""
        try:
            orders = api.list_orders(status="open", symbols=symbol)
            return any(float(order.limit_price) == price for order in orders)
        except Exception as e:
            messagebox.showerror("API Error", f"Error checking orders: {e}")
            return False

    def get_max_entry_price(self, symbol: str) -> float:
        """Return the highest filled average price for recent orders of the symbol."""
        try:
            orders = api.list_orders(status="filled", limit=50)
            prices = [
                float(order.filled_avg_price)
                for order in orders
                if order.filled_avg_price and order.symbol == symbol
            ]
            return max(prices) if prices else -1
        except Exception as e:
            messagebox.showerror("API Error", f"Error fetching orders: {e}")
            return 0

    def trade_systems(self) -> None:
        """Iterate through equities and manage limit orders based on drawdown levels."""
        for symbol, data in self.equities.items():
            if data["status"] != "On":
                continue

            try:
                # If position exists, use the max entry price; otherwise place a market order
                position = api.get_position(symbol)
                entry_price = self.get_max_entry_price(symbol)
            except Exception:
                if data["position"] == 0:
                    api.submit_order(
                        symbol=symbol,
                        qty=1,
                        side="buy",
                        type="market",
                        time_in_force="gtc",
                    )
                    time.sleep(2)
                    entry_price = self.get_max_entry_price(symbol)
                    self.equities[symbol]["position"] = 1
                else:
                    continue

            levels = len(data["levels"])
            level_prices = {
                i + 1: round(entry_price * (1 - data["drawdown"] * (i + 1)), 2)
                for i in range(levels)
            }
            existing_levels = data.get("levels", {})
            for level, price in level_prices.items():
                if level not in existing_levels and -level not in existing_levels:
                    existing_levels[level] = price

            data["entry_price"] = entry_price
            data["levels"] = existing_levels
            data["position"] = 1

            for level, price in level_prices.items():
                if level in data["levels"]:
                    self.place_order(symbol, price, level)

        self.save_equities()
        self.refresh_table()

    def place_order(self, symbol: str, price: float, level: int) -> None:
        """Place a limit order if appropriate and update level tracking."""
        if price <= 0:
            print(f"Skipping order for {symbol} at non‑positive price: {price}")
            return
        if -level in self.equities[symbol]["levels"]:
            return

        try:
            api.submit_order(
                symbol=symbol,
                qty=1,
                side="buy",
                type="limit",
                time_in_force="gtc",
                limit_price=price,
            )
            self.equities[symbol]["levels"][-level] = price
            del self.equities[symbol]["levels"][level]
            print(f"Placed order for {symbol} @ {price}")
        except Exception as e:
            messagebox.showerror("Order Error", f"Error placing order: {e}")

    def refresh_table(self) -> None:
        """Refresh the Treeview widget with current equity data."""
        for row in self.tree.get_children():
            self.tree.delete(row)
        for symbol, data in self.equities.items():
            self.tree.insert(
                "", "end",
                values=(
                    symbol,
                    data["position"],
                    data["entry_price"],
                    str(data["levels"]),
                    data["status"],
                ),
            )

    def auto_update(self) -> None:
        """Background thread to periodically run trading logic."""
        while self.running:
            time.sleep(5)
            self.trade_systems()

    def save_equities(self) -> None:
        """Persist the equities dictionary to a JSON file."""
        with open(DATA_FILE, "w") as f:
            json.dump(self.equities, f)

    def load_equities(self) -> dict:
        """Load equities from a JSON file, handling missing or invalid files."""
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def on_close(self) -> None:
        """Handler for the window close event."""
        self.running = False
        self.save_equities()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TradingBotGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
