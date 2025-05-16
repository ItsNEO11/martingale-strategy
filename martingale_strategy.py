import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import datetime

st.set_page_config(page_title="Martingale Strategy Simulator", layout="wide")
st.title("📊 Martingale Position Averaging Simulator")
st.markdown("💡 All results include **0.05% open + 0.05% close trading fees**")

# === Sidebar Parameters ===
st.sidebar.header("Strategy Settings")
total_capital = st.sidebar.number_input("Total Capital (USD)", value=10000, step=500)
mode = st.sidebar.radio("Positioning Mode", ["Martingale", "Fixed Capital"], index=0)
martin_ratio = st.sidebar.slider("Martingale Ratio", 1.0, 3.0, 2.0, 0.1) if mode == "Martingale" else 1.0
num_entries = st.sidebar.slider("Number of Entries", 2, 10, 4)
target_price = st.sidebar.number_input("Target Rebound Price (USD)", value=15000, step=100)

# === Entry Settings ===
st.sidebar.subheader("Entry Price & Leverage Settings")
entry_prices, leverage_list = [], []
for i in range(num_entries):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        entry_prices.append(st.number_input(f"Entry {i+1} Price", value=14000 - i * 1000, step=100, key=f"price_{i}"))
    with col2:
        leverage_list.append(st.number_input(f"Entry {i+1} Leverage", value=5 if i == 0 else 10, min_value=1, max_value=100, step=1, key=f"lev_{i}"))

# === Capital Allocation ===
if mode == "Fixed Capital":
    capital_distribution = [total_capital / num_entries] * num_entries
else:
    weights = [martin_ratio ** i for i in range(num_entries)]
    capital_distribution = [total_capital * (w / sum(weights)) for w in weights]

# === Backtest Simulation ===
fee_rate = 0.0005
total_net_position = 0
total_quantity = 0
total_fee = 0
records = []

for i in range(num_entries):
    price = entry_prices[i]
    capital = capital_distribution[i]
    lev = leverage_list[i]

    position_value = capital * lev
    open_fee = position_value * fee_rate
    net_position = position_value - open_fee
    quantity = net_position / price if price > 0 else 0

    total_net_position += net_position
    total_quantity += quantity
    total_fee += open_fee

    avg_entry_price = total_net_position / total_quantity if total_quantity > 0 else 0
    avg_leverage = total_net_position / sum(capital_distribution[:i + 1])
    liquidation_price = avg_entry_price * (1 - 1 / avg_leverage)
    capital_ratio = sum(capital_distribution[:i + 1]) / total_capital
    avg_price_drop = (entry_prices[0] - avg_entry_price) / entry_prices[0]

    records.append({
        "Round": i + 1,
        "Entry Price": price,
        "Capital": round(capital, 2),
        "Leverage": lev,
        "Position Size": round(position_value, 2),
        "Fee": round(open_fee, 2),
        "Net Position": round(total_net_position, 2),
        "Avg Entry Price": round(avg_entry_price, 2),
        "Avg Leverage": round(avg_leverage, 2),
        "Liq. Price": round(liquidation_price, 2),
        "Capital Ratio": f"{capital_ratio * 100:.1f}%",
        "Avg Price Drop": f"{avg_price_drop * 100:.2f}%"
    })

df = pd.DataFrame(records)

# === Table Output ===
st.subheader("📈 Strategy Result Table")
st.dataframe(df, use_container_width=True)

# === Download CSV ===
st.subheader("📤 Export CSV")
csv = df.to_csv(index=False).encode('utf-8-sig')
filename = f"strategy_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
st.download_button("📥 Download CSV", data=csv, file_name=filename, mime="text/csv")

# === ROI Curve ===
st.subheader("📉 ROI Curve (With Fees)")
rebound_range = np.arange(min(entry_prices), target_price + 3000, 200)
if target_price not in rebound_range:
    rebound_range = np.sort(np.append(rebound_range, target_price))

fig1, ax1 = plt.subplots(figsize=(10, 5))
colors = plt.cm.tab10.colors

for step in range(1, num_entries + 1):
    sub_df = df.iloc[:step]
    net_position_value = sub_df["Net Position"].iloc[-1]
    quantity = ((sub_df["Position Size"] - sub_df["Fee"]) / sub_df["Entry Price"]).sum()
    open_fee = sub_df["Fee"].sum()

    roi_curve, profit_curve = [], []
    for p in rebound_range:
        total_value = p * quantity
        close_fee = total_value * fee_rate
        profit = total_value - net_position_value - close_fee
        roi = profit / total_capital
        roi_curve.append(roi * 100)
        profit_curve.append(profit)

    color = colors[(step - 1) % len(colors)]
    ax1.plot(rebound_range, roi_curve, linewidth=2, label=f"Up to Entry {step}", color=color)

    idx = np.abs(rebound_range - target_price).argmin()
    roi_at_target = roi_curve[idx]
    profit_at_target = profit_curve[idx]

    ax1.annotate(f"ROI: {roi_at_target:.2f}%", (target_price, roi_at_target),
                 textcoords="offset points", xytext=(-60, 20), ha='right',
                 fontsize=9, color=color, arrowprops=dict(arrowstyle='->', color=color, lw=1))
    ax1.annotate(f"Profit: ${profit_at_target:.0f}", (target_price, roi_at_target),
                 textcoords="offset points", xytext=(60, -30), ha='left',
                 fontsize=9, color=color, arrowprops=dict(arrowstyle='->', color=color, lw=1))

ax1.axvline(target_price, color='red', linestyle='--', linewidth=1.5, label="🎯 Target Price")
ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
ax1.set_title("ROI Curve per Entry Round", fontsize=14, weight='bold')
ax1.set_xlabel("BTC Price", fontsize=12)
ax1.set_ylabel("ROI (%)", fontsize=12)
ax1.grid(True, linestyle='--', linewidth=0.5, color='lightgray')
ax1.legend()
fig1.subplots_adjust(top=0.88)
st.pyplot(fig1)

# === Position Size Bar Chart ===
st.subheader("📊 Entry Price vs Position Size")
green_shades = ['#e6f4ea', '#c7e9c0', '#a8ddb5', '#74c476', '#4daf4a', '#238b45']
green_cmap = LinearSegmentedColormap.from_list("green_shades", green_shades)

prices = df["Entry Price"]
amounts = df["Position Size"]
normed = (amounts - amounts.min()) / (amounts.max() - amounts.min() + 1e-9)
colors = [green_cmap(val) for val in normed]

fig2, ax2 = plt.subplots(figsize=(10, 5))
bars = ax2.bar(prices, amounts, color=colors, width=200)
ax2.set_title("Position Size per Entry", fontsize=14, weight='bold')
ax2.set_xlabel("Entry Price", fontsize=12)
ax2.set_ylabel("Position Size (USD)", fontsize=12)
ax2.grid(axis='y', linestyle='--', linewidth=0.5, color='lightgray')
ymax = amounts.max() * 1.15
ax2.set_ylim(0, ymax)
for bar, amt in zip(bars, amounts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + ymax * 0.02,
             f"{int(amt):,}", ha='center', va='bottom', fontsize=9)
fig2.subplots_adjust(top=0.88)
st.pyplot(fig2)

# === Liquidation Margin Chart ===
st.subheader("🛡️ Liquidation Safety Margin per Entry")
avg_costs = df["Avg Entry Price"]
liq_prices = df["Liq. Price"]
margin_pct = ((avg_costs - liq_prices) / avg_costs * 100).round(2)

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(df["Round"], margin_pct, marker='o', color='orange', linewidth=2.5)
ax3.set_title("Safety Margin from Liquidation", fontsize=14, weight='bold')
ax3.set_xlabel("Entry Round", fontsize=12)
ax3.set_ylabel("Safety Margin (%)", fontsize=12)
ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
ax3.set_ylim(0, margin_pct.max() * 1.15)
for i, val in enumerate(margin_pct):
    ax3.annotate(f"{val:.2f}%", (df["Round"][i], val),
                 textcoords="offset points", xytext=(0, 8),
                 ha='center', fontsize=10)
fig3.subplots_adjust(top=0.88)
st.pyplot(fig3)

# === Summary ===
final_net_cost = df["Net Position"].iloc[-1]
final_quantity = ((df["Position Size"] - df["Fee"]) / df["Entry Price"]).sum()
final_close_fee = target_price * final_quantity * fee_rate
final_profit = target_price * final_quantity - final_net_cost - final_close_fee
final_roi = final_profit / total_capital

st.subheader("📌 Final Summary at Target Price")
st.markdown(f"""
- 🎯 Target Rebound Price: `{target_price} USD`
- 💰 Net Position Cost (w/ fees): `{final_net_cost:.2f} USD`
- 💸 Total Trading Fees (open+close): `{total_fee + final_close_fee:.2f} USD`
- 📈 Net Profit: `{final_profit:.2f} USD`
- 📊 Total ROI: `{final_roi * 100:.2f}%`
""")
