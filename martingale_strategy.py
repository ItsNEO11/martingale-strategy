import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
import datetime, json, os

# === 中文字体设置 ===
font_prop = None
try:
    font_path = os.path.join("fonts", "PingFangSC.ttf")  # 可替换为 NotoSansSC.ttf
    if os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
except:
    font_prop = None

# === 参数保存/加载 ===
PARAM_FILE = "saved_params.json"

def save_params(params: dict, file_path=PARAM_FILE):
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(params, f, indent=2)

def load_params(file_path=PARAM_FILE):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding='utf-8') as f:
            return json.load(f)
    return {}

# === 页面设置 ===
st.set_page_config(page_title="马丁格尔策略模拟器", layout="wide")
st.markdown('<h1 style="font-size:26px;">📊 马丁格尔加仓策略可视化模拟</h1>', unsafe_allow_html=True)
st.markdown("💡 所有计算结果已纳入 **0.05% 开仓 + 0.05% 平仓手续费**")

# === 参数输入 ===
saved = load_params()
st.sidebar.header("策略参数设置")
total_capital = st.sidebar.number_input("总本金（USD）", value=float(saved.get("total_capital", 10000)), step=100.0)

mode = st.sidebar.radio("加仓方式", ["马丁加仓", "固定金额"],
                        index=0 if saved.get("mode", "马丁加仓") == "马丁加仓" else 1)
martin_ratio = st.sidebar.slider("马丁倍率", 1.0, 3.0, float(saved.get("martin_ratio", 2.0)), 0.1)
num_entries = st.sidebar.slider("加仓轮次", 2, 10, int(saved.get("num_entries", 4)))

# === 小数位设置
st.sidebar.subheader("🔧 精度设置")
decimal_places = st.sidebar.selectbox("价格小数位数", options=[0, 1, 2, 3, 4, 5, 6], index=int(saved.get("decimal_places", 2)))
step_size = 1 / (10 ** decimal_places)
price_format = f"%.{decimal_places}f"

# === 目标反弹价格
target_price = st.sidebar.number_input("目标反弹价格（USD）",
                                       value=float(saved.get("target_price", 15000)),
                                       step=step_size, format=price_format)

# === 每轮加仓价格与杠杆 ===
st.sidebar.subheader("每轮加仓价格与杠杆设置")
entry_prices, leverage_list = [], []
key_prefix = f"v{num_entries}_{mode.replace(' ', '_')}"

for i in range(num_entries):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        entry_prices.append(st.number_input(f"第{i+1}轮加仓价格",
                                            value=float(saved.get(f"price_{i}", round(14000 - i * 1000, decimal_places))),
                                            step=step_size, format=price_format, key=f"{key_prefix}_price_{i}"))
    with col2:
        leverage_list.append(st.number_input(f"第{i+1}轮杠杆",
                                             value=int(saved.get(f"lev_{i}", 5 if i == 0 else 10)),
                                             min_value=1, max_value=100, step=1, key=f"{key_prefix}_lev_{i}"))

# === 保存按钮
if st.sidebar.button("💾 保存当前参数设置"):
    param_to_save = {
        "total_capital": total_capital,
        "mode": mode,
        "martin_ratio": martin_ratio,
        "num_entries": num_entries,
        "target_price": target_price,
        "decimal_places": decimal_places,
    }
    for i in range(num_entries):
        param_to_save[f"price_{i}"] = entry_prices[i]
        param_to_save[f"lev_{i}"] = leverage_list[i]
    save_params(param_to_save)
    st.sidebar.success("✅ 参数保存成功！")

# === 资金分配计算
if mode == "固定金额":
    capital_distribution = [total_capital / num_entries] * num_entries
else:
    weights = [martin_ratio ** i for i in range(num_entries)]
    capital_distribution = [total_capital * (w / sum(weights)) for w in weights]

# === 仓位计算
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
        "轮次": i + 1,
        "加仓价格": price,
        "加仓金额": round(capital, 2),
        "杠杆": lev,
        "加仓总额": round(position_value, 2),
        "手续费": round(open_fee, 2),
        "总持仓额": round(total_net_position, 2),
        "平均成本": round(avg_entry_price, 2),
        "平均杠杆": round(avg_leverage, 2),
        "爆仓价格": round(liquidation_price, 2),
        "资金占比": f"{capital_ratio * 100:.1f}%",
        "均价下移幅度": f"{avg_price_drop * 100:.2f}%"
    })

df = pd.DataFrame(records)

# === 表格展示
st.markdown(r'<h3 style="font-size:20px;">📈 策略模拟结果表</h3>', unsafe_allow_html=True)
st.dataframe(df, use_container_width=True)

# === 导出按钮
st.markdown(r'<h3 style="font-size:20px;">📤 导出策略明细</h3>', unsafe_allow_html=True)
csv = df.to_csv(index=False).encode('utf-8-sig')
filename = f"martingale_strategy_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
st.download_button("📥 下载策略明细 CSV", data=csv, file_name=filename, mime="text/csv")

# === ROI 曲线图
st.markdown(r'<h3 style="font-size:20px;">📉 ROI曲线图（含手续费）</h3>', unsafe_allow_html=True)
min_price = min(entry_prices)
x_margin = (target_price - min_price) * 0.6
x_left = min_price - x_margin * 0.1
x_right = target_price + x_margin * 1.2
rebound_range = np.arange(x_left, x_right, step_size)
if target_price not in rebound_range:
    rebound_range = np.sort(np.append(rebound_range, target_price))

fig1, ax1 = plt.subplots(figsize=(10, 5))
colors = plt.cm.tab10.colors
max_roi = 0

for step in range(1, num_entries + 1):
    sub_df = df.iloc[:step]
    net_position_value = sub_df["总持仓额"].iloc[-1]
    quantity = ((sub_df["加仓总额"] - sub_df["手续费"]) / sub_df["加仓价格"]).sum()
    roi_curve, profit_curve = [], []

    for p in rebound_range:
        total_value = p * quantity
        close_fee = total_value * fee_rate
        profit = total_value - net_position_value - close_fee
        roi = profit / total_capital
        roi_curve.append(roi * 100)
        profit_curve.append(profit)

    color = colors[(step - 1) % len(colors)]
    ax1.plot(rebound_range, roi_curve, linewidth=2, label=f"前{step}轮加仓", color=color)

    idx = np.abs(rebound_range - target_price).argmin()
    roi_at_target = roi_curve[idx]
    profit_at_target = profit_curve[idx]
    max_roi = max(max_roi, roi_at_target)

    ax1.annotate(f"ROI: {roi_at_target:.2f}%", (target_price, roi_at_target),
                 textcoords="offset points", xytext=(-60, 20), ha='right',
                 fontsize=9, color=color, arrowprops=dict(arrowstyle='->', color=color))
    ax1.annotate(f"Profit: ${profit_at_target:.0f}", (target_price, roi_at_target),
                 textcoords="offset points", xytext=(60, -30), ha='left',
                 fontsize=9, color=color, arrowprops=dict(arrowstyle='->', color=color))

ax1.axvline(target_price, color='red', linestyle='--', linewidth=1.5, label="★目标反弹价")
ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
ax1.set_xlim(x_left, x_right)
ax1.set_ylim(-10, max_roi * 1.4 if max_roi > 0 else 20)
ax1.set_title("分轮加仓后 ROI 曲线对比（含手续费）", fontsize=14, weight='bold', fontproperties=font_prop)
ax1.set_xlabel("标的价格", fontsize=12, fontproperties=font_prop)
ax1.set_ylabel("收益率 (%)", fontsize=12, fontproperties=font_prop)
ax1.legend(prop=font_prop, loc="best")
ax1.grid(True, linestyle='--', linewidth=0.5, color='lightgray')
fig1.subplots_adjust(top=0.88)
st.pyplot(fig1)

# === 每轮加仓价格 vs 加仓头寸金额图
st.markdown(r'<h3 style="font-size:20px;">📊 每轮加仓价格 vs 加仓头寸金额</h3>', unsafe_allow_html=True)
green_cmap = LinearSegmentedColormap.from_list("green_shades",
    ['#e6f4ea', '#c7e9c0', '#a8ddb5', '#74c476', '#4daf4a', '#238b45'])
prices, amounts = df["加仓价格"], df["加仓总额"]
normed = (amounts - amounts.min()) / (amounts.max() - amounts.min() + 1e-9)
colors = [green_cmap(val) for val in normed]
bar_width = (prices.max() - prices.min()) * 0.05 if prices.max() > prices.min() else 1
x_min, x_max = prices.min() - bar_width * 1.5, prices.max() + bar_width * 1.5

fig2, ax2 = plt.subplots(figsize=(10, 5))
bars = ax2.bar(prices, amounts, color=colors, width=bar_width)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(0, amounts.max() * 1.15)
ax2.set_title("每轮加仓头寸金额", fontsize=14, weight='bold', fontproperties=font_prop)
ax2.set_xlabel("加仓价格", fontsize=12, fontproperties=font_prop)
ax2.set_ylabel("加仓头寸（USD）", fontsize=12, fontproperties=font_prop)
ax2.grid(axis='y', linestyle='--', linewidth=0.5, color='lightgray')
for bar, amt in zip(bars, amounts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, height + 5,
             f"{int(amt):,}", ha='center', va='bottom', fontsize=9)
fig2.subplots_adjust(top=0.88)
st.pyplot(fig2)

# === 🛡️ 爆仓边界安全比例图
st.markdown(r'<h3 style="font-size:20px;">🛡️ 每轮加仓后爆仓价格安全边界</h3>', unsafe_allow_html=True)
avg_costs = df["平均成本"]
liq_prices = df["爆仓价格"]
margin_pct = ((avg_costs - liq_prices) / avg_costs * 100).round(2)

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(df["轮次"], margin_pct, marker='o', color='orange', linewidth=2.5)
ax3.set_title("每轮加仓后距离爆仓边界比例", fontsize=14, weight='bold', fontproperties=font_prop)
ax3.set_xlabel("加仓轮次", fontsize=12, fontproperties=font_prop)
ax3.set_ylabel("距离爆仓的安全边际 (%)", fontsize=12, fontproperties=font_prop)
ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
ax3.set_ylim(0, margin_pct.max() * 1.15)
for i, val in enumerate(margin_pct):
    ax3.annotate(f"{val:.2f}%", (df["轮次"][i], val), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=10)
fig3.subplots_adjust(top=0.88)
st.pyplot(fig3)

# === 📌 收益总结
st.markdown(r'<h3 style="font-size:20px;">📌 当标的反弹至目标价格时</h3>', unsafe_allow_html=True)
final_net_cost = df["总持仓额"].iloc[-1]
final_quantity = ((df["加仓总额"] - df["手续费"]) / df["加仓价格"]).sum()
final_close_fee = target_price * final_quantity * fee_rate
final_profit = target_price * final_quantity - final_net_cost - final_close_fee
final_roi = final_profit / total_capital

st.markdown(f"""
- 🎯 目标反弹价格：`{target_price} USD`
- 💰 当前持仓总成本（含手续费）：`{final_net_cost:.2f} USD`
- 💸 总交易手续费（开+平）：`{total_fee + final_close_fee:.2f} USD`
- 📈 持仓浮盈（净收益）：`{final_profit:.2f} USD`
- 📊 总收益率（ROI）：`{final_roi * 100:.2f}%`
""")
