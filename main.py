import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp 
import matplotlib.pyplot as plt
import seaborn as sns

def blackScholes(r, S, K, T, sigma, type="C"):
    """
    Calculate Black-Scholes option price.

    Parameters:
    r (float): Risk-free interest rate
    S (float): Current stock price
    K (float): Option strike price
    T (float): Time to expiration (in years)
    sigma (float): Volatility of the underlying asset

    Returns:
    float: Black-Scholes option price
    """
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if type == "C":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif type == "P":
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'C' for call or 'P' for put.")

with st.sidebar:
    r = st.number_input("Risk-free interest rate (r)", value=0.05)
    S = st.number_input("Current stock price (S)", value=100)
    K = st.number_input("Option strike price (K)", value=100)
    T = st.number_input("Time to expiration (T in years)", value=1)
    sigma = st.number_input("Volatility (σ)", value=0.2)

    st.write("---")
    #Heatmap parameters
    min_stock = st.number_input("Min Stock Price", value=50)
    max_stock = st.number_input("Max Stock Price", value=150)
    min_vol = st.slider("Min Volatility", value=0.1)
    max_vol = st.slider("Max Volatility", value=0.5)

    st.write("### Black-Scholes Option Pricing Model")
    st.write("Calculate the price of European call and put options using the Black-Scholes model.")
    st.write("#### Inputs:")
    input_data = {
        "Parameter": ["Risk-free interest rate (r)", "Current stock price (S)", "Option strike price (K)", "Time to expiration (T)", "Volatility (σ)"],
        "Value": [r, S, K, T, sigma]
    }
    df = pd.DataFrame(input_data)
    df.index = [""] * len(df)  # Remove index numbers
    st.table(df)
    st.write("#### Results:")
    call_market_price = st.number_input("Call Option Purchase Price", value=1.0, min_value=0.0, step=0.01)
    put_market_price = st.number_input("Put Option Purchase Price", value=1.0, min_value=0.0, step=0.01)
    st.write(f"Call Option Price: {blackScholes(r, S, K, T, sigma, 'C')}")
    st.write(f"Put Option Price: {blackScholes(r, S, K, T, sigma, 'P')}")

 

# Two heatmaps: one for call PnL, one for put PnL
spot_range = np.linspace(min_stock, max_stock, 10)
vol_range = np.linspace(min_vol, max_vol, 10)

call_pnl = np.zeros((len(vol_range), len(spot_range)))
put_pnl = np.zeros((len(vol_range), len(spot_range)))
for i, sigma_val in enumerate(vol_range):
    for j, S_val in enumerate(spot_range):
        call_val = blackScholes(r, S_val, K, T, sigma_val, type="C")
        put_val = blackScholes(r, S_val, K, T, sigma_val, type="P")
        call_pnl[i, j] = call_val - call_market_price
        put_pnl[i, j] = put_val - put_market_price

import matplotlib.pyplot as plt
import seaborn as sns

col1, col2 = st.columns(2)
with col1:
    st.subheader("Call Option PnL Heatmap")
    fig_call, ax_call = plt.subplots(figsize=(8, 6))
    sns.heatmap(call_pnl, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), cmap="RdYlGn", ax=ax_call, annot=True, fmt=".2f")
    ax_call.set_xlabel("Spot Price")
    ax_call.set_ylabel("Volatility")
    ax_call.set_title("Call Option PnL Heatmap")
    st.pyplot(fig_call)

with col2:
    st.subheader("Put Option PnL Heatmap")
    fig_put, ax_put = plt.subplots(figsize=(8, 6))
    sns.heatmap(put_pnl, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), cmap="RdYlGn", ax=ax_put, annot=True, fmt=".2f")
    ax_put.set_xlabel("Spot Price")
    ax_put.set_ylabel("Volatility")
    ax_put.set_title("Put Option PnL Heatmap")
    st.pyplot(fig_put)

