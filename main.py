import streamlit as st
import numpy as np
from scipy.stats import norm

st.title("Black-Scholes Option Pricing Calculator")
st.number_input("Risk-free interest rate (r)", min_value=0.0, value=0.05, step=0.01)
st.number_input("Current stock price (S)", min_value=0.0, value=100.0, step=1.0)
st.number_input("Strike price (K)", min_value=0.0, value=100.0, step=1.0)
st.number_input("Time to expiration (T)", min_value=0.0, value=1.0, step=0.01)
st.number_input("Volatility (σ)", min_value=0.0, value=0.2, step=0.01)

def blackScholes(r, S, K, T, sigma, type = "C"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type == "C":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif type == "P":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'C' for call or 'P' for put.")