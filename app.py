import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. إعدادات الصفحة العامة للـ UI
# ==========================================
st.set_page_config(page_title="منصة المؤشرات الرقمية الشاملة", layout="wide", page_icon="📊")
st.title("لوحة القياس والمؤشرات الرقمية لجميع الشركات 📈")

# ==========================================
# 2. الدوال الرياضية الفنية القياسية (مطابقة لـ TradingView بدقة)
# ==========================================
def calc_wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calc_hma(series, length):
    half_len = int(length / 2)
    sqrt_len = int(np.sqrt(length))
    wma_half = calc_wma(series, half_len)
    wma_full = calc_wma(series, length)
    return calc_wma(2 * wma_half - wma_full, sqrt_len)

def calc_rsi(series, length=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calc_atr(high, low, close, length=14):
    hl = high - low
    h_cp = abs(high - close.shift(1))
    l_cp = abs(low - close.shift(1))
    tr = pd.concat([hl, h_cp, l_cp], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def calc_cci(high, low, close, length=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=length).mean()
    mad_tp = tp.rolling(window=length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad_tp)

def calc_stoch(high, low, close, length=14):
    low_min = low.rolling(window=length).min()
    high_max = high.rolling(window=length).max()
    return 100 * (close - low_min) / (high_max - low_min)

def calc_adx_dmi(high, low, close, length=14):
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    tr_s = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_s = pd.Series(plus_dm, index=close.index).ewm(alpha=1/length, adjust=False).mean()
    minus_s = pd.Series(minus_dm, index=close.index).ewm(alpha=1/length, adjust=False).mean()
    
    plus_di = 100 * plus_s / tr_s
    minus_di = 100 * minus_s / tr_s
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return plus_di, minus_di, adx

def calc_cmf(high, low, close, volume, length):
    ad = ((close - low) - (high - close)) / (high - low).replace(0, 0.0001)
    ad = ad.replace([np.inf, -np.inf], 0).fillna(0)
    return (ad * volume).rolling(length).sum() / volume.rolling(length).sum()

def calc_kama(close, length, fast=2, slow=30):
    change = abs(close.diff(length))
    volatility = abs(close.diff()).rolling(length).sum()
    er = (change / volatility).replace([np.inf, -np.inf], 0).fillna(0)
    fast_alpha = 2 / (fast + 1)
    slow_alpha = 2 / (slow + 1)
    sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
    kama = np.zeros(len(close))
    kama[:] = np.nan
    close_arr = close.values
    sc_arr = sc.values
    
    first_valid = -1
    for i in range(len(sc_arr)):
        if not np.isnan(sc_arr[i]):
            first_valid = i
            break
    if first_valid != -1 and first_valid > 0:
        kama[first_valid - 1] = close_arr[first_valid - 1]
        for i in range(first_valid, len(close_arr)):
            kama[i] = kama[i-1] + sc_arr[i] * (close_arr[i] - kama[i-1])
    return pd.Series(kama, index=close.index)

# ==========================================
# 3. استراتيجية الإشارات المتقدمة (Trinity Pro)
# ==========================================
def apply_trinity_pro(df):
    if len(df) < 30: return df
    df['KAMA'] = calc_kama(df['Close'], length=min(100, len(df)-1))
    df['HMA'] = calc_hma(df['Close'], length=min(21, len(df)-1))
    df['Price_HMA'] = calc_hma(df['Close'], length=min(9, len(df)-1))
    df['RSI_Fast'] = calc_rsi(df['Price_HMA'], length=14)
    df['RSI_Slow'] = calc_rsi(df['Price_HMA'], length=25)
    df['Vol_HMA'] = calc_hma(df['Volume'], length=min(9, len(df)-1))
    df['Is_High_Vol'] = df['Volume'] > df['Vol_HMA']
    df['CMF'] = calc_cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=min(20, len(df)-1))
    df['CMF_Fast'] = calc_hma(df['CMF'].fillna(0), length=min(9, len(df)-1))
    df['CMF_Slow'] = calc_hma(df['CMF'].fillna(0), length=min(21, len(df)-1))
    df['Is_CMF_Bullish'] = df['CMF_Fast'] > df['CMF_Slow']

    high_low_diff = (df['High'] - df['Low']).replace(0, 0.0001) 
    buy_vol = df['Volume'] * (df['Close'] - df['Low']) / high_low_diff
    sell_vol = df['Volume'] * (df['High'] - df['Close']) / high_low_diff
    df['Delta_Filter'] = (buy_vol - sell_vol) > 0
    df['RSI_Val'] = calc_rsi(df['Close'], length=14)
    df['RSI_Filter'] = df['RSI_Val'] < 70
    df['Is_MTF_Bullish'] = df['Close'] > calc_hma(df['Close'], length=min(50, len(df)-1))

    df['Uptrend'] = df['Close'] > df['KAMA']
    df['HMA_Turns_Up'] = (df['HMA'] > df['HMA'].shift(1)) & (df['HMA'].shift(1) < df['HMA'].shift(2))
    df['RSI_Cross'] = (df['RSI_Fast'] > df['RSI_Slow']) & (df['RSI_Fast'].shift(1) <= df['RSI_Slow'].shift(1))
    df['Trinity_Buy'] = df['RSI_Cross'] & df['Is_High_Vol'] & df['Is_CMF_Bullish'] & df['Is_MTF_Bullish']
    df['Strong_Buy_Signal'] = df['Uptrend'] & df['HMA_Turns_Up'] & df['Delta_Filter'] & df['RSI_Filter'] & df['Trinity_Buy']
    return df

# ==========================================
# 4. قاعدة بيانات الشركات والقطاعات المعتمدة
# ==========================================
tickers_dict = {
    "أرامكو (طاقة)": "2222.SR", "لوبريف (طاقة)": "2223.SR", "الدريس (طاقة)": "4200.SR", "البحري (طاقة)": "4030.SR", "كهرباء السعودية (مرافق)": "5110.SR", "غازكو (مرافق)": "2080.SR",
    "سابك (مواد أساسية)": "2010.SR", "معادن (مواد أساسية)": "1211.SR", "ينساب (مواد أساسية)": "2230.SR", "سبكيم العالمية (مواد أساسية)": "2310.SR", "بترورابغ (مواد أساسية)": "2380.SR", "المتقدمة (مواد أساسية)": "2330.SR", "كيان السعودية (مواد أساسية)": "2350.SR", "التصنيع الوطنية (مواد أساسية)": "2060.SR", "أسمنت اليمامة (مواد أساسية)": "3020.SR", "أسمنت السعودية (مواد أساسية)": "3030.SR",
    "الراجحي (بنوك)": "1120.SR", "الأهلي (بنوك)": "1180.SR", "الإنماء (بنوك)": "1150.SR", "البلاد (بنوك)": "1140.SR", "بنك الرياض (بنوك)": "1010.SR", "بنك الجزيرة (بنوك)": "1020.SR", "مجموعة تداول (خدمات مالية)": "1111.SR",
    "اس تي سي STC (اتصالات)": "7010.SR", "موبايلي (اتصالات)": "7020.SR", "زين السعودية (اتصالات)": "7030.SR", "سلوشنز (تقنية)": "7200.SR", "علم (تقنية)": "7203.SR",
    "جرير (تجزئة)": "4190.SR", "إكسترا (تجزئة)": "4003.SR", "أسواق العثيم (تجزئة أغذية)": "4001.SR", "سينومي ريتيل (تجزئة)": "4240.SR", "ساسكو (تجزئة طاقة)": "4050.SR", "المراعي (إنتاج أغذية)": "2280.SR", "نادك (إنتاج أغذية)": "2270.SR", "صافولا
