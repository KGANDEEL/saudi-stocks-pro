import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- إعدادات الصفحة ---
st.set_page_config(page_title="Trinity Pro Dashboard", layout="wide")
st.title("📊 منصة Trinity Pro: التحليل والمسح الشامل")

# --- قائمة الشركات (تقدر تزودها براحتك) ---
tickers_list = {
    "أرامكو": "2222.SR", "لوبريف": "2223.SR", "سابك": "2010.SR", "معادن": "1211.SR", "ينساب": "2230.SR",
    "سبكيم": "2310.SR", "المتقدمة": "2330.SR", "كيان": "2350.SR", "التصنيع": "2060.SR", "بترورابغ": "2380.SR",
    "جرير": "4190.SR", "إكسترا": "4003.SR", "العثيم": "4001.SR", "النهدي": "4164.SR", "ساسكو": "4050.SR",
    "المراعي": "2280.SR", "صافولا": "2050.SR", "الراجحي": "1120.SR", "الأهلي": "1180.SR", "الإنماء": "1150.SR",
    "البلاد": "1140.SR", "STC": "7010.SR", "موبايلي": "7020.SR", "سلوشنز": "7200.SR", "علم": "7203.SR"
}

# --- الدوال الفنية ---
def calc_kama(close, length=50):
    change = abs(close.diff(length))
    vol = abs(close.diff()).rolling(length).sum()
    er = (change / vol).fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    return close.ewm(alpha=sc.mean()).mean()

def apply_trinity_pro(df):
    df['KAMA_Fast'] = calc_kama(df['Close'], 50)
    df['KAMA_Slow'] = calc_kama(df['Close'], 100)
    df['Early_Buy'] = df['Close'] > df['Close'].shift(1)
    df['Strong_Buy'] = df['Close'] > df['KAMA_Slow']
    return df

def get_ticker_df(data, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        try: return data.xs(ticker, axis=1, level=1)
        except: return pd.DataFrame()
    return data

@st.cache_data(ttl=3600)
def load_all_data():
    return yf.download(list(tickers_list.values()), period="1y", interval="1d", group_by='ticker')

data = load_all_data()

# --- التابات ---
tab_chart, tab1, tab2, tab3 = st.tabs(["📉 شارت فردي", "⚡ فرص الاختراق", "📈 ترند صاعد", "🎯 الصفقات الماسية"])

# التاب 1: الشارت الفردي
with tab_chart:
    selected_name = st.selectbox("اختر السهم:", list(tickers_list.keys()))
    df = get_ticker_df(data, tickers_list[selected_name]).dropna()
    if not df.empty:
        df = apply_trinity_pro(df)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="السعر"))
        fig.add_trace(go.Scatter(x=df.index, y=df['KAMA_Fast'], name="KAMA سريع", line=dict(color='cyan', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['KAMA_Slow'], name="KAMA بطيء", line=dict(color='orange', width=2)))
        st.plotly_chart(fig, use_container_width=True)
    else: st.error("لا توجد بيانات لهذا السهم.")

# التابات للماسح
def run_scanner(filter_col):
    results = []
    for name, ticker in tickers_list.items():
        df = get_ticker_df(data, ticker).dropna()
        if not df.empty:
            df = apply_trinity_pro(df)
            if df[filter_col].iloc[-1]:
                results.append({"الشركة": name, "الإغلاق": round(df['Close'].iloc[-1], 2)})
    st.dataframe(pd.DataFrame(results), use_container_width=True)

with tab1: run_scanner('Early_Buy')
with tab2: run_scanner('Strong_Buy')
with tab3: st.info("المنطقة المخصصة لإشاراتك المركبة القادمة...")
