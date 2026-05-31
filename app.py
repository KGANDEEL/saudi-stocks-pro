import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Trinity Pro", layout="wide")
st.title("📊 منصة Trinity Pro المحدثة")

tickers_list = {
    "أرامكو": "2222.SR", "لوبريف": "2223.SR", "سابك": "2010.SR", "معادن": "1211.SR",
    "الراجحي": "1120.SR", "الإنماء": "1150.SR", "STC": "7010.SR", "سلوشنز": "7200.SR"
}

# دالة تحميل ذكية لكل سهم على حدة (تضمن عدم ضياع البيانات)
@st.cache_data(ttl=3600)
def load_single_ticker(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        return df if not df.empty else pd.DataFrame()
    except:
        return pd.DataFrame()

# دالة فنية بسيطة للتجربة
def apply_trinity_pro(df):
    df = df.copy()
    df['KAMA_Slow'] = df['Close'].rolling(50).mean()
    df['Strong_Buy'] = df['Close'] > df['KAMA_Slow']
    return df

# عرض النتائج
tab1, tab2 = st.tabs(["📉 شارت فردي", "🎯 الماسح الشامل"])

with tab1:
    selected = st.selectbox("اختر السهم:", list(tickers_list.keys()))
    df = load_single_ticker(tickers_list[selected])
    if not df.empty:
        df = apply_trinity_pro(df)
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("جاري محاولة الاتصال بـ Yahoo Finance... يرجى الانتظار أو تحديث الصفحة.")

with tab2:
    if st.button("بدء المسح المباشر"):
        results = []
        progress_bar = st.progress(0)
        for i, (name, ticker) in enumerate(tickers_list.items()):
            df = load_single_ticker(ticker)
            if not df.empty:
                df = apply_trinity_pro(df)
                if df['Strong_Buy'].iloc[-1]:
                    results.append({"الشركة": name, "الإغلاق": round(df['Close'].iloc[-1], 2)})
            progress_bar.progress((i + 1) / len(tickers_list))
        
        st.dataframe(pd.DataFrame(results))
