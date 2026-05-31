import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Trinity Pro Pro", layout="wide")
st.title("🚀 منصة Trinity Pro: المسح الشامل")

# قائمة شركاتك (يمكنك إكمال الـ 100 هنا)
tickers_list = {
    "أرامكو": "2222.SR", "الراجحي": "1120.SR", "سابك": "2010.SR", 
    "الإنماء": "1150.SR", "STC": "7010.SR", "معادن": "1211.SR"
}

@st.cache_data(ttl=3600)
def load_data():
    # تحميل البيانات بأسلوب قوي
    data = yf.download(list(tickers_list.values()), period="1y", interval="1d", group_by='ticker')
    return data

data = load_data()

def get_df(ticker):
    # دالة استخراج بيانات تحمي من الـ KeyError
    if ticker in data.columns.levels[0]:
        return data[ticker].dropna()
    return pd.DataFrame()

# دالة الاستراتيجية
def apply_trinity_pro(df):
    df = df.copy()
    df['KAMA'] = df['Close'].rolling(50).mean()
    df['Strong_Buy'] = df['Close'] > df['KAMA']
    return df

# التابات
tab1, tab2, tab3 = st.tabs(["📉 شارت فردي", "⚡ ماسح الاختراقات", "🎯 الصفقات الماسية"])

with tab1:
    s = st.selectbox("اختر السهم:", list(tickers_list.keys()))
    df = get_df(tickers_list[s])
    if not df.empty:
        df = apply_trinity_pro(df)
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.add_trace(go.Scatter(x=df.index, y=df['KAMA'], name="KAMA"))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("الأسهم في ترند صاعد")
    res = []
    for name, ticker in tickers_list.items():
        df = get_df(ticker)
        if not df.empty:
            df = apply_trinity_pro(df)
            if df['Strong_Buy'].iloc[-1]:
                res.append({"الشركة": name, "الإغلاق": round(df['Close'].iloc[-1], 2)})
    st.dataframe(pd.DataFrame(res), use_container_width=True)

with tab3:
    st.info("المنطقة المخصصة للفلتر المركب (الدايفرجنس + الفيبو)...")
