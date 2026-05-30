import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# 1. الدوال الرياضية (بديل المكتبات الخارجية لضمان استقرار السيرفر)
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

def calc_rsi(series, length):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_cmf(high, low, close, volume, length):
    ad = ((close - low) - (high - close)) / (high - low)
    ad = ad.replace([np.inf, -np.inf], 0).fillna(0)
    mf_vol = ad * volume
    return mf_vol.rolling(length).sum() / volume.rolling(length).sum()

def calc_kama(close, length, fast=2, slow=30):
    change = abs(close.diff(length))
    volatility = abs(close.diff()).rolling(length).sum()
    er = change / volatility
    er = er.replace([np.inf, -np.inf], 0).fillna(0)

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
# 2. إعدادات الصفحة الأساسية وقائمة الشركات
# ==========================================
st.set_page_config(page_title="منصة التحليل الاحترافية", layout="wide", page_icon="📈")
st.title("لوحة تحليل الأسهم المتقدمة 📊")

tickers_dict = {
    "أرامكو (طاقة)": "2222.SR", "سابك (بتروكيماويات)": "2010.SR", "ينساب (بتروكيماويات)": "2230.SR",
    "سبكيم (بتروكيماويات)": "2310.SR", "بترورابغ (طاقة)": "2380.SR", "المتقدمة (بتروكيماويات)": "2330.SR",
    "كيان (بتروكيماويات)": "2350.SR", "الدريس (طاقة)": "4200.SR", "البحري (طاقة)": "4030.SR",
    "جرير (تجزئة)": "4190.SR", "إكسترا (تجزئة)": "4003.SR", "أسواق العثيم (تجزئة)": "4001.SR",
    "سينومي ريتيل (تجزئة)": "4240.SR", "ساسكو (تجزئة)": "4050.SR"
}

# ==========================================
# 3. دالة حساب المؤشرات (Trinity Pro) 
# ==========================================
def apply_trinity_pro(df):
    if len(df) < 120: return None
        
    df['KAMA'] = calc_kama(df['Close'], length=100)
    df['HMA'] = calc_hma(df['Close'], length=21)
    
    df['Price_HMA'] = calc_hma(df['Close'], length=9)
    df['RSI_Fast'] = calc_rsi(df['Price_HMA'], length=14)
    df['RSI_Slow'] = calc_rsi(df['Price_HMA'], length=25)

    df['Vol_HMA'] = calc_hma(df['Volume'], length=9)
    df['Is_High_Vol'] = df['Volume'] > df['Vol_HMA']

    df['CMF'] = calc_cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
    df['CMF_Fast'] = calc_hma(df['CMF'].fillna(0), length=9)
    df['CMF_Slow'] = calc_hma(df['CMF'].fillna(0), length=21)
    df['Is_CMF_Bullish'] = df['CMF_Fast'] > df['CMF_Slow']

    high_low_diff = (df['High'] - df['Low']).replace(0, 0.0001) 
    buy_vol = df['Volume'] * (df['Close'] - df['Low']) / high_low_diff
    sell_vol = df['Volume'] * (df['High'] - df['Close']) / high_low_diff
    df['Delta_Filter'] = (buy_vol - sell_vol) > 0

    df['RSI_Filter'] = calc_rsi(df['Close'], length=14) < 70
    df['Is_MTF_Bullish'] = df['Close'] > calc_hma(df['Close'], length=50)

    # شروط الدخول
    df['Uptrend'] = df['Close'] > df['KAMA']
    df['HMA_Turns_Up'] = (df['HMA'] > df['HMA'].shift(1)) & (df['HMA'].shift(1) < df['HMA'].shift(2))
    df['RSI_Cross'] = (df['RSI_Fast'] > df['RSI_Slow']) & (df['RSI_Fast'].shift(1) <= df['RSI_Slow'].shift(1))
    df['Trinity_Buy'] = df['RSI_Cross'] & df['Is_High_Vol'] & df['Is_CMF_Bullish'] & df['Is_MTF_Bullish']
    df['Strong_Buy_Signal'] = df['Uptrend'] & df['HMA_Turns_Up'] & df['Delta_Filter'] & df['RSI_Filter'] & df['Trinity_Buy']

    return df

# ==========================================
# 4. بناء التبويبات (الواجهة)
# ==========================================
tab1, tab2, tab3 = st.tabs(["🚀 ماسح HMA Trinity", "⚡ التاب الثاني (قريباً)", "🔥 التاب الثالث (قريباً)"])

with tab1:
    st.header("ماسح إشارات HMA Trinity Pro")
    st.markdown("يبحث عن الشركات التي حققت إشارة **Strong Buy** بناءً على تقاطع الزخم، السيولة الإيجابية، وحجم التداول الشرائي.")
    
    if st.button("بدء المسح 🔍"):
        with st.spinner("جاري سحب البيانات وتحليل الأسهم..."):
            results = []
            for name, ticker in tickers_dict.items():
                try:
                    df = yf.download(ticker, period="1y", interval="1d", progress=False)
                    if df.empty: continue
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                        
                    df_analyzed = apply_trinity_pro(df.copy())
                    if df_analyzed is not None and df_analyzed.iloc[-1]['Strong_Buy_Signal']:
                        last_row = df_analyzed.iloc[-1]
                        results.append({
                            "الشركة": name, "الرمز": ticker, "الإغلاق": round(last_row['Close'], 2),
                            "السيولة": "إيجابية 🟢", "فوليوم دلتا": "إيجابي 🟢"
                        })
                except Exception:
                    pass
            
            if results:
                st.success("تم العثور على فرص توافق شروطك!")
                st.dataframe(pd.DataFrame(results), use_container_width=True)
            else:
                st.info("لم تحقق أي شركة شروط الدخول اليوم. السوق قد يحتاج إلى مزيد من الزخم.")

with tab2:
    st.info("التاب الثاني جاهز لاستقبال المؤشر القادم.")
