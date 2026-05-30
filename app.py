import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
# 2. دالة حساب المؤشرات (Trinity Pro) 
# ==========================================
def apply_trinity_pro(df):
    if len(df) < 30:
        return df
        
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

    # شروط الدخول
    df['Uptrend'] = df['Close'] > df['KAMA']
    df['HMA_Turns_Up'] = (df['HMA'] > df['HMA'].shift(1)) & (df['HMA'].shift(1) < df['HMA'].shift(2))
    df['RSI_Cross'] = (df['RSI_Fast'] > df['RSI_Slow']) & (df['RSI_Fast'].shift(1) <= df['RSI_Slow'].shift(1))
    df['Trinity_Buy'] = df['RSI_Cross'] & df['Is_High_Vol'] & df['Is_CMF_Bullish'] & df['Is_MTF_Bullish']
    df['Strong_Buy_Signal'] = df['Uptrend'] & df['HMA_Turns_Up'] & df['Delta_Filter'] & df['RSI_Filter'] & df['Trinity_Buy']

    return df

# ==========================================
# 3. قائمة الشركات الموسعة والشاملة لكافة القطاعات
# ==========================================
tickers_dict = {
    # 🛢️ قطاع الطاقة والمرافق
    "أرامكو (طاقة)": "2222.SR", 
    "لوبريف (طاقة)": "2223.SR",
    "الدريس (طاقة)": "4200.SR", 
    "البحري (طاقة)": "4030.SR",
    "كهرباء السعودية (مرافق عامة)": "5110.SR",
    "غازكو (مرافق عامة)": "2080.SR",
    # 🏗️ قطاع المواد الأساسية (البتروكيماويات والأسمنت)
    "سابك (مواد أساسية)": "2010.SR", 
    "معادن (مواد أساسية)": "1211.SR",
    "ينساب (مواد أساسية)": "2230.SR", 
    "سبكيم العالمية (مواد أساسية)": "2310.SR", 
    "بترورابغ (مواد أساسية)": "2380.SR", 
    "المتقدمة (مواد أساسية)": "2330.SR", 
    "كيان السعودية (مواد أساسية)": "2350.SR",
    "التصنيع الوطنية (مواد أساسية)": "2060.SR",
    "أسمنت اليمامة (مواد أساسية)": "3020.SR",
    "أسمنت السعودية (مواد أساسية)": "3030.SR",
    # 🏦 قطاع البنوك والخدمات المالية
    "الراجحي (بنوك)": "1120.SR", 
    "الأهلي (بنوك)": "1180.SR", 
    "الإنماء (بنوك)": "1150.SR", 
    "البلاد (بنوك)": "1140.SR", 
    "بنك الرياض (بنوك)": "1010.SR",
    "بنك الجزيرة (بنوك)": "1020.SR",
    "مجموعة تداول (خدمات مالية)": "1111.SR",
    # 📞 قطاع الاتصالات والتقنية
    "اس تي سي STC (اتصالات)": "7010.SR", 
    "موبايلي (اتصالات)": "7020.SR", 
    "زين السعودية (اتصالات)": "7030.SR",
    "سلوشنز (تقنية)": "7200.SR",
    "علم (تقنية)": "7203.SR",
    # 🛒 قطاع التجزئة وإنتاج الأغذية
    "جرير (تجزئة)": "4190.SR", 
    "إكسترا (تجزئة)": "4003.SR", 
    "أسواق العثيم (تجزئة أغذية)": "4001.SR",
    "سينومي ريتيل (تجزئة)": "4240.SR", 
    "ساسكو (تجزئة طاقة)": "4050.SR", 
    "المراعي (إنتاج أغذية)": "2280.SR",
    "نادك (إنتاج أغذية)": "2270.SR",
    "صافولا (إنتاج أغذية)": "2050.SR",
    "النهدي (صيدليات وتجزئة)": "4164.SR",
    # 🩺 قطاع الرعاية الصحية
    "سليمان الحبيب (رعاية صحية)": "4013.SR", 
    "دله الصحية (رعاية صحية)": "4004.SR",
    "المواساة (رعاية صحية)": "4002.SR",
    # 🏢 قطاع العقارات والتطوير
    "دار الأركان (عقارات)": "4300.SR", 
    "جبل عمر (عقارات)": "4250.SR",
    # 🛡️ قطاع التأمين والترفيه والسياحة
    "بوبا العربية (تأمين)": "8210.SR",
    "التعاونية (تأمين)": "8010.SR",
    "سيرا (سياحة وسفر)": "1810.SR",
    "وقت اللياقة - لجام (ترفيه)": "1830.SR"
}

# ==========================================
# 4. بناء التبويبات والواجهة الرئيسية
# ==========================================
st.set_page_config(page_title="منصة التحليل الاحترافية", layout="wide", page_icon="📈")
st.title("لوحة تحليل الأسهم المتقدمة 📊")

tab1, tab2, tab3 = st.tabs(["📉 مخطط الشموع ومؤشر HMA", "🚀 ماسح HMA Trinity", "⚡ التاب القادم (قريباً)"])

# ------------------------------------------
# التاب الأول: الشارت الفردي بالشموع والقائمة المنسدلة
# ------------------------------------------
with tab1:
    st.header("التحليل الفني ومخطط الشموع مع متوسطات HMA")
    st.caption("🔵 الخط الأزرق: HMA السريع (9) | 🟠 الخط البرتقالي: HMA البطيء (21)")
    
    col1, col2 = st.columns([2, 2])
    
    with col1:
        # 🔴 هنا تحويل الخانة المكتوبة إلى Dropdown (قائمة منسدلة ذكية)
        company_options = list(tickers_dict.keys()) + ["➕ كتابة رمز مخصص..."]
        selected_option = st.selectbox("اختر الشركة التي تريد تحليلها:", options=company_options, index=0)
        
        # معالجة المدخلات بناءً على اختيار المستخدم
        if selected_option == "➕ كتابة رمز مخصص...":
            user_ticker = st.text_input("اكتب رمز السهم المخصص هنا (مثال: 1120 أو 2010):", value="1120")
            if user_ticker and not user_ticker.endswith(".SR") and user_ticker.isdigit():
                ticker_symbol = f"{user_ticker}.SR"
            else:
                ticker_symbol = user_ticker.upper()
        else:
            ticker_symbol = tickers_dict[selected_option]
            
    with col2:
        timeframe_label = st.radio("اختر المدة الزمنية للشارت:", ["شهر", "سنة", "5 سنوات"], horizontal=True)
        if timeframe_label == "شهر":
            selected_period, selected_interval = "1mo", "1d"
        elif timeframe_label == "سنة":
            selected_period, selected_interval = "1y", "1d"
        else:
            selected_period, selected_interval = "5y", "1wk"

    if ticker_symbol:
        with st.spinner(f"جاري جلب بيانات السهم {ticker_symbol}..."):
            try:
                fetch_period = "1y" if selected_period == "1mo" else selected_period
                df_stock = yf.download(ticker_symbol, period=fetch_period, interval=selected_interval, progress=False)
                
                if not df_stock.empty:
                    if isinstance(df_stock.columns, pd.MultiIndex):
                        df_stock.columns = df_stock.columns.get_level_values(0)
                        
                    df_stock = apply_trinity_pro(df_stock)
                    
                    if selected_period == "1mo":
                        df_display = df_stock.tail(30)
                    else:
                        df_display = df_stock
                    
                    last_price = round(df_display.iloc[-1]['Close'], 2)
                    st.metric(label=f"آخر سعر إغلاق للسهم الحالي ({ticker_symbol})", value=f"{last_price} ر.س")
                    
                    st.subheader("مخطط الشموع ومؤشر HMA التفاعلي 📊")
                    
                    fig = go.Figure(data=[go.Candlestick(
                        x=df_display.index,
                        open=df_display['Open'],
                        high=df_display['High'],
                        low=df_display['Low'],
                        close=df_display['Close'],
                        name="الشموع",
                        increasing_line_color='#26a69a',
                        decreasing_line_color='#ef5350'
                    )])
                    
                    fig.add_trace(go.Scatter(
                        x=df_display.index,
                        y=df_display['Price_HMA'],
                        mode='lines',
                        name='HMA السريع (9)',
                        line=dict(color='#2196f3', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df_display.index,
                        y=df_display['HMA'],
                        mode='lines',
                        name='HMA البطيء (21)',
                        line=dict(color='#ff9800', width=2.5)
                    ))
                    
                    fig.update_layout(
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='rgba(200,200,200,0.15)'),
                        yaxis=dict(gridcolor='rgba(200,200,200,0.15)', title="السعر / المؤشر")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # تقرير الوضع الحالي للمؤشرات
                    st.subheader("التقرير الفني اللحظي (Trinity Pro) 📋")
                    last_row = df_stock.iloc[-1]
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write(f"**مؤشر RSI:** {round(last_row.get('RSI_Val', 50), 2)}")
                        if last_row.get('RSI_Val', 50) > 70:
                            st.warning("تشبع شرائي ⚠️")
                        elif last_row.get('RSI_Val', 50) < 30:
                            st.success("تشبع بيعي (فرصة ارتداد) 🟢")
                        else:
                            st.write("منطقة محايدة ⚖️")
                            
                    with c2:
                        st.write("**حالة السيولة (CMF):**")
                        if last_row.get('Is_CMF_Bullish', False):
                            st.success("السيولة إيجابية وتدخل السهم 🟢")
                        else:
                            st.error("السيولة ضعيفة أو تخرج من السهم 🔴")
                            
                    with c3:
                        st.write("**إشارة الدخول:**")
                        if last_row.get('Strong_Buy_Signal', False):
                            st.button("🔥 إشارة شراء قوية (Strong Buy)!", disabled=True)
                        else:
                            st.info("لا توجد إشارة دخول مؤكدة حالياً.")
                            
                else:
                    st.error("لم نتمكن من العثور على بيانات لهذا الرمز. تأكد من كتابة الرقم الصحيح.")
            except Exception as e:
                st.error(f"حدث خطأ أثناء جلب البيانات: {e}")

# ------------------------------------------
# التاب الثاني: ماسح السوق التلقائي
# ------------------------------------------
with tab2:
    st.header("ماسح إشارات HMA Trinity Pro التلقائي")
    st.markdown(f"يقوم هذا الماسح بفحص **كافة الشركات القيادية الـ {len(tickers_dict)} المضافة** دفعة واحدة للبحث عن الأسهم ذات الإشارات الإيجابية.")
    
    if st.button("بدء مسح السوق الموسع 🔍"):
        with st.spinner(f"جاري سحب بيانات {len(tickers_dict)} شركة وتحليلها..."):
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
                            "الشركة": name, "الرمز": ticker, "الإغلاق الحالي": round(last_row['Close'], 2),
                            "مؤشر RSI": round(last_row['RSI_Val'], 2), "السيولة": "إيجابية 🟢", "فوليوم دلتا": "إيجابي 🟢"
                        })
                except Exception:
                    pass
            
            if results:
                st.success(f"تم العثور على {len(results)} فرص واعدة توافق شروطك الفنية تماماً!")
                st.dataframe(pd.DataFrame(results), use_container_width=True)
            else:
                st.info("لم تحقق أي شركة شروط الدخول الصارمة (Strong Buy) اليوم. انتظر تماسك السوق.")

# ------------------------------------------
# التاب الثالث: للاستخدام المستقبلي
# ------------------------------------------
with tab3:
    st.info("هذا التاب جاهز لاستقبال المؤشر القادم.")
