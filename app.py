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
    "جرير (تجزئة)": "4190.SR", "إكسترا (تجزئة)": "4003.SR", "أسواق العثيم (تجزئة أغذية)": "4001.SR", "سينومي ريتيل (تجزئة)": "4240.SR", "ساسكو (تجزئة طاقة)": "4050.SR", "المراعي (إنتاج أغذية)": "2280.SR", "نادك (إنتاج أغذية)": "2270.SR", "صافولا (إنتاج أغذية)": "2050.SR", "النهدي (صيدليات وتجزئة)": "4164.SR",
    "سليمان الحبيب (رعاية صحية)": "4013.SR", "دله الصحية (رعاية صحية)": "4004.SR", "المواساة (رعاية صحية)": "4002.SR",
    "دار الأركان (عقارات)": "4300.SR", "جبل عمر (عقارات)": "4250.SR", "بوبا العربية (تأمين)": "8210.SR", "التعاونية (تأمين)": "8010.SR", "سيرا (سياحة)": "1810.SR", "وقت اللياقة - لجام (ترفيه)": "1830.SR"
}

# ==========================================
# 5. سحب البيانات الجماعي والتحميل المسبق الفوري لجميع الشركات
# ==========================================
@st.cache_data(ttl=600)  # تخزين مؤقت لمدة 10 دقائق لضمان السرعة القصوى
def load_bulk_market_data(tickers_list):
    return yf.download(tickers_list, period="1y", interval="1d", progress=False)

with st.spinner("جاري سحب قراءات السوق لجميع الشركات الآن... 🔄"):
    df_market_raw = load_bulk_market_data(list(tickers_dict.values()))

# دالة ذكية لفصل تيكر محدد من البيانات المجمعة بدون إرسال طلبات نت جديدة
def extract_ticker_data(df_all, ticker):
    try:
        return pd.DataFrame({
            'Open': df_all['Open'][ticker], 'High': df_all['High'][ticker],
            'Low': df_all['Low'][ticker], 'Close': df_all['Close'][ticker],
            'Volume': df_all['Volume'][ticker]
        }).dropna()
    except:
        return pd.DataFrame()

# ==========================================
# 6. دالة تفكيك الـ Pine Script لحساب سطر المؤشرات الفنية لكل شركة
# ==========================================
def compute_indicators_row(df_t, company_name):
    if len(df_t) < 30: return None
    close_s, high_s, low_s, vol_s = df_t['Close'], df_t['High'], df_t['Low'], df_t['Volume']
    last_close = close_s.iloc[-1]
    
    # 1. HMA (9)
    hma9_series = calc_hma(close_s, 9)
    hma9_val = hma9_series.iloc[-1]
    hma9_status = "🟢 صاعد" if last_close > hma9_val else "🔴 هابط"
    
    # 2. RSI (14)
    rsi_val = calc_rsi(close_s, 14).iloc[-1]
    rsi_status = "🔴 مفرط شراء" if rsi_val > 70 else "🟢 مفرط بيع" if rsi_val < 30 else "🟢 إيجابي" if rsi_val > 50 else "🔴 سلبي"
    
    # 3. MACD
    ml, sl = calc_macd(close_s, 12, 26, 9)
    macd_status = "🟢 تقاطع شرائي" if ml.iloc[-1] > sl.iloc[-1] else "🔴 تقاطع بيعي"
    
    # 4. EMA 50
    ema50 = close_s.ewm(span=50, adjust=False).mean().iloc[-1]
    ema_status = "🟢 فوق المتوسط" if last_close > ema50 else "🔴 تحت المتوسط"
    
    # 5. SMA 200
    sma200 = close_s.rolling(window=min(200, len(df_t))).mean().iloc[-1]
    sma_status = "🟢 اتجاه صاعد" if last_close > sma200 else "🔴 اتجاه هابط"
    
    # 6. ATR (14)
    atr_s = calc_atr(high_s, low_s, close_s, 14)
    atr_status = "⚡ تذبذب عالي" if atr_s.diff().iloc[-1] > 0 else "💤 تذبذب هادئ"
    
    # 7. CCI (20)
    cci_val = calc_cci(high_s, low_s, close_s, 20).iloc[-1]
    cci_status = "🔴 مفرط شراء" if cci_val > 100 else "🟢 مفرط بيع" if cci_val < -100 else "🟢 إيجابي" if cci_val > 0 else "🔴 سلبي"
    
    # 8. Stochastic
    stoch_k = calc_stoch(high_s, low_s, close_s, 14).iloc[-1]
    stoch_status = "🔴 تشبع شرائي" if stoch_k > 80 else "🟢 تشبع بيعي" if stoch_k < 20 else "🟢 إيجابي" if stoch_k > 50 else "🔴 سلبي"
    
    # 9. ADX / DMI
    dp, dm, adx_s = calc_adx_dmi(high_s, low_s, close_s, 14)
    adx_val = adx_s.iloc[-1]
    adx_status = ("🟢 اتجاه قوي" if dp.iloc[-1] > dm.iloc[-1] else "🔴 هبوط قوي") if adx_val > 25 else "🟡 مسار عرضي"
    
    # 10. Momentum
    mom_val = (close_s - close_s.shift(10)).iloc[-1]
    mom_status = "🟢 زخم إيجابي" if mom_val > 0 else "🔴 زخم سلبي"
    
    # 11. OBV
    obv_s = (np.sign(close_s.diff()).fillna(0) * vol_s).cumsum()
    obv_status = "🟢 تجميع" if obv_s.diff().iloc[-1] > 0 else "🔴 تصريف"
    
    return {
        "اسم الشركة": company_name, "الإغلاق": round(last_close, 2),
        "مؤشر HMA (9)": f"{hma9_val:.2f} ({hma9_status})", "RSI (14)": f"{rsi_val:.1f} ({rsi_status})",
        "MACD": macd_status, "EMA 50": f"{ema50:.2f} ({ema_status})", "SMA 200": f"{sma200:.2f} ({sma_status})",
        "ATR": atr_status, "CCI (20)": f"{cci_val:.1f} ({cci_status})", "Stochastic": f"{stoch_k:.1f} ({stoch_status})",
        "ADX": adx_status, "Momentum": mom_status, "OBV": obv_status
    }

# ==========================================
# 7. بناء الواجهة الرسومية وهيكل الـ Tabs المشتركة
# ==========================================
col_header1, col_header2 = st.columns([2, 2])
with col_header1:
    company_options = list(tickers_dict.keys()) + ["➕ كتابة رمز مخصص..."]
    selected_option = st.selectbox("اختر السهم الفردي للشارت (التاب 1):", options=company_options, index=0)
    
    if selected_option == "➕ كتابة رمز مخصص...":
        user_ticker = st.text_input("اكتب رمز السهم المخصص هنا (مثال: 1120):", value="1120")
        ticker_symbol = f"{user_ticker}.SR" if (user_ticker and not user_ticker.endswith(".SR") and user_ticker.isdigit()) else user_ticker.upper()
    else:
        ticker_symbol = tickers_dict[selected_option]

with col_header2:
    timeframe_label = st.radio("المدة الزمنية للشارت الفردي:", ["شهر", "سنة"], horizontal=True)
    selected_period = "1mo" if timeframe_label == "شهر" else "1y"

# استخراج بيانات السهم المختار للشارت الفردي
if ticker_symbol in tickers_dict.values():
    df_global = extract_ticker_data(df_market_raw, ticker_symbol)
else:
    try:
        df_global = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if not df_global.empty and isinstance(df_global.columns, pd.MultiIndex):
            df_global.columns = df_global.columns.get_level_values(0)
    except:
        df_global = pd.DataFrame()

# إنشاء التبويبات الفنية الثلاثة
tab1, tab2, tab3 = st.tabs(["📉 شارت الشموع الفردي", "📊 لوحة المؤشرات الـ 10 لجميع الشركات", "🚀 ماسح صفقات Trinity Pro"])

# ------------------------------------------
# التاب الأول: الشارت الفردي التفاعلي لمتابعة تفاصيل سهم محدد
# ------------------------------------------
with tab1:
    st.header(f"التحليل البصري للسهم المختار ({ticker_symbol})")
    if not df_global.empty:
        df_stock = apply_trinity_pro(df_global.copy())
        df_display = df_stock.tail(30) if selected_period == "1mo" else df_stock
        
        st.metric(label="آخر سعر إغلاق للسهم المختار الحالي", value=f"{round(df_display.iloc[-1]['Close'], 2)} ر.س")
        
        fig = go.Figure(data=[go.Candlestick(
            x=df_display.index, open=df_display['Open'], high=df_display['High'], low=df_display['Low'], close=df_display['Close'],
            name="الشموع", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        )])
        fig.add_trace(go.Scatter(x=df_display.index, y=df_display['Price_HMA'], mode='lines', name='HMA السريع (9)', line=dict(color='#2196f3', width=2)))
        fig.add_trace(go.Scatter(x=df_display.index, y=df_display['HMA'], mode='lines', name='HMA البطيء (21)', line=dict(color='#ff9800', width=2.5)))
        
        fig.update_layout(xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(gridcolor='rgba(200,200,200,0.15)'), yaxis=dict(gridcolor='rgba(200,200,200,0.15)'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("الرجاء اختيار أو كتابة سهم صحيح لعرض الشارت الخاص به.")

# ------------------------------------------
# التاب الثاني: مصفوفة المؤشرات الـ 10 لكافة الشركات دفعة واحدة
# ------------------------------------------
with tab2:
    st.header("مصفوفة الرصد الرقمي الموحد لجميع شركات السوق 📊")
    st.markdown("تعرض هذه اللوحة **كل الشركات المدرجة دفعة واحدة** جنباً إلى جنب مع قيم وحالات المؤشرات الفنية الـ 10 الأساسية، بالإضافة لمؤشر HMA (9) السريع.")
    
    # بناء مصفوفة البيانات الضخمة
    matrix_rows = []
    for name, ticker in tickers_dict.items():
        df_t = extract_ticker_data(df_market_raw, ticker)
        if not df_t.empty:
            row_res = compute_indicators_row(df_t, name)
            if row_res:
                matrix_rows.append(row_res)
                
    if matrix_rows:
        df_matrix = pd.DataFrame(matrix_rows)
        
        # عرض الجدول بشكل احترافي مع ميزات البحث، التصفية، والتكبير المدمجة في سترمليت
        st.dataframe(
            df_matrix, 
            use_container_width=True, 
            height=650, 
            hide_index=True
        )
        st.caption("💡 يمكنك الضغط على رأس أي عمود لترتيب السوق بأكمله تنازلياً أو تصاعدياً بناءً على ذلك المؤشر، أو استخدام مربع البحث بالداخل لفلترة قطاع معين.")
    else:
        st.error("فشل في معالجة مصفوفة الشركات، يرجى إعادة تحديث الصفحة.")

# ------------------------------------------
# التاب الثالث: ماسح الصفقات التلقائي السريع لشروط التوافق الصارمة
# ------------------------------------------
with tab3:
    st.header("فلتر الاستراتيجية الآلي (Trinity Pro Signals)")
    st.markdown("يبحث هذا الفلتر في السوق بالكامل عن الأسهم التي تعطي إشارة دخول توافقية مؤكدة بالكامل (Strong Buy) في اللحظة الحالية.")
    
    if st.button("تحديث ومسح السوق فوراً 🔍"):
        scan_results = []
        for name, ticker in tickers_dict.items():
            df_t = extract_ticker_data(df_market_raw, ticker)
            if df_t.empty: continue
            
            df_anal = apply_trinity_pro(df_t.copy())
            if df_anal is not None and not df_anal.empty and df_anal.iloc[-1]['Strong_Buy_Signal']:
                last_row = df_anal.iloc[-1]
                scan_results.append({
                    "الشركة": name, "رمز السهم": ticker, "سعر الإغلاق": round(last_row['Close'], 2),
                    "RSI": round(last_row['RSI_Val'], 2), "السيولة (CMF)": "إيجابية وتجميع 🟢", "حالة الاتجاه": "صاعد قوي 📈"
                })
                
        if scan_results:
            st.success(f"تم رصد {len(scan_results)} شركات تخترق الشروط الصارمة للدخول الآمن اليوم!")
            st.dataframe(pd.DataFrame(scan_results), use_container_width=True, hide_index=True)
        else:
            st.info("لا توجد شركات تعطي اختراقاً كاملاً للشروط الصارمة اليوم. يفضل الانتظار ومراقبة المصفوفة بالتاب الثاني.")
