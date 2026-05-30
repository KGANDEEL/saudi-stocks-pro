import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. الدوال الرياضية الفنية (مطابقة لمعادلات TradingView بدقة)
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
# 2. دالة استراتيجية التحليل المتقدمة (Trinity Pro)
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
# 3. قاعدة بيانات الشركات المضافة والقطاعات
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
# 4. بناء الهيكل الرئيسي للواجهة (UI Header)
# ==========================================
st.set_page_config(page_title="منصة التحليل الاحترافية", layout="wide", page_icon="📈")
st.title("لوحة تحليل الأسهم المتقدمة 📊")

# 🟢 هنا تم نقل الاختيار ليصبح مشتركاً لكل التبويبات بالكامل فوق الـ Tabs
col_ui1, col_ui2 = st.columns([2, 2])
with col_ui1:
    company_options = list(tickers_dict.keys()) + ["➕ كتابة رمز مخصص..."]
    selected_option = st.selectbox("اختر الشركة التي تريد تحليلها:", options=company_options, index=0)
    
    if selected_option == "➕ كتابة رمز مخصص...":
        user_ticker = st.text_input("اكتب رمز السهم المخصص هنا (مثال: 1120):", value="1120")
        ticker_symbol = f"{user_ticker}.SR" if (user_ticker and not user_ticker.endswith(".SR") and user_ticker.isdigit()) else user_ticker.upper()
    else:
        ticker_symbol = tickers_dict[selected_option]
        
with col_ui2:
    timeframe_label = st.radio("اختر المدة الزمنية للشارت الفردي:", ["شهر", "سنة", "5 سنوات"], horizontal=True)
    selected_period, selected_interval = ("1mo", "1d") if timeframe_label == "شهر" else ("1y", "1d") if timeframe_label == "سنة" else ("5y", "1wk")

# جلب البيانات المشتركة مرة واحدة لضمان استقرار وسرعة السيرفر
df_global = pd.DataFrame()
if ticker_symbol:
    try:
        fetch_p = "1y" if selected_period == "1mo" else selected_period
        df_global = yf.download(ticker_symbol, period=fetch_p, interval=selected_interval, progress=False)
        if not df_global.empty and isinstance(df_global.columns, pd.MultiIndex):
            df_global.columns = df_global.columns.get_level_values(0)
    except Exception as e:
        st.error(f"حدث خطأ في جلب بيانات السهم المختار: {e}")

# ==========================================
# 5. بناء التبويبات الثلاثة المحدثة
# ==========================================
tab1, tab2, tab3 = st.tabs(["📉 مخطط الشموع ومؤشر HMA", "📊 لوحة الـ 10 مؤشرات + فيبوناتشي", "🚀 ماسح HMA Trinity"])

# ------------------------------------------
# التاب الأول: مخطط الشموع اليابانية التفاعلي
# ------------------------------------------
with tab1:
    st.header("التحليل الفني ومخطط الشموع مع متوسطات HMA")
    st.caption("🔵 الخط الأزرق: HMA السريع (9) | 🟠 الخط البرتقالي: HMA البطيء (21)")
    
    if not df_global.empty:
        df_stock = apply_trinity_pro(df_global.copy())
        df_display = df_stock.tail(30) if selected_period == "1mo" else df_stock
        
        last_price = round(df_display.iloc[-1]['Close'], 2)
        st.metric(label=f"آخر سعر إغلاق للسهم الحالي ({ticker_symbol})", value=f"{last_price} ر.س")
        
        fig = go.Figure(data=[go.Candlestick(
            x=df_display.index, open=df_display['Open'], high=df_display['High'], low=df_display['Low'], close=df_display['Close'],
            name="الشموع", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        )])
        fig.add_trace(go.Scatter(x=df_display.index, y=df_display['Price_HMA'], mode='lines', name='HMA السريع (9)', line=dict(color='#2196f3', width=2)))
        fig.add_trace(go.Scatter(x=df_display.index, y=df_display['HMA'], mode='lines', name='HMA البطيء (21)', line=dict(color='#ff9800', width=2.5)))
        
        fig.update_layout(xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(gridcolor='rgba(200,200,200,0.15)'), yaxis=dict(gridcolor='rgba(200,200,200,0.15)', title="السعر / المؤشر"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("الرجاء التحقق من رمز السهم المحدد بالأعلى.")

# ------------------------------------------
# التاب الثاني: لوحة الـ 10 مؤشرات وفيبوناتشي (تحويل الكود الفني المرسل)
# ------------------------------------------
with tab2:
    st.header("لوحة القياس المتعددة الفنية 📊")
    st.markdown("مبنية ومطابقة تماماً لـ **Pine Script جدول الـ 10 مؤشرات المطور** لحساب التوافق اللحظي وقرب السعر من مستويات فيبوناتشي.")
    
    if not df_global.empty and len(df_global) >= 30:
        df_ind = df_global.copy()
        close_ser = df_ind['Close']
        high_ser = df_ind['High']
        low_ser = df_ind['Low']
        vol_ser = df_ind['Volume']
        
        # حساب المؤشرات اللحظية لآخر شمعة
        rsi_val = calc_rsi(close_ser, 14).iloc[-1]
        macd_l, sig_l = calc_macd(close_ser, 12, 26, 9)
        macd_val, sig_val = macd_l.iloc[-1], sig_l.iloc[-1]
        ema50_val = close_ser.ewm(span=50, adjust=False).mean().iloc[-1]
        sma200_val = close_ser.rolling(window=min(200, len(df_ind))).mean().iloc[-1]
        
        atr_ser = calc_atr(high_ser, low_ser, close_ser, 14)
        atr_val = atr_ser.iloc[-1]
        atr_change = atr_ser.diff().iloc[-1]
        
        cci_val = calc_cci(high_ser, low_ser, close_ser, 20).iloc[-1]
        stoch_k = calc_stoch(high_ser, low_ser, close_ser, 14).iloc[-1]
        
        di_p, di_m, adx_ser = calc_adx_dmi(high_ser, low_ser, close_ser, 14)
        adx_val, dip_val, dim_val = adx_ser.iloc[-1], di_p.iloc[-1], di_m.iloc[-1]
        
        mom_val = (close_ser - close_ser.shift(10)).iloc[-1]
        
        direction = np.sign(close_ser.diff()).fillna(0)
        obv_ser = (direction * vol_ser).cumsum()
        obv_val = obv_ser.iloc[-1]
        obv_change = obv_ser.diff().iloc[-1]
        
        last_close = close_ser.iloc[-1]
        
        # خوارزمية فيبوناتشي (آخر 100 شمعة)
        df_fib = df_ind.tail(100)
        fibHigh = df_fib['High'].max()
        fibLow = df_fib['Low'].min()
        fibRange = (fibHigh - fibLow) if (fibHigh - fibLow) != 0 else 0.0001
        
        f236 = fibLow + fibRange * 0.236
        f382 = fibLow + fibRange * 0.382
        f500 = fibLow + fibRange * 0.500
        f618 = fibLow + fibRange * 0.618
        f786 = fibLow + fibRange * 0.786
        
        diff236 = abs(last_close - f236)
        diff382 = abs(last_close - f382)
        diff500 = abs(last_close - f500)
        diff618 = abs(last_close - f618)
        diff786 = abs(last_close - f786)
        
        minDiff = min(diff236, diff382, diff500, diff618, diff786)
        
        # تحديد مستوى وحالة فيبوناتشي والألوان
        if minDiff == diff618:
            fibLevelStr = f"61.8% ({f618:.2f})"
            fibStatus = "المنطقة الذهبية (دعم قوي)" if last_close >= f618 else "المنطقة الذهبية (مقاومة قوية)"
            fibColor = "#26a69a"
        elif minDiff == diff382:
            fibLevelStr = f"38.2% ({f382:.2f})"
            fibStatus = "دعم فيبوناتشي صامد" if last_close >= f382 else "مقاومة فيبوناتشي قريبة"
            fibColor = "#008080"
        elif minDiff == diff500:
            fibLevelStr = f"50.0% ({f500:.2f})"
            fibStatus = "منطقة اتزان وتذبذب"
            fibColor = "#78909c"
        elif minDiff == diff236:
            fibLevelStr = f"23.6% ({f236:.2f})"
            fibStatus = "قريب من القمة العظمى" if last_close >= f236 else "مقاومة فيبوناتشي أخيرة"
            fibColor = "#ff9800"
        else:
            fibLevelStr = f"78.6% ({f786:.2f})"
            fibStatus = "دعم أخير قبل القاع" if last_close >= f786 else "قريب من القاع السعري"
            fibColor = "#ef5350"
            
        # تحديد حالات باقي المؤشرات الـ 10 والألوان الفنية لها
        rsiStatus = "تشبع شرائي (حذر)" if rsi_val > 70 else "تشبع بيعي (فرصة)" if rsi_val < 30 else "إيجابي" if rsi_val > 50 else "سلبي"
        rsiColor = "#ef5350" if rsi_val > 70 else "#26a69a" if rsi_val < 30 else "#26a69a" if rsi_val > 50 else "#ef5350"
        
        macdStatus = "إيجابي (تقاطع شرائي)" if macd_val > sig_val else "سلبي (تقاطع بيعي)"
        macdColor = "#26a69a" if macd_val > sig_val else "#ef5350"
        
        emaStatus = "اتجاه صاعد (متوسط)" if last_close > ema50_val else "اتجاه هابط (متوسط)"
        emaColor = "#26a69a" if last_close > ema50_val else "#ef5350"
        
        smaStatus = "اتجاه صاعد (رئيسي)" if last_close > sma200_val else "اتجاه هابط (رئيسي)"
        smaColor = "#26a69a" if last_close > sma200_val else "#ef5350"
        
        atrStatus = "تذبذب عالي" if atr_change > 0 else "تذبذب هادئ"
        atrColor = "#ffeb3b" if atr_change > 0 else "#78909c"
        
        cciStatus = "تشبع شرائي" if cci_val > 100 else "تشبع بيعي" if cci_val < -100 else "إيجابي" if cci_val > 0 else "سلبي"
        cciColor = "#ef5350" if cci_val > 100 else "#26a69a" if cci_val < -100 else "#26a69a" if cci_val > 0 else "#ef5350"
        
        stochStatus = "تشبع شرائي" if stoch_k > 80 else "تشبع بيعي" if stoch_k < 20 else "إيجابي" if stoch_k > 50 else "سلبي"
        stochColor = "#ef5350" if stoch_k > 80 else "#26a69a" if stoch_k < 20 else "#26a69a" if stoch_k > 50 else "#ef5350"
        
        adxStatus = ( "صاعد قوي" if dip_val > dim_val else "هابط قوي" ) if adx_val > 25 else "مسار عرضي"
        adxColor = ( "#26a69a" if dip_val > dim_val else "#ef5350" ) if adx_val > 25 else "#78909c"
        
        momStatus = "زخم إيجابي" if mom_val > 0 else "زخم سلبي"
        momColor = "#26a69a" if mom_val > 0 else "#ef5350"
        
        obvStatus = "تجمع (دخول سيولة)" if obv_change > 0 else "تصريف (خروج سيولة)"
        obvColor = "#26a69a" if obv_change > 0 else "#ef5350"
        
        # بناء وعرض جدول HTML الاحترافي لمطابقة بيئة TradingView التكتيكية
        html_table = f"""
        <table style="width:100%; border-collapse: collapse; text-align: right; font-family: Arial;">
            <tr style="background-color: #1e3a8a; color: white; font-weight: bold;">
                <th style="padding: 12px; border: 1px solid #475569;">المؤشر الفني</th>
                <th style="padding: 12px; border: 1px solid #475569;">القراءة اللحظية / المستوى</th>
                <th style="padding: 12px; border: 1px solid #475569;">الحالة والتحليل الفني</th>
            </tr>
            <tr style="background-color: #0f172a; color: white;">
                <td style="padding: 10px; border: 1px solid #334155; font-weight: bold;">Fibonacci (100)</td>
                <td style="padding: 10px; border: 1px solid #334155;">{fibLevelStr}</td>
                <td style="padding: 10px; border: 1px solid #334155; color: {fibColor}; font-weight: bold;">{fibStatus}</td>
            </tr>
            <tr style="background-color: #1e293b; color: white;">
                <td style="padding: 10px; border: 1px solid #334155;">RSI (14)</td>
                <td style="padding: 10px; border: 1px solid #334155;">{rsi_val:.2f}</td>
                <td style="padding: 10px; border: 1px solid #334155; color: {rsiColor}; font-weight: bold;">{rsiStatus}</td>
            </tr>
            <tr style="background-color: #1e293b; color: white;">
                <td style="padding: 10px; border: 1px solid #334155;">MACD</td>
                <td style="padding: 10px; border: 1px solid #334155;">{macd_val:.2f}</td>
                <td style="padding: 10px; border: 1px solid #334155; color: {macdColor}; font-weight: bold;">{macdStatus}</td>
            </tr>
            <tr style="background-color: #1e293b; color: white;">
                <td style="padding: 10px; border: 1px solid #334155;">EMA 50</td>
                <td style="padding: 10px; border: 1px solid #334155;">{ema50_val:.2f}</td>
                <td style="padding: 10px; border: 1px solid #334155; color: {emaColor}; font-weight: bold;">{emaStatus}</td>
            </tr>
            <tr style="background-color: #1e293b; color: white;">
                <td style="padding: 10px; border: 1px solid #334155;">SMA 200</td>
                <td style="padding: 10px; border: 1px solid #334155;">{sma200_val:.2f}</td>
                <td style="padding: 10px; border: 1px solid #334155; color: {smaColor}; font-weight: bold;">{smaStatus}</td>
            </tr>
            <tr style="background-color: #1e293b; color: white;">
                <td style="padding: 10px; border: 1px solid #334155;">ATR (14)</td>
                <td style="padding: 10px; border: 1px solid #334155;">{atr_val:.2f}</td>
                <td style="padding: 10px; border: 1px solid #334155; color: {atrColor}; font-weight: bold;">{atrStatus}</td>
            </tr>
            <tr style="background-color: #1e293b; color: white;">
                <td style="padding: 10px; border: 1px solid #334155;">CCI (20)</td>
                <td style="padding: 10px; border: 1px solid #334155;">{cci_val:.2f}</td>
                <td style="padding: 10px; border: 1px solid #334155; color: {cciColor}; font-weight: bold;">{cciStatus}</td>
            </tr>
            <tr style="background-color: #1e293b; color: white;">
                <td style="padding: 10px; border: 1px solid #334155;">Stochastic</td>
                <td style="padding: 10px; border: 1px solid #334155;">{stoch_k:.2f}</td>
                <td style="padding: 10px; border: 1px solid #334155; color: {stochColor}; font-weight: bold;">{stochStatus}</td>
            </tr>
            <tr style="background-color: #1e293b; color: white;">
                <td style="padding: 10px; border: 1px solid #334155;">ADX / DMI</td>
                <td style="padding: 10px; border: 1px solid #334155;">{adx_val:.2f}</td>
                <td style="padding: 10px; border: 1px solid #334155; color: {adxColor}; font-weight: bold;">{adxStatus}</td>
            </tr>
            <tr style="background-color: #1e293b; color: white;">
                <td style="padding: 10px; border: 1px solid #334155;">Momentum (10)</td>
                <td style="padding: 10px; border: 1px solid #334155;">{mom_val:.2f}</td>
                <td style="padding: 10px; border: 1px solid #334155; color: {momColor}; font-weight: bold;">{momStatus}</td>
            </tr>
            <tr style="background-color: #1e293b; color: white;">
                <td style="padding: 10px; border: 1px solid #334155;">OBV</td>
                <td style="padding: 10px; border: 1px solid #334155;">{obv_val:.2f}</td>
                <td style="padding: 10px; border: 1px solid #334155; color: {obvColor}; font-weight: bold;">{obvStatus}</td>
            </tr>
        </table>
        """
        st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.warning("البيانات التاريخية المتوفرة للسهم غير كافية لحساب كافة المؤشرات الـ 10 (تحتاج 200 شمعة على الأقل).")

# ------------------------------------------
# التاب الثالث: ماسح السوق التلقائي (نقل ذكي للمحافظة على الترتيب)
# ------------------------------------------
with tab3:
    st.header("ماسح إشارات HMA Trinity Pro التلقائي")
    st.markdown(f"يقوم هذا الماسح بفحص **كافة الشركات القيادية الـ {len(tickers_dict)} المضافة** دفعة واحدة للبحث عن الأسهم ذات الإشارات الإيجابية اليوم.")
    
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
