import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import requests 
import inspect
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI
import altair as alt
import graphviz
from datetime import datetime
from PIL import Image

# ==============================================================================
# 0. [CRITICAL FIX] TimeSHAP Altair Theme Error Patch
# ==============================================================================
def placeholder_theme():
    return {"config": {}}

if "feedzai" not in alt.themes.names():
    alt.themes.register("feedzai", placeholder_theme)
    alt.themes.enable("feedzai")

# ------------------------------------------------------------------------------
# 1. Page Config & TOBIT Theme CSS
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')

try:
    if os.path.exists(LOGO_PATH):
        icon_img = Image.open(LOGO_PATH)
    else:
        icon_img = "🪙"
except Exception:
    icon_img = "🪙"

st.set_page_config(
    page_title="TOBIT | Bitcoin Forecast",
    page_icon=icon_img, 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Roboto+Mono:wght@400;700&display=swap');
    
    .stApp { background-color: #0b0e11; font-family: 'Inter', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #11141a; border-right: 1px solid #262a33; }
    
    header {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
    
    .kpi-card {
        background: linear-gradient(145deg, #161b22, #11141a); border: 1px solid #262a33;
        border-radius: 12px; padding: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .kpi-card:hover { border-color: #58a6ff; transform: translateY(-2px); }
    
    .kpi-label { font-size: 0.8rem; color: #8b949e; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { font-family: 'Roboto Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #e6edf3; }
    .kpi-delta { font-family: 'Roboto Mono', monospace; font-size: 0.85rem; margin-top: 5px; font-weight: 600; }
    
    .text-green { color: #3fb950; } .text-red { color: #f85149; } .text-blue { color: #58a6ff; } .text-gold { color: #d29922; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #262a33; padding-bottom: 5px; }
    .stTabs [data-baseweb="tab"] { height: 40px; background-color: transparent; border: 1px solid transparent; color: #8b949e; font-weight: 600; border-radius: 6px; font-size: 0.9rem; }
    .stTabs [aria-selected="true"] { background-color: #1f242c; color: #58a6ff; border: 1px solid #262a33; }
    
    code { font-family: 'Roboto Mono', monospace !important; }
    .plot-container { background-color: transparent !important; }
    
    .ai-chat-box {
        background-color: #1f242c; border: 1px solid #58a6ff;
        border-radius: 10px; padding: 15px; margin-top: 15px;
        color: #e6edf3; font-size: 0.95rem; line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. API Key Setup
# ------------------------------------------------------------------------------
if "UPSTAGE_API_KEY" in st.secrets:
    UPSTAGE_API_KEY = st.secrets["UPSTAGE_API_KEY"]
else:
    UPSTAGE_API_KEY = "dummy_key"

BASE_URL = "https://api.upstage.ai/v1"
client = OpenAI(api_key=UPSTAGE_API_KEY, base_url=BASE_URL)

# ------------------------------------------------------------------------------
# 3. Import Dependencies
# ------------------------------------------------------------------------------
try:
    from timeshap.explainer import local_pruning, local_event, local_feat, local_cell_level
except ImportError: pass 
except Exception as e: st.warning(f"TimeSHAP 초기화 경고: {e}")

try:
    from model import LSTMModel, DLinear, PatchTST, iTransformer, TCN, MLP
    from data_utils import fetch_multi_data, load_scaler, TICKERS, send_discord_message
except ImportError as e:
    st.error(f"🚨 필수 모듈 임포트 실패: {e}")
    st.stop()

# ------------------------------------------------------------------------------
# 4. Helper Functions
# ------------------------------------------------------------------------------
def get_pruning_plot(plot_data, pruning_idx, title="Pruning Plot"):
    if plot_data is None: return None
    df_plot = pd.DataFrame([{'Index': item[1], 'Value': item[2]} for item in plot_data]) if isinstance(plot_data, list) else plot_data.copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0b0e11')
    ax.set_facecolor('#0b0e11')
    ax.spines['bottom'].set_color('#8b949e'); ax.spines['left'].set_color('#8b949e')
    ax.tick_params(colors='#8b949e', labelsize=10)
    ax.fill_between(df_plot.iloc[:, 1], df_plot.iloc[:, 2], color='#58a6ff', alpha=0.6)
    ax.axvline(x=pruning_idx, color='#f85149', linestyle='-', linewidth=1.5)
    ax.set_title(title, fontsize=12, loc='left', color='#e6edf3')
    sns.despine()
    return fig

def get_event_heatmap(df, title):
    if df is None or df.empty: return None
    df_plot = df.copy()
    if 'Feature' not in df_plot.columns: df_plot['Feature'] = df_plot.index
    try:
        df_plot['sort_key'] = df_plot['Feature'].str.extract(r'([-]?\d+)').astype(int)
        df_plot = df_plot.sort_values('sort_key', ascending=False).drop(columns=['sort_key'])
    except: pass
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_plot.pivot_table(index='Feature', values='Shapley Value'), cmap='coolwarm', center=0, annot=True, fmt=".3f", ax=ax, cbar=False, annot_kws={"size": 10})
    ax.set_title(title, fontsize=12, color='#e6edf3'); ax.set_ylabel(""); ax.tick_params(axis='y', colors='#8b949e', labelsize=10); ax.set_xticks([])
    return fig

def get_feature_bar(df, title):
    if df is None or df.empty: return None
    df_plot = df.copy()
    df_plot['abs_val'] = df_plot['Shapley Value'].abs()
    df_plot = df_plot.sort_values(by='abs_val', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0b0e11'); ax.set_facecolor('#0b0e11')
    ax.spines['bottom'].set_color('#8b949e'); ax.spines['left'].set_color('#8b949e')
    ax.tick_params(colors='#8b949e', labelsize=10)
    sns.barplot(x='Shapley Value', y='Feature', data=df_plot, color='#58a6ff', ax=ax)
    ax.axvline(x=0, color='gray', linewidth=0.8); ax.set_title(title, fontsize=12, loc='left', color='#e6edf3'); ax.set_ylabel("")
    return fig

def get_cell_heatmap(cell_df, title):
    if cell_df is None or cell_df.empty: return None
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cell_df.pivot(index='Feature', columns='Event', values='Shapley Value'), cmap='coolwarm', center=0, annot=True, fmt=".3f", ax=ax, cbar=False, annot_kws={"size": 9})
    ax.set_title(title, fontsize=12, color='#e6edf3'); ax.tick_params(colors='#8b949e', labelsize=9); ax.set_xlabel(""); ax.set_ylabel("")
    return fig

# ------------------------------------------------------------------------------
# 5. Model Logic
# ------------------------------------------------------------------------------
MODELS_LIST = ["MLP", "DLinear", "TCN", "LSTM", "PatchTST", "iTransformer"]
MODEL_CLASSES = {"MLP": MLP, "DLinear": DLinear, "TCN": TCN, "LSTM": LSTMModel, "PatchTST": PatchTST, "iTransformer": iTransformer}

@st.cache_resource
def get_model(name, seq_len):
    input_size = len(TICKERS)
    pred_len = 7
    if name == "MLP": model = MLP(seq_len=seq_len, input_size=input_size, pred_len=pred_len)
    elif name == "DLinear": model = DLinear(seq_len=seq_len, pred_len=pred_len, input_size=input_size, kernel_size=25)
    elif name == "TCN": model = TCN(input_size=input_size, output_size=pred_len, num_channels=[64, 64, 64], kernel_size=3, dropout=0.2)
    elif name == "LSTM": model = LSTMModel(input_size=input_size, output_size=pred_len)
    elif name == "PatchTST": model = PatchTST(input_size=input_size, seq_len=seq_len, pred_len=pred_len, patch_len=7, stride=3, d_model=64, n_heads=4, n_layers=2, dropout=0.2)
    elif name == "iTransformer": model = iTransformer(seq_len=seq_len, pred_len=pred_len, input_size=input_size, d_model=256, n_heads=4, n_layers=3, dropout=0.2)
    
    path = os.path.join(WEIGHTS_DIR, f"{name}_{seq_len}.pth")
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location='cpu'))
        except: pass
    model.eval()
    return model

# ------------------------------------------------------------------------------
# Data Loading with Cache (File-based)
# ------------------------------------------------------------------------------
DATA_CACHE_PATH = os.path.join(BASE_DIR, "daily_btc_data.csv")

def get_data_with_cache():
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # 1. 파일이 있고, 오늘 날짜인지 확인
    if os.path.exists(DATA_CACHE_PATH):
        try:
            file_time = datetime.fromtimestamp(os.path.getmtime(DATA_CACHE_PATH)).strftime("%Y-%m-%d")
            if file_time == today_str:
                df = pd.read_csv(DATA_CACHE_PATH, index_col=0, parse_dates=True)
                if len(df) > 10: return df
        except: pass

    # 2. 파일이 없거나 날짜가 지났으면 API 호출
    with st.spinner("🔄 최신 데이터를 서버에서 받아오는 중입니다... (하루 1회)"):
        try:
            new_df = fetch_multi_data()
            if new_df is not None and not new_df.empty:
                new_df.to_csv(DATA_CACHE_PATH)
                st.toast("✅ 데이터 업데이트 완료!")
                return new_df
        except: pass
        if os.path.exists(DATA_CACHE_PATH):
            return pd.read_csv(DATA_CACHE_PATH, index_col=0, parse_dates=True)
    return None

scaler = load_scaler()
df = get_data_with_cache()
features = list(TICKERS.keys())

if df is None or df.empty or len(df) < 5:
    st.error("🚨 데이터를 불러올 수 없습니다.")
    st.stop()

try: btc_idx = features.index('BTC_Close')
except: btc_idx = 0

# ------------------------------------------------------------------------------
# 6. Sidebar
# ------------------------------------------------------------------------------
with st.sidebar:
    try:
        if os.path.exists(LOGO_PATH):
            st.image(Image.open(LOGO_PATH), use_container_width=True)
        else:
            st.markdown("## 🐻 **TOBIT**")
    except: st.markdown("## 🐻 **TOBIT**")
    
    st.markdown("### **TOBIT**\n*AI 기반 비트코인 투자 분석 플랫폼*")
    st.markdown("---")
    menu = st.radio("MENU", ["📊 Market Forecast", "🧠 Deep Insight (XAI)", "📘 Model Specs", "⚡ Strategy Backtest"])
    st.markdown("---")
    st.markdown("<div style='color: #8b949e; font-size: 12px; margin-bottom: 5px;'>PARAMETERS</div>", unsafe_allow_html=True)
    selected_seq_len = st.select_slider("Lookback Window", options=[14, 21, 45], value=14, format_func=lambda x: f"{x} Days")
    selected_model = st.selectbox("Target Model", MODELS_LIST, index=3)
    
    st.markdown(f"""
    <div style="background-color: #161b22; padding: 10px; border-radius: 8px; border: 1px solid #262a33; margin-top: 20px;">
        <div style="font-size: 11px; color: #8b949e;">SYSTEM STATUS</div>
        <div style="display: flex; justify-content: space-between; margin-top: 5px;">
            <span style="color: #e6edf3; font-size: 12px;">Engine</span>
            <span style="color: #3fb950; font-size: 12px;">● Online</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 2px;">
            <span style="color: #e6edf3; font-size: 12px;">Model</span>
            <span style="color: #58a6ff; font-size: 12px;">{selected_model}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # [RESTORED] 디스코드 알람 로직
    st.markdown("---")
    if st.button("🔔 Send Report to Discord", use_container_width=True):
        with st.spinner("AI Analyzing Signal..."):
            try:
                model = get_model(selected_model, selected_seq_len)
                input_raw = df[features].tail(selected_seq_len).values
                input_tensor = torch.tensor(scaler.transform(input_raw)).float().unsqueeze(0)
                with torch.no_grad(): preds_scaled = model(input_tensor).numpy()[0]
                dummy = np.zeros(len(features)); dummy[btc_idx] = preds_scaled[-1]
                target_price = scaler.inverse_transform(dummy.reshape(1, -1))[0][btc_idx]
                
                current_price = df['BTC_Close'].iloc[-1]
                pct = ((target_price - current_price) / current_price) * 100
                if pct > 1.0: signal, color = "STRONG BUY 🚀", 0x3fb950
                elif pct < -1.0: signal, color = "STRONG SELL 📉", 0xf85149
                else: signal, color = "HOLD ✋", 0xffeb3b
                
                fields = [
                    {"name": "💰 Price", "value": f"${current_price:,.0f}", "inline": True},
                    {"name": "🎯 Target", "value": f"${target_price:,.0f} ({pct:+.2f}%)", "inline": True},
                    {"name": "🔮 Signal", "value": signal, "inline": False},
                    {"name": "🤖 Model", "value": selected_model, "inline": True}
                ]
                success, msg = send_discord_message("📢 TOBIT Investment Alert", f"AI 모델({selected_model}) 분석 결과", fields, color)
                if success: st.success("전송 완료!")
                else: st.error(f"전송 실패: {msg}")
            except Exception as e: st.error(f"오류: {e}")
            
    # [RESTORED] 캡션 및 초대 버튼
    st.caption("ℹ️ 클릭 시 현재 시황과 AI 예측(7일 후)이 포함된 요약 리포트를 디스코드로 전송합니다.")
    st.link_button("👾 Join TOBIT Discord", "https://discord.gg/mQDsWnpx", use_container_width=True)

if menu != "📘 Model Specs":
    c_logo, c_title = st.columns([0.08, 0.92])
    with c_logo: st.write(icon_img)
    with c_title: st.markdown("<h2 style='margin-top: 5px;'>TOBIT Analysis Dashboard</h2>", unsafe_allow_html=True)

    if len(df) >= 2:
        last_row, prev_row = df.iloc[-1], df.iloc[-2]
        price_diff = last_row['BTC_Close'] - prev_row['BTC_Close']
        def kpi(label, val, delta, color): return f"""<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div><div class="kpi-delta {color}">{delta}</div></div>"""
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(kpi("BTC Price", f"${last_row['BTC_Close']:,.0f}", f"{'▲' if price_diff>=0 else '▼'} {price_diff:+.2f}", "text-green" if price_diff>=0 else "text-red"), unsafe_allow_html=True)
        with c2: st.markdown(kpi("Sentiment", f"{last_row['Fear_Greed_Index']:.0f}", "Extreme Greed" if last_row['Fear_Greed_Index']>75 else "Neutral", "text-blue"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("RSI (14)", f"{last_row['RSI']:.1f}", "Neutral", "text-green"), unsafe_allow_html=True)
        with c4: st.markdown(kpi("US 10Y", f"{last_row['US_10Y']:.3f}%", "Macro Index", "text-blue"), unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 7. Main Tabs
# ------------------------------------------------------------------------------

# [TAB 1] Forecast
if menu == "📊 Market Forecast":
    st.markdown(f"#### 🤖 AI Model Projection: {selected_model}")
    model = get_model(selected_model, selected_seq_len)
    
    if model:
        input_raw = df[features].tail(selected_seq_len).values
        input_tensor = torch.tensor(scaler.transform(input_raw)).float().unsqueeze(0)
        with torch.no_grad(): preds_scaled = model(input_tensor).numpy()[0]
        
        preds = []
        for p in preds_scaled:
            dummy = np.zeros(len(features)); dummy[btc_idx] = p
            preds.append(scaler.inverse_transform(dummy.reshape(1, -1))[0][btc_idx])
            
        future_dates = [pd.to_datetime(df.index[-1]) + pd.Timedelta(days=i) for i in range(1, 8)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-90:], y=df['BTC_Close'].tail(90), name="Historical", mode='lines', line=dict(color='rgba(139, 148, 158, 0.5)', width=2), fill='tozeroy', fillcolor='rgba(139, 148, 158, 0.1)'))
        pred_color = '#3fb950' if preds[-1] > preds[0] else '#f85149'
        fig.add_trace(go.Scatter(x=future_dates, y=preds, name=f"TOBIT Forecast", mode='lines+markers', line=dict(color=pred_color, width=3), marker=dict(size=6, color='#161b22', line=dict(width=2, color=pred_color))))
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#262a33'), hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""<div style="padding: 15px; border-left: 3px solid {pred_color}; background-color: #161b22;"><span style="color: #8b949e; font-size: 13px;">TOBIT Analysis Summary:</span><br><span style="font-size: 16px; font-weight: bold; color: #e6edf3;">Target Price (7D): ${preds[-1]:,.0f}</span></div>""", unsafe_allow_html=True)
    else: st.warning("Model weights not found.")

# [TAB 2] Deep Insight
elif menu == "🧠 Deep Insight (XAI)":
    st.markdown(f"#### 🧠 Deep Explainable AI: {selected_model}")
    model = get_model(selected_model, selected_seq_len)
    
    if model:
        X_all = scaler.transform(df[features].values)
        input_raw = df[features].tail(selected_seq_len).values
        input_scaled = scaler.transform(input_raw)
        f_hs = lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy()[:, 0]
        instance_data = input_scaled.reshape(1, selected_seq_len, -1)
        average_event = pd.DataFrame(X_all.mean(axis=0).reshape(1, -1), columns=features)
        
        c_head, c_param = st.columns([2, 1])
        with c_head: st.markdown("### 1️⃣ TimeSHAP Analysis")
        with c_param: pruning_tol = st.slider("✂️ Pruning Tolerance", 0.01, 0.30, 0.10, 0.01)
            
        with st.spinner("Calculating Pruning..."):
            plot_data, prun_idx = local_pruning(f_hs, instance_data, {'tol': pruning_tol}, average_event, None, None, False)
            st.pyplot(get_pruning_plot(plot_data, prun_idx, "0. Pruning Plot"), use_container_width=True)
            pos_prun_idx = selected_seq_len + prun_idx

        t_l1, t_l2, t_l3 = st.tabs(["Event", "Feature", "Cell"])
        cache_key = f"l_event_{pruning_tol}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = local_event(f_hs, instance_data, {'rs':42, 'nsamples':800}, None, None, average_event, pos_prun_idx)
            st.session_state[f'l_feat_{pruning_tol}'] = local_feat(f_hs, instance_data, {'rs':42, 'nsamples':800, 'feature_names': features}, None, None, average_event, pos_prun_idx)
            st.session_state[f'l_cell_{pruning_tol}'] = local_cell_level(f_hs, instance_data, {'rs':42, 'nsamples':800, 'top_x_events':3, 'top_x_feats':3}, st.session_state[cache_key], st.session_state[f'l_feat_{pruning_tol}'], None, None, average_event, pos_prun_idx)

        with t_l1: st.pyplot(get_event_heatmap(st.session_state[cache_key], "1. Local Event Importance"), use_container_width=True)
        with t_l2: st.pyplot(get_feature_bar(st.session_state[f'l_feat_{pruning_tol}'], "2. Local Feature Importance"), use_container_width=True)
        with t_l3: st.pyplot(get_cell_heatmap(st.session_state[f'l_cell_{pruning_tol}'], "3. Local Cell Importance"), use_container_width=True)

        st.markdown("#### 🌍 Global Analysis")
        if st.button("Run Global Analysis"):
            with st.spinner("Running..."):
                sample_indices = np.random.choice(len(X_all) - selected_seq_len - 7, 5, replace=False)
                g_feats, g_evts = [], []
                for i in sample_indices:
                    s_in = X_all[i:i+selected_seq_len].reshape(1, selected_seq_len, -1)
                    g_feats.append(local_feat(f_hs, s_in, {'rs':42, 'nsamples':100, 'feature_names': features}, None, None, average_event, 0))
                    g_evts.append(local_event(f_hs, s_in, {'rs':42, 'nsamples':100}, None, None, average_event, 0))
                
                global_feat = pd.concat(g_feats).groupby("Feature")["Shapley Value"].apply(lambda x: x.abs().mean()).reset_index()
                evt_list = []
                for df_evt in g_evts:
                    if 'Feature' not in df_evt.columns: df_evt = df_evt.reset_index(); df_evt.columns = ['Feature', 'Shapley Value']
                    evt_list.append(df_evt)
                global_evt = pd.concat(evt_list).groupby("Feature")["Shapley Value"].apply(lambda x: x.abs().mean()).reset_index()
                
                c1, c2 = st.columns(2)
                with c1: st.pyplot(get_feature_bar(global_feat, "4. Global Feature"), use_container_width=True)
                with c2: st.pyplot(get_event_heatmap(global_evt, "5. Global Event"), use_container_width=True)

        st.markdown("---")
        st.markdown("### 2️⃣ Counterfactual Simulator")
        cf_c1, cf_c2, cf_c3 = st.columns([1, 1, 2])
        with cf_c1: target = st.selectbox("Feature", features, index=btc_idx)
        with cf_c2: 
            cur_val = input_raw[-1, features.index(target)]
            delta = st.slider("Change (%)", -30, 30, 0)
        
        mod_raw = input_raw.copy()
        mod_raw[-1, features.index(target)] = cur_val * (1 + delta/100)
        
        with torch.no_grad():
            orig_p = model(torch.tensor(input_scaled).float().unsqueeze(0)).numpy()[0]
            mod_p = model(torch.tensor(scaler.transform(mod_raw)).float().unsqueeze(0)).numpy()[0]
            
        def inv(p): d = np.zeros(len(features)); d[btc_idx] = p; return scaler.inverse_transform(d.reshape(1, -1))[0][btc_idx]
        orig_real = [inv(p) for p in orig_p]
        mod_real = [inv(p) for p in mod_p]
        diff = mod_real[-1] - orig_real[-1]
        
        with cf_c3: st.metric("Impact (Day 7)", f"{diff:+.2f} USD")
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Scatter(y=orig_real, name="Original", line=dict(dash='dot', color='#8b949e')))
        fig_cf.add_trace(go.Scatter(y=mod_real, name="What-If", line=dict(color='#58a6ff')))
        fig_cf.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_cf, use_container_width=True)

        if st.button("✨ Ask AI Analyst (Simulation)"):
            with st.spinner("AI analyzing..."):
                prompt = f"[Role] Crypto Analyst.\n[Scenario] {target} changes by {delta}%, Price changes by {diff:.2f}.\n[Task] Interpret sensitivity (Korean, 3 sentences)."
                res = client.chat.completions.create(model="solar-pro2", messages=[{"role":"user","content":prompt}])
                st.markdown(f"""<div class="ai-chat-box"><h4>🤖 Solar Pro 2 Insight</h4><p>{res.choices[0].message.content}</p></div>""", unsafe_allow_html=True)

# [TAB 3] Model Specs (Detail Updated)
elif menu == "📘 Model Specs":
    st.markdown("#### 📘 Model Specifications & Architecture")
    st.info("TOBIT 플랫폼에서 활용하는 6가지 시계열 모델의 아키텍처와 상세 스펙입니다.")
    
    graph_attr = {'bgcolor': 'transparent', 'rankdir': 'LR', 'nodesep': '0.5', 'ranksep': '0.5'}
    node_attr = {'shape': 'box', 'style': 'filled', 'fillcolor': '#1f242c', 'fontcolor': 'white', 'color': '#58a6ff', 'fontname': 'Roboto'}
    edge_attr = {'color': '#8b949e'}

    tab_mlp, tab_dl, tab_tcn, tab_lstm, tab_patch, tab_itr = st.tabs(
        ["MLP", "DLinear", "TCN", "LSTM", "PatchTST", "iTransformer"]
    )
    
    with tab_mlp:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### **MLP (Multi-Layer Perceptron)**")
            st.write("가장 기초적인 심층 신경망으로, 비선형 패턴을 단순하게 학습합니다.")
            dot = graphviz.Digraph(graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr)
            dot.edge('Input (Lag)', 'Hidden Layer 1')
            dot.edge('Hidden Layer 1', 'Hidden Layer 2')
            dot.edge('Hidden Layer 2', 'Output (Price)')
            st.graphviz_chart(dot)
        with c2:
            st.markdown("**🔧 Key Hyperparameters**")
            st.table(pd.DataFrame({"Parameter": ["Hidden Size", "Num Layers", "Activation"], "Value": ["128 ~ 256", "2 ~ 3", "ReLU"]}).set_index("Parameter"))
            st.markdown("""
            **✅ 장점 (Pros)**
            - 구조가 단순하여 **학습 및 추론 속도가 가장 빠름**.
            - 적은 데이터로도 기본적인 패턴 학습 가능.
            
            **❌ 단점 (Cons)**
            - 시계열의 **시간적 순서(Temporal Order)**를 무시함.
            - 복잡한 장기 의존성 패턴 학습에 한계.
            """)

    with tab_dl:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### **DLinear (Decomposition Linear)**")
            st.write("시계열을 추세(Trend)와 계절성(Seasonality)으로 분해하여 각각 예측 후 합치는 모델입니다.")
            dot = graphviz.Digraph(graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr)
            dot.node('Decomp', 'Series Decomposition\n(Moving Avg)', shape='ellipse', color='#d29922')
            dot.edge('Input', 'Decomp'); dot.edge('Decomp', 'Trend'); dot.edge('Decomp', 'Seasonal'); dot.edge('Trend', 'Output'); dot.edge('Seasonal', 'Output')
            st.graphviz_chart(dot)
        with c2:
            st.markdown("**🔧 Key Hyperparameters**")
            st.table(pd.DataFrame({"Parameter": ["Moving Avg Kernel", "Individual Head"], "Value": ["25", "False"]}).set_index("Parameter"))
            st.markdown("""
            **✅ 장점 (Pros)**
            - **SOTA급 성능**: 복잡한 트랜스포머보다 시계열 예측 성능이 우수할 때가 많음.
            - 해석이 쉽고(Trend/Seasonal) 매우 가벼움.
            
            **❌ 단점 (Cons)**
            - 선형 결합 기반이므로 **급격하고 복잡한 비선형 변동성** 모델링에 한계.
            """)

    with tab_tcn:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### **TCN (Temporal Convolutional Network)**")
            st.write("Dilated Convolution을 활용하여 긴 시계열 데이터를 효율적으로 처리하는 CNN 기반 모델입니다.")
            dot = graphviz.Digraph(graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr)
            dot.edge('Input', 'Dilated Conv 1'); dot.edge('Dilated Conv 1', 'Dilated Conv 2'); dot.edge('Dilated Conv 2', 'Residual'); dot.edge('Residual', 'Output')
            st.graphviz_chart(dot)
        with c2:
            st.markdown("**🔧 Key Hyperparameters**")
            st.table(pd.DataFrame({"Parameter": ["Kernel Size", "Channels", "Dilation"], "Value": ["3", "64", "2^i"]}).set_index("Parameter"))
            st.markdown("""
            **✅ 장점 (Pros)**
            - **병렬 처리**가 가능하여 LSTM보다 학습 속도가 빠름.
            - Dilated Conv로 **매우 긴 과거 데이터(Receptive Field)** 참조 가능.
            
            **❌ 단점 (Cons)**
            - 모델 깊이에 따라 **메모리 사용량이 급증**할 수 있음.
            """)

    with tab_lstm:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### **LSTM (Long Short-Term Memory)**")
            st.write("금융 시계열 예측의 표준 모델로, 게이트 구조를 통해 장/단기 기억을 모두 활용합니다.")
            dot = graphviz.Digraph(graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr)
            dot.node('Cell', 'LSTM Cell\n(Forget/Input/Output)', shape='ellipse', color='#d29922')
            dot.edge('Input(t)', 'Cell'); dot.edge('Hidden(t-1)', 'Cell'); dot.edge('Cell', 'Hidden(t)'); dot.edge('Hidden(t)', 'Output')
            st.graphviz_chart(dot)
        with c2:
            st.markdown("**🔧 Key Hyperparameters**")
            st.table(pd.DataFrame({"Parameter": ["Hidden Size", "Layers", "Dropout"], "Value": ["64~128", "2", "0.2"]}).set_index("Parameter"))
            st.markdown("""
            **✅ 장점 (Pros)**
            - **금융 시계열의 표준**: 노이즈가 많은 데이터에서도 안정적.
            - 시퀀스(Sequence) 패턴 학습에 최적화됨.
            
            **❌ 단점 (Cons)**
            - 순차적 계산으로 인해 **학습 속도가 느림(병렬화 불가)**.
            - 시퀀스가 매우 길어지면 초기 정보를 잊는 경향 있음.
            """)

    with tab_patch:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### **PatchTST (Patch Time Series Transformer)**")
            st.write("시계열을 이미지 패치처럼 잘라서 트랜스포머에 입력하는 최신 SOTA 모델입니다.")
            dot = graphviz.Digraph(graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr)
            dot.node('Patch', 'Patching\n(Stride=3)', shape='component')
            dot.edge('Input', 'Patch'); dot.edge('Patch', 'Transformer'); dot.edge('Transformer', 'Head'); dot.edge('Head', 'Output')
            st.graphviz_chart(dot)
        with c2:
            st.markdown("**🔧 Key Hyperparameters**")
            st.table(pd.DataFrame({"Parameter": ["Patch Len", "Stride", "d_model"], "Value": ["7", "3", "64"]}).set_index("Parameter"))
            st.markdown("""
            **✅ 장점 (Pros)**
            - **장기 시계열 예측(Long-term)**에서 압도적인 성능.
            - 지역적 의미(Local Semantic) 보존 및 연산 효율성 극대화.
            
            **❌ 단점 (Cons)**
            - 데이터가 적을 경우 **과적합(Overfitting)** 위험이 큼.
            """)

    with tab_itr:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### **iTransformer (Inverted Transformer)**")
            st.write("시간 축이 아닌 변수(Variate) 축을 임베딩하여 다변량 상관관계를 집중 학습합니다.")
            dot = graphviz.Digraph(graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr)
            dot.node('Embed', 'Inverted Embed\n(Tokens=Variables)', shape='box', color='#d29922')
            dot.edge('Input', 'Embed'); dot.edge('Embed', 'Self-Attention'); dot.edge('Self-Attention', 'Output')
            st.graphviz_chart(dot)
        with c2:
            st.markdown("**🔧 Key Hyperparameters**")
            st.table(pd.DataFrame({"Parameter": ["d_model", "Layers", "Heads"], "Value": ["256", "3", "4"]}).set_index("Parameter"))
            st.markdown("""
            **✅ 장점 (Pros)**
            - **다변량 상관관계(Multivariate Correlation)** 학습에 탁월.
            - 변수(Feature)가 많을수록 강점을 발휘함.
            
            **❌ 단점 (Cons)**
            - 모델이 무거워 **학습 및 추론 리소스 소모가 큼**.
            """)

# [TAB 4] Backtest
elif menu == "⚡ Strategy Backtest":
    st.markdown("#### 🧪 Backtest Simulation")
    c1, c2 = st.columns([1.5, 1])
    with c1: st.info("매일 아침 AI 예측 수익률을 기반으로 매수/매도/관망을 결정합니다.")
    with c2:
        thresh = st.slider("🎯 Target Return (%)", 1.0, 10.0, 5.0, 0.5)
        cap = st.number_input("💰 Initial Capital ($)", 100, 100000, 10000)

    if st.button("🚀 Run Simulation"):
        model = get_model(selected_model, selected_seq_len)
        if model:
            with st.spinner("Simulating..."):
                window = 180
                data = df.tail(window + selected_seq_len).reset_index(drop=True)
                hist_tensor = torch.tensor(scaler.transform(data[features].values)).float()
                
                cash, coin = float(cap), 0.0
                res, port_hist, bh_hist = [], [], []
                
                for i in range(window):
                    idx = i + selected_seq_len
                    with torch.no_grad():
                        p_seq = model(hist_tensor[i:idx].unsqueeze(0)).numpy()[0]
                    
                    pred_prices = []
                    for p in p_seq:
                        d = np.zeros(len(features)); d[btc_idx] = p
                        pred_prices.append(scaler.inverse_transform(d.reshape(1, -1))[0][btc_idx])
                    
                    avg_pred = np.mean(pred_prices)
                    cur_price = data.iloc[idx-1]['BTC_Close']
                    ret_pct = ((avg_pred - cur_price) / cur_price) * 100
                    
                    action = "HOLD"
                    if ret_pct >= thresh and cash > 0:
                        coin += (cash * 0.9995) / cur_price; cash = 0; action = "BUY"
                    elif ret_pct <= -thresh and coin > 0:
                        cash += (coin * cur_price * 0.9995); coin = 0; action = "SELL"
                        
                    total = cash + (coin * cur_price)
                    port_hist.append(total)
                    bh_hist.append((cur_price / data.iloc[selected_seq_len-1]['BTC_Close']) * cap)
                    
                    res.append({"Date": data.iloc[idx-1].name, "Price": cur_price, "Return(%)": round(ret_pct, 2), "Action": action, "Total": round(total, 2)})
                
                f_ret = (port_hist[-1] - cap) / cap * 100
                b_ret = (bh_hist[-1] - cap) / cap * 100
                
                c1, c2 = st.columns(2)
                c1.metric("Strategy Return", f"{f_ret:.2f}%", f"${port_hist[-1]:,.0f}")
                c2.metric("Buy & Hold Return", f"{b_ret:.2f}%", f"${bh_hist[-1]:,.0f}")
                st.caption("ℹ️ Buy & Hold: 시작일에 전액 매수 후 보유했을 경우의 가치")
                
                df_res = pd.DataFrame(res)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_res.index, y=port_hist, name="TOBIT", line=dict(color='#58a6ff', width=3)))
                fig.add_trace(go.Scatter(x=df_res.index, y=bh_hist, name="Hold", line=dict(color='#8b949e', dash='dot')))
                fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_res, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center; color:#8b949e; font-size:12px;'>TOBIT v2.5 | AI-Driven Investment Analysis Platform</div>", unsafe_allow_html=True)
