import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import inspect
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI
import altair as alt

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
# [수정] 현재 파일(app.py)의 절대 경로를 구해서 assets 경로를 정확히 지정합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")

st.set_page_config(
    page_title="TOBIT | From Data to Bitcoin",
    page_icon="🐻",
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
    st.error("🚨 API 키가 없습니다. .streamlit/secrets.toml 파일을 확인해주세요.")
    st.stop()

BASE_URL = "https://api.upstage.ai/v1"
client = OpenAI(api_key=UPSTAGE_API_KEY, base_url=BASE_URL)

# ------------------------------------------------------------------------------
# 3. Import Dependencies
# ------------------------------------------------------------------------------
try:
    from timeshap.explainer import local_pruning, local_event, local_feat, local_cell_level
except ImportError as e:
    st.error(f"🚨 TimeSHAP 라이브러리 로드 실패: {e}")
    st.stop()
except Exception as e:
    st.error(f"🚨 TimeSHAP 초기화 오류: {e}")
    st.stop()

try:
    from model import LSTMModel, DLinear, PatchTST, iTransformer, TCN, MLP
    from data_utils import fetch_multi_data, load_scaler, TICKERS
except ImportError:
    st.error("🚨 필수 파일(model.py, data_utils.py)이 누락되었습니다.")
    st.stop()

# ------------------------------------------------------------------------------
# 4. Helper Functions (Visualization) - [수정됨: 그래프 크기 키움]
# ------------------------------------------------------------------------------
def get_pruning_plot(plot_data, pruning_idx, title="Pruning Plot"):
    if plot_data is None: return None
    df_plot = pd.DataFrame([{'Index': item[1], 'Value': item[2]} for item in plot_data]) if isinstance(plot_data, list) else plot_data.copy()
    
    fig, ax = plt.subplots(figsize=(10, 4)) # figsize 확대
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
    
    fig, ax = plt.subplots(figsize=(8, 6)) # figsize 확대
    sns.heatmap(df_plot.pivot_table(index='Feature', values='Shapley Value'), 
                cmap='coolwarm', center=0, annot=True, fmt=".3f", 
                ax=ax, cbar=False, annot_kws={"size": 10})
    ax.set_title(title, fontsize=12, color='#e6edf3'); ax.set_ylabel(""); 
    ax.tick_params(axis='y', colors='#8b949e', labelsize=10); ax.set_xticks([])
    return fig

def get_feature_bar(df, title):
    if df is None or df.empty: return None
    df_plot = df.copy()
    df_plot['abs_val'] = df_plot['Shapley Value'].abs()
    df_plot = df_plot.sort_values(by='abs_val', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(8, 5)) # figsize 확대
    fig.patch.set_facecolor('#0b0e11'); ax.set_facecolor('#0b0e11')
    ax.spines['bottom'].set_color('#8b949e'); ax.spines['left'].set_color('#8b949e')
    ax.tick_params(colors='#8b949e', labelsize=10)
    sns.barplot(x='Shapley Value', y='Feature', data=df_plot, color='#58a6ff', ax=ax)
    ax.axvline(x=0, color='gray', linewidth=0.8); ax.set_title(title, fontsize=12, loc='left', color='#e6edf3'); ax.set_ylabel("")
    return fig

def get_cell_heatmap(cell_df, title):
    if cell_df is None or cell_df.empty: return None
    fig, ax = plt.subplots(figsize=(10, 6)) # figsize 확대
    sns.heatmap(cell_df.pivot(index='Feature', columns='Event', values='Shapley Value'), 
                cmap='coolwarm', center=0, annot=True, fmt=".3f", 
                ax=ax, cbar=False, annot_kws={"size": 9})
    ax.set_title(title, fontsize=12, color='#e6edf3'); ax.tick_params(colors='#8b949e', labelsize=9); ax.set_xlabel(""); ax.set_ylabel("")
    return fig

# ------------------------------------------------------------------------------
# 5. Model Logic
# ------------------------------------------------------------------------------
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights') # 경로 수정
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
        try: model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        except: model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

scaler, df = load_scaler(), fetch_multi_data()
features = list(TICKERS.keys())
try: btc_idx = features.index('BTC_Close')
except: btc_idx = 0

# ------------------------------------------------------------------------------
# 6. Sidebar & KPI
# ------------------------------------------------------------------------------
with st.sidebar:
    # [수정] 위에서 구한 절대 경로(LOGO_PATH)를 사용
    if os.path.exists(LOGO_PATH): 
        st.image(LOGO_PATH, width=200)
    else: 
        st.markdown("## 🐻 **TOBIT**")
        # 디버깅용: 경로가 틀렸을 때 어디를 찾고 있는지 표시하려면 아래 주석 해제
        # st.caption(f"Logo not found at: {LOGO_PATH}")
    
    st.markdown("### **TOBIT**\n*From Data to Bitcoin*")
    st.markdown("---")
    menu = st.radio("MENU", ["📊 Market Forecast", "🧠 Deep Insight (XAI)", "📘 Model Specs", "⚡ Strategy Backtest"])
    st.markdown("---")
    st.markdown("<div style='color: #8b949e; font-size: 12px; margin-bottom: 5px;'>PARAMETERS</div>", unsafe_allow_html=True)
    selected_seq_len = st.select_slider("Lookback Window", options=[14, 21, 45], value=14, format_func=lambda x: f"{x} Days")
    selected_model = st.selectbox("Target Model", MODELS_LIST, index=3)
    st.markdown(f"""<div style="background-color: #161b22; padding: 10px; border-radius: 8px; border: 1px solid #262a33; margin-top: 20px;"><div style="font-size: 11px; color: #8b949e;">SYSTEM STATUS</div><div style="display: flex; justify-content: space-between; margin-top: 5px;"><span style="color: #e6edf3; font-size: 12px;">Engine</span><span style="color: #3fb950; font-size: 12px;">● Online</span></div><div style="display: flex; justify-content: space-between; margin-top: 2px;"><span style="color: #e6edf3; font-size: 12px;">Model</span><span style="color: #58a6ff; font-size: 12px;">{selected_model}</span></div></div>""", unsafe_allow_html=True)

if menu != "📘 Model Specs":
    c_logo, c_title = st.columns([0.08, 0.92])
    with c_logo: 
        # [수정] 메인 화면 로고에도 절대 경로 적용
        if os.path.exists(LOGO_PATH): 
            st.image(LOGO_PATH, width=50)
        else: 
            st.markdown("🐻")
    with c_title: st.markdown("<h2 style='margin-top: 5px;'>TOBIT Analysis Dashboard</h2>", unsafe_allow_html=True)

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
# 7. Main Content Tabs
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
            dummy = np.zeros(len(features))
            dummy[btc_idx] = p
            preds.append(scaler.inverse_transform(dummy.reshape(1, -1))[0][btc_idx])
            
        future_dates = [pd.to_datetime(df['timestamp'].values[-1]) + pd.Timedelta(days=i) for i in range(1, 8)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'].tail(90), y=df['BTC_Close'].tail(90), name="Historical", mode='lines', line=dict(color='rgba(139, 148, 158, 0.5)', width=2), fill='tozeroy', fillcolor='rgba(139, 148, 158, 0.1)'))
        pred_color = '#3fb950' if preds[-1] > preds[0] else '#f85149'
        fig.add_trace(go.Scatter(x=future_dates, y=preds, name=f"TOBIT Forecast", mode='lines+markers', line=dict(color=pred_color, width=3), marker=dict(size=6, color='#161b22', line=dict(width=2, color=pred_color))))
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#262a33'), hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""<div style="padding: 15px; border-left: 3px solid {pred_color}; background-color: #161b22;"><span style="color: #8b949e; font-size: 13px;">TOBIT Analysis Summary:</span><br><span style="font-size: 16px; font-weight: bold; color: #e6edf3;">Target Price (7D): ${preds[-1]:,.0f}</span></div>""", unsafe_allow_html=True)
    else: st.warning("Model weights not found.")

# [TAB 2] Deep Insight (XAI)
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
                    if 'Feature' not in df_evt.columns: 
                        df_evt = df_evt.reset_index()
                        df_evt.columns = ['Feature', 'Shapley Value']
                    evt_list.append(df_evt)
                global_evt = pd.concat(evt_list).groupby("Feature")["Shapley Value"].apply(lambda x: x.abs().mean()).reset_index()
                
                c1, c2 = st.columns(2)
                with c1: st.pyplot(get_feature_bar(global_feat, "4. Global Feature"), use_container_width=True)
                with c2: st.pyplot(get_event_heatmap(global_evt, "5. Global Event"), use_container_width=True)

        if st.button("✨ Ask AI Analyst (TimeSHAP)"):
            with st.spinner("AI analyzing..."):
                try:
                    feat_df = st.session_state.get(f'l_feat_{pruning_tol}')
                    evt_df = st.session_state.get(cache_key)
                    
                    feat_txt = "\n".join([f"- {r.Feature}: {r['Shapley Value']:.4f}" for _, r in feat_df.head(3).iterrows()]) if feat_df is not None else "N/A"
                    evt_txt = "N/A"
                    if evt_df is not None:
                        if 'Feature' in evt_df.columns: evt_df = evt_df.set_index('Feature')
                        evt_txt = "\n".join([f"- {i}: {r['Shapley Value']:.4f}" for i, r in evt_df.head(3).iterrows()])
                        
                    prompt = f"[Role] Crypto Analyst.\n[Data]\nFeatures:\n{feat_txt}\nEvents:\n{evt_txt}\n[Task] Explain WHY based on data (Korean, 3 sentences)."
                    res = client.chat.completions.create(model="solar-pro2", messages=[{"role":"user","content":prompt}])
                    st.markdown(f"""<div class="ai-chat-box"><h4>🤖 Solar Pro 2 Insight</h4><p>{res.choices[0].message.content}</p></div>""", unsafe_allow_html=True)
                except Exception as e: st.error(str(e))

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
            
        def inv(p): 
            d = np.zeros(len(features)); d[btc_idx] = p
            return scaler.inverse_transform(d.reshape(1, -1))[0][btc_idx]
            
        orig_real = [inv(p) for p in orig_p]
        mod_real = [inv(p) for p in mod_p]
        diff = mod_real[-1] - orig_real[-1]
        
        with cf_c3: 
            st.metric("Impact (Day 7)", f"{diff:+.2f} USD")
            
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
                    
                    # [SAFE FIX] Replaced one-liner with loop and explicit reshape
                    pred_prices = []
                    for p in p_seq:
                        d = np.zeros(len(features)); d[btc_idx] = p
                        # reshape(1, -1) guarantees 2D array: (1, n_features)
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
                    
                    res.append({"Date": data.iloc[idx-1]['timestamp'], "Price": cur_price, "Return(%)": round(ret_pct, 2), "Action": action, "Total": round(total, 2)})
                
                f_ret = (port_hist[-1] - cap) / cap * 100
                b_ret = (bh_hist[-1] - cap) / cap * 100
                
                c1, c2 = st.columns(2)
                c1.metric("Strategy Return", f"{f_ret:.2f}%", f"${port_hist[-1]:,.0f}")
                c2.metric("Buy & Hold Return", f"{b_ret:.2f}%", f"${bh_hist[-1]:,.0f}")
                st.caption("ℹ️ Buy & Hold: 시작일에 전액 매수 후 보유했을 경우의 가치")
                
                df_res = pd.DataFrame(res)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_res['Date'], y=port_hist, name="TOBIT", line=dict(color='#58a6ff', width=3)))
                fig.add_trace(go.Scatter(x=df_res['Date'], y=bh_hist, name="Hold", line=dict(color='#8b949e', dash='dot')))
                fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_res, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center; color:#8b949e; font-size:12px;'>TOBIT v2.1 | Deep Learning Time Series Forecasting</div>", unsafe_allow_html=True)
