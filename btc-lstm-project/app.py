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

# ------------------------------------------------------------------------------
# [설정] 업스테이지 API 키 설정 (Secrets 사용)
# ------------------------------------------------------------------------------
if "UPSTAGE_API_KEY" in st.secrets:
    UPSTAGE_API_KEY = st.secrets["UPSTAGE_API_KEY"]
else:
    st.error("🚨 API 키가 없습니다. .streamlit/secrets.toml 파일을 확인해주세요.")
    st.stop()

BASE_URL = "https://api.upstage.ai/v1"

client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url=BASE_URL
)

# ------------------------------------------------------------------------------
# [Dependency Check] TimeSHAP 및 Custom Modules
# ------------------------------------------------------------------------------
try:
    from timeshap.explainer import local_pruning, local_event, local_feat, local_cell_level
except ImportError:
    st.error("TimeSHAP 라이브러리가 없습니다. `pip install timeshap`을 실행하세요.")

try:
    from model import LSTMModel, DLinear, PatchTST, iTransformer, TCN, MLP
    from data_utils import fetch_multi_data, load_scaler, TICKERS
except ImportError:
    st.error("model.py 또는 data_utils.py가 같은 경로에 있어야 합니다.")

# ==============================================================================
# 1. Page Config & TOBIT Theme CSS
# ==============================================================================
st.set_page_config(
    page_title="TOBIT | From Data to Bitcoin",
    page_icon="🐻",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Roboto+Mono:wght@400;700&display=swap');
    
    /* 전체 테마: Deep Navy & Blue */
    .stApp { background-color: #0b0e11; font-family: 'Inter', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #11141a; border-right: 1px solid #262a33; }
    
    /* 헤더 숨김 */
    header {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
    
    /* KPI 카드 스타일 */
    .kpi-card {
        background: linear-gradient(145deg, #161b22, #11141a); border: 1px solid #262a33;
        border-radius: 12px; padding: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .kpi-card:hover { border-color: #58a6ff; transform: translateY(-2px); }
    
    .kpi-label { font-size: 0.8rem; color: #8b949e; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { font-family: 'Roboto Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #e6edf3; }
    .kpi-delta { font-family: 'Roboto Mono', monospace; font-size: 0.85rem; margin-top: 5px; font-weight: 600; }
    
    /* 색상 유틸리티 */
    .text-green { color: #3fb950; } .text-red { color: #f85149; } .text-blue { color: #58a6ff; } .text-gold { color: #d29922; }
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #262a33; padding-bottom: 5px; }
    .stTabs [data-baseweb="tab"] { height: 40px; background-color: transparent; border: 1px solid transparent; color: #8b949e; font-weight: 600; border-radius: 6px; font-size: 0.9rem; }
    .stTabs [aria-selected="true"] { background-color: #1f242c; color: #58a6ff; border: 1px solid #262a33; }
    
    /* 코드 블록 폰트 */
    code { font-family: 'Roboto Mono', monospace !important; }

    /* Matplotlib 배경 투명화 */
    .plot-container { background-color: transparent !important; }
    
    /* AI Agent 스타일 */
    .ai-chat-box {
        background-color: #1f242c;
        border: 1px solid #58a6ff;
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
        color: #e6edf3;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. TimeSHAP Plotting Helpers (사이즈 축소 및 최적화)
# ==============================================================================
def get_pruning_plot(plot_data, pruning_idx, title="Pruning Plot"):
    if plot_data is None: return None
    df_plot = pd.DataFrame([{'Index': item[1], 'Value': item[2]} for item in plot_data]) if isinstance(plot_data, list) else plot_data.copy()
    
    # [수정] 사이즈 대폭 축소 (10,4 -> 6, 2.5)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    
    fig.patch.set_facecolor('#0b0e11')
    ax.set_facecolor('#0b0e11')
    ax.spines['bottom'].set_color('#8b949e')
    ax.spines['left'].set_color('#8b949e')
    ax.tick_params(axis='x', colors='#8b949e', labelsize=8)
    ax.tick_params(axis='y', colors='#8b949e', labelsize=8)
    
    ax.fill_between(df_plot.iloc[:, 1], df_plot.iloc[:, 2], color='#58a6ff', alpha=0.6)
    ax.axvline(x=pruning_idx, color='#f85149', linestyle='-', linewidth=1.5)
    ax.set_title(title, fontsize=10, loc='left', color='#e6edf3')
    sns.despine()
    return fig

def get_event_heatmap(df, title):
    if df is None or df.empty: return None
    df_plot = df.copy()
    
    # 'Event' 컬럼이 없고 인덱스가 Event라면 컬럼으로 만들어줌
    if 'Feature' not in df_plot.columns: 
        df_plot['Feature'] = df_plot.index

    # 정렬 로직 (Event -1, Event -2 ... 순서)
    try:
        df_plot['sort_key'] = df_plot['Feature'].str.extract(r'([-]?\d+)').astype(int)
        df_plot = df_plot.sort_values('sort_key', ascending=False).drop(columns=['sort_key'])
    except: pass
    
    # [수정] 사이즈 축소 (3.5, 6 -> 3, 5) & 폰트 사이즈 조절
    fig, ax = plt.subplots(figsize=(3, 5))
    
    sns.heatmap(df_plot.pivot_table(index='Feature', values='Shapley Value'), 
                cmap='coolwarm', center=0, annot=True, fmt=".3f", 
                ax=ax, cbar=False, annot_kws={"size": 8})
    
    ax.set_title(title, fontsize=10, color='#e6edf3')
    ax.set_ylabel("")
    ax.tick_params(axis='y', colors='#8b949e', labelsize=8)
    ax.set_xticks([]) # X축 숨김
    return fig

def get_feature_bar(df, title):
    if df is None or df.empty: return None
    df_plot = df.copy()
    df_plot['abs_val'] = df_plot['Shapley Value'].abs()
    df_plot = df_plot.sort_values(by='abs_val', ascending=False).head(10) # Top 10만 표시 (공간 절약)
    
    # [수정] 사이즈 축소 (7, 5 -> 5, 4)
    fig, ax = plt.subplots(figsize=(5, 4))
    
    fig.patch.set_facecolor('#0b0e11')
    ax.set_facecolor('#0b0e11')
    ax.spines['bottom'].set_color('#8b949e')
    ax.spines['left'].set_color('#8b949e')
    ax.tick_params(axis='x', colors='#8b949e', labelsize=8)
    ax.tick_params(axis='y', colors='#8b949e', labelsize=8)

    sns.barplot(x='Shapley Value', y='Feature', data=df_plot, color='#58a6ff', ax=ax)
    ax.axvline(x=0, color='gray', linewidth=0.8)
    ax.set_title(title, fontsize=10, loc='left', color='#e6edf3')
    ax.set_ylabel("")
    return fig

def get_cell_heatmap(cell_df, title):
    if cell_df is None or cell_df.empty: return None
    
    # [수정] 사이즈 축소 (8, 6 -> 6, 4)
    fig, ax = plt.subplots(figsize=(6, 4))
    
    sns.heatmap(cell_df.pivot(index='Feature', columns='Event', values='Shapley Value'), 
                cmap='coolwarm', center=0, annot=True, fmt=".3f", 
                ax=ax, cbar=False, annot_kws={"size": 7})
    
    ax.set_title(title, fontsize=10, color='#e6edf3')
    ax.tick_params(axis='x', colors='#8b949e', labelsize=7, rotation=45)
    ax.tick_params(axis='y', colors='#8b949e', labelsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("")
    return fig

# ==============================================================================
# 3. 데이터 및 모델 로딩
# ==============================================================================
WEIGHTS_DIR = 'weights'
MODELS_LIST = ["MLP", "DLinear", "TCN", "LSTM", "PatchTST", "iTransformer"]
MODEL_CLASSES = {
    "MLP": MLP, "DLinear": DLinear, "TCN": TCN, 
    "LSTM": LSTMModel, "PatchTST": PatchTST, "iTransformer": iTransformer
}

@st.cache_resource
def get_model(name, seq_len):
    input_size = len(TICKERS)
    pred_len = 7
    
    if name == "MLP": model = MLP(seq_len=seq_len, input_size=input_size, pred_len=pred_len)
    elif name == "DLinear": model = DLinear(seq_len=seq_len, pred_len=pred_len, input_size=input_size, kernel_size=25)
    elif name == "TCN": model = TCN(input_size=input_size, output_size=pred_len, num_channels=[64, 64, 64], kernel_size=3, dropout=0.2)
    elif name == "LSTM": model = LSTMModel(input_size=input_size, output_size=pred_len)
    elif name == "PatchTST": model = PatchTST(input_size=input_size, seq_len=seq_len, pred_len=pred_len,
                              patch_len=7, stride=3, d_model=64, n_heads=4, n_layers=2, dropout=0.2)
    elif name == "iTransformer": model = iTransformer(seq_len=seq_len, pred_len=pred_len, input_size=input_size,
                                 d_model=256, n_heads=4, n_layers=3, dropout=0.2)
    
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

# ==============================================================================
# 4. 사이드바 (TOBIT Branding)
# ==============================================================================
with st.sidebar:
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)
    else:
        st.markdown("## 🐻 **TOBIT**")

    st.markdown("### **TOBIT**\n*From Data to Bitcoin*")
    st.markdown("---")
    
    menu = st.radio("MENU", ["📊 Market Forecast", "🧠 Deep Insight (XAI)", "📘 Model Specs", "⚡ Strategy Backtest"])
    
    st.markdown("---")
    st.markdown("<div style='color: #8b949e; font-size: 12px; margin-bottom: 5px;'>PARAMETERS</div>", unsafe_allow_html=True)
    
    selected_seq_len = st.select_slider("Lookback Window", options=[14, 21, 45], value=14, format_func=lambda x: f"{x} Days")
    selected_model = st.selectbox("Target Model", MODELS_LIST, index=3) # Default: LSTM

    # 상태창
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

# ==============================================================================
# 5. KPI Cards
# ==============================================================================
if menu != "📘 Model Specs":
    # 상단 헤더
    c_logo, c_title = st.columns([0.08, 0.92])
    with c_logo:
        if os.path.exists("assets/logo.png"):
            st.image("assets/logo.png", width=50)
        else:
            st.markdown("🐻")
    with c_title:
        st.markdown("<h2 style='margin-top: 5px;'>TOBIT Analysis Dashboard</h2>", unsafe_allow_html=True)

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    price_now = last_row['BTC_Close']
    price_diff = price_now - prev_row['BTC_Close']
    pct_diff = (price_diff / prev_row['BTC_Close']) * 100
    rsi, fg_index = last_row['RSI'], last_row['Fear_Greed_Index']

    def create_kpi_card(label, value, delta, color_class):
        return f"""<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-delta {color_class}">{delta}</div></div>"""

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(create_kpi_card("BTC Price", f"${price_now:,.0f}", f"{'▲' if price_diff>=0 else '▼'} {price_diff:+.2f} ({pct_diff:+.2f}%)", "text-green" if price_diff>=0 else "text-red"), unsafe_allow_html=True)
    with c2: st.markdown(create_kpi_card("Sentiment", f"{fg_index:.0f}", "Extreme Greed" if fg_index>75 else "Neutral", "text-blue"), unsafe_allow_html=True)
    with c3: st.markdown(create_kpi_card("RSI (14)", f"{rsi:.1f}", "Overbought" if rsi>70 else "Neutral", "text-red" if rsi>70 else "text-green"), unsafe_allow_html=True)
    with c4: st.markdown(create_kpi_card("US 10Y Yield", f"{last_row['US_10Y']:.3f}%", "Macro Index", "text-blue"), unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# 6. Main Content
# ==============================================================================

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
            preds.append(scaler.inverse_transform([dummy])[0][btc_idx])
            
        future_dates = [pd.to_datetime(df['timestamp'].values[-1]) + pd.Timedelta(days=i) for i in range(1, 8)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'].tail(90), y=df['BTC_Close'].tail(90), name="Historical", mode='lines', line=dict(color='rgba(139, 148, 158, 0.5)', width=2), fill='tozeroy', fillcolor='rgba(139, 148, 158, 0.1)'))
        pred_color = '#3fb950' if preds[-1] > preds[0] else '#f85149'
        fig.add_trace(go.Scatter(x=future_dates, y=preds, name=f"TOBIT Forecast", mode='lines+markers', line=dict(color=pred_color, width=3), marker=dict(size=6, color='#161b22', line=dict(width=2, color=pred_color))))
        
        # [수정] 높이 350으로 줄임
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#262a33'), hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        diff_pred = preds[-1] - preds[0]
        st.markdown(f"""
        <div style="padding: 15px; border-left: 3px solid {'#3fb950' if diff_pred > 0 else '#f85149'}; background-color: #161b22;">
            <span style="color: #8b949e; font-size: 13px;">TOBIT Analysis Summary:</span><br>
            <span style="font-size: 16px; font-weight: bold; color: #e6edf3;">Target Price (7D): ${preds[-1]:,.0f}</span>
            <span class="{'text-green' if diff_pred > 0 else 'text-red'}" style="font-weight: bold; margin-left: 10px;">{'BULLISH 🚀' if diff_pred > 0 else 'BEARISH 📉'} ({diff_pred/preds[0]*100:+.2f}%)</span>
        </div>
        """, unsafe_allow_html=True)
    else: st.warning("Model weights not found. Please train first.")

# [TAB 2] XAI (TimeSHAP + Counterfactual + AI Agent)
elif menu == "🧠 Deep Insight (XAI)":
    st.markdown(f"#### 🧠 Deep Explainable AI: {selected_model}")
    st.info("TimeSHAP 알고리즘으로 모델의 예측 근거를 분석하고, AI Agent가 이를 쉽게 설명합니다.")

    model = get_model(selected_model, selected_seq_len)
    
    if model:
        X_all = scaler.transform(df[features].values)
        input_raw = df[features].tail(selected_seq_len).values
        input_scaled = scaler.transform(input_raw)
        
        f_hs = lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy()[:, 0]
        instance_data = input_scaled.reshape(1, selected_seq_len, -1)
        average_event = pd.DataFrame(X_all.mean(axis=0).reshape(1, -1), columns=features)
        
        # 1. Pruning
        c_head, c_param = st.columns([2, 1])
        with c_head: st.markdown("### 1️⃣ TimeSHAP Analysis")
        with c_param: pruning_tol = st.slider("✂️ Pruning Tolerance (tol)", 0.01, 0.30, 0.10, 0.01)
            
        with st.spinner(f"Calculating Pruning Statistics..."):
            plot_data, prun_idx = local_pruning(f_hs, instance_data, {'tol': pruning_tol}, average_event, None, None, False)
            st.pyplot(get_pruning_plot(plot_data, prun_idx, title="0. Pruning Plot"))
            pos_prun_idx = selected_seq_len + prun_idx

        # 2. Local Analysis (Tabs)
        t_l1, t_l2, t_l3 = st.tabs(["Event (Time)", "Feature", "Cell (Detailed)"])
        
        cache_key = f"l_event_{pruning_tol}"
        if cache_key not in st.session_state:
            with st.spinner("Analyzing Local Contributions..."):
                st.session_state[cache_key] = local_event(f_hs, instance_data, {'rs':42, 'nsamples':800}, None, None, average_event, pos_prun_idx)
                st.session_state[f'l_feat_{pruning_tol}'] = local_feat(f_hs, instance_data, {'rs':42, 'nsamples':800, 'feature_names': features}, None, None, average_event, pos_prun_idx)
                st.session_state[f'l_cell_{pruning_tol}'] = local_cell_level(f_hs, instance_data, {'rs':42, 'nsamples':800, 'top_x_events':3, 'top_x_feats':3}, st.session_state[cache_key], st.session_state[f'l_feat_{pruning_tol}'], None, None, average_event, pos_prun_idx)

        with t_l1: st.pyplot(get_event_heatmap(st.session_state[cache_key], "1. Local Event Importance"))
        with t_l2: st.pyplot(get_feature_bar(st.session_state[f'l_feat_{pruning_tol}'], "2. Local Feature Importance"))
        with t_l3: st.pyplot(get_cell_heatmap(st.session_state[f'l_cell_{pruning_tol}'], "3. Local Cell Importance"))

        # 3. Global Analysis
        st.markdown("#### 🌍 Global Analysis")
        if st.button("Run Global Analysis (Click to Start)"):
            with st.spinner("Running Global Analysis..."):
                sample_indices = np.random.choice(len(X_all) - selected_seq_len - 7, 5, replace=False)
                g_feats, g_evts = [], []
                for i in sample_indices:
                    s_in = X_all[i:i+selected_seq_len].reshape(1, selected_seq_len, -1)
                    g_feats.append(local_feat(f_hs, s_in, {'rs':42, 'nsamples':100, 'feature_names': features}, None, None, average_event, 0))
                    g_evts.append(local_event(f_hs, s_in, {'rs':42, 'nsamples':100}, None, None, average_event, 0))
                
                global_feat_agg = pd.concat(g_feats).groupby("Feature")["Shapley Value"].apply(lambda x: x.abs().mean()).reset_index()
                
                # [수정] Global Event Aggregation 로직 개선
                evt_list = []
                for df_evt in g_evts:
                    # 인덱스가 Event 이름인 경우
                    if 'Feature' not in df_evt.columns:
                        df_evt = df_evt.reset_index()
                        df_evt.columns = ['Feature', 'Shapley Value'] # 강제 컬럼명 통일
                    evt_list.append(df_evt)
                
                global_evt_agg = pd.concat(evt_list).groupby("Feature")["Shapley Value"].apply(lambda x: x.abs().mean()).reset_index()
                
                c_g1, c_g2 = st.columns(2)
                with c_g1: st.pyplot(get_feature_bar(global_feat_agg, "4. Global Feature Importance"))
                with c_g2: st.pyplot(get_event_heatmap(global_evt_agg, "5. Global Event Importance"))
        
        st.markdown("---")

        # AI Agent (TimeSHAP)
        st.markdown("### 🤖 AI Agent: TimeSHAP 분석")
        if st.button("✨ TimeSHAP 결과 해석 요청"):
            with st.spinner("Solar Pro 2 is analyzing..."):
                try:
                    l_feat_df = st.session_state.get(f'l_feat_{pruning_tol}')
                    if l_feat_df is not None:
                        top_features = l_feat_df.reindex(l_feat_df['Shapley Value'].abs().sort_values(ascending=False).index).head(3)
                        feat_context = "\n".join([f"- {row.Feature}: Impact {row['Shapley Value']:.4f}" for _, row in top_features.iterrows()])
                    else: feat_context = "N/A"

                    l_event_df = st.session_state.get(cache_key)
                    if l_event_df is not None:
                        if 'Feature' in l_event_df.columns: l_event_df = l_event_df.set_index('Feature')
                        top_events = l_event_df.reindex(l_event_df['Shapley Value'].abs().sort_values(ascending=False).index).head(3)
                        event_context = "\n".join([f"- {idx}: Impact {row['Shapley Value']:.4f}" for idx, row in top_events.iterrows()])
                    else: event_context = "N/A"

                    prompt = f"""
                    [Role] Expert Crypto Quant Analyst.
                    [Task] Interpret TimeSHAP XAI results for Bitcoin prediction.
                    
                    [Data]
                    1. Top Features:
                    {feat_context}
                    2. Top Events (Time steps):
                    {event_context}
                    
                    [Requirement]
                    - Explain WHY the model predicted this way using the features and time steps.
                    - Keep it concise (3-4 sentences).
                    - Use a professional tone like a financial report (Korean).
                    """

                    response = client.chat.completions.create(
                        model="solar-pro2",
                        messages=[{"role": "system", "content": "You are a financial analyst."}, {"role": "user", "content": prompt}],
                        stream=False
                    )
                    st.markdown(f"""<div class="ai-chat-box"><h4 style="color: #58a6ff;">🤖 Solar Pro 2 Insight (TimeSHAP)</h4><p>{response.choices[0].message.content}</p></div>""", unsafe_allow_html=True)
                except Exception as e: st.error(f"AI Analysis Failed: {e}")

        st.markdown("---")

        # --------------------------------------------------------------------------
        # B. Counterfactual Simulator
        # --------------------------------------------------------------------------
        st.markdown("### 2️⃣ Counterfactual Simulator (What-If Analysis)")
        st.info("💡 **가상 시뮬레이션:** '만약 오늘 특정 지표가 다르게 마감되었다면, 미래 7일 예측은 어떻게 변했을까?'")

        cf_col1, cf_col2, cf_col3 = st.columns([1, 1, 2])
        with cf_col1:
            target_feat = st.selectbox("Select Feature to Tweak", features, index=btc_idx)
        with cf_col2:
            current_val = input_raw[-1, features.index(target_feat)]
            delta_pct = st.slider(f"Change {target_feat} (%)", min_value=-30, max_value=30, value=0, step=1, format="%d%%")
        
        modified_input_raw = input_raw.copy()
        feat_idx = features.index(target_feat)
        modified_val = current_val * (1 + delta_pct / 100.0)
        modified_input_raw[-1, feat_idx] = modified_val
        
        mod_tensor = torch.tensor(scaler.transform(modified_input_raw)).float().unsqueeze(0)
        orig_tensor = torch.tensor(input_scaled).float().unsqueeze(0)
        
        with torch.no_grad():
            mod_pred_scaled = model(mod_tensor).numpy()[0]
            orig_pred_scaled = model(orig_tensor).numpy()[0]
            
        def inverse_preds(pred_scaled):
            res = []
            for p in pred_scaled:
                dummy = np.zeros(len(features))
                dummy[btc_idx] = p
                res.append(scaler.inverse_transform([dummy])[0][btc_idx])
            return res

        orig_preds_real = inverse_preds(orig_pred_scaled)
        mod_preds_real = inverse_preds(mod_pred_scaled)
        diff_final = mod_preds_real[-1] - orig_preds_real[-1]

        with cf_col3:
            st.markdown(f"**Changed Input:** {current_val:,.2f} ➡️ <span style='color:#58a6ff'>{modified_val:,.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"**Impact on 7th Day Price:** <span style='color:{'#3fb950' if diff_final>0 else '#f85149'}'>{diff_final:+.2f} USD</span>", unsafe_allow_html=True)

        future_dates = [pd.to_datetime(df['timestamp'].values[-1]) + pd.Timedelta(days=i) for i in range(1, 8)]
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Scatter(x=future_dates, y=orig_preds_real, name="Original Forecast", mode='lines+markers', line=dict(color='#8b949e', width=2, dash='dot')))
        fig_cf.add_trace(go.Scatter(x=future_dates, y=mod_preds_real, name="Counterfactual (What-If)", mode='lines+markers', line=dict(color='#58a6ff', width=4)))
        # [수정] 높이 350으로 축소
        fig_cf.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, hovermode="x unified", title=dict(text="Forecast Comparison", font=dict(size=14)), margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_cf, use_container_width=True)

        # AI Agent (Counterfactual)
        if st.button("✨ AI Analyst: 시나리오 해석 요청"):
            with st.spinner("Solar Pro 2 is analyzing..."):
                try:
                    prompt_cf = f"""
                    [Role] Expert Crypto Quant Analyst.
                    [Task] Analyze a Counterfactual (What-If) simulation result.
                    
                    [Scenario]
                    - If today's '{target_feat}' changes by {delta_pct}% (from {current_val:.2f} to {modified_val:.2f})...
                    - The predicted Bitcoin price 7 days later changes by {diff_final:+.2f} USD.
                    
                    [Requirement]
                    - Explain the sensitivity of Bitcoin price to '{target_feat}' based on this result.
                    - Is this impact significant?
                    - Provide a strategic insight for a trader in Korean.
                    - Keep it short (3-4 sentences).
                    """
                    response_cf = client.chat.completions.create(
                        model="solar-pro2",
                        messages=[{"role": "system", "content": "You are a helpful financial analyst."}, {"role": "user", "content": prompt_cf}],
                        stream=False
                    )
                    st.markdown(f"""<div class="ai-chat-box"><h4 style="color: #58a6ff;">🤖 Solar Pro 2 Insight (Simulation)</h4><p>{response_cf.choices[0].message.content}</p></div>""", unsafe_allow_html=True)
                except Exception as e: st.error(f"AI Analysis Failed: {e}")

# [TAB 3] Model Specs
elif menu == "📘 Model Specs":
    st.markdown(f"## 📘 Model Specs: {selected_model}")
    t1, t2, t3 = st.tabs(["📝 Theory", "🏗️ Architecture", "💻 Code"])
    model = get_model(selected_model, selected_seq_len)
    
    with t1:
        if selected_model == "PatchTST": st.info("**PatchTST** divides time-series into patches like ViT, enabling SOTA long-term forecasting.")
        elif selected_model == "TCN": st.info("**TCN** uses dilated convolutions to capture long history efficiently.")
        else: st.info(f"Details for **{selected_model}** model.")
        
    with t2: st.code(str(model), language="text")
    with t3:
        try: st.code(inspect.getsource(MODEL_CLASSES[selected_model]), language="python")
        except: st.error("Source code not found.")

# [TAB 4] Backtest
elif menu == "⚡ Strategy Backtest":
    st.markdown("#### 🧪 Strategy Backtest Simulation")
    
    # 1. UI: 전략 및 자본금 설정
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("""
        **[어떻게 작동하나요?]**
        1. **매일 아침**, AI가 향후 일주일간의 가격을 예측합니다.
        2. 예측된 **평균 가격**과 **오늘의 가격**을 비교하여 **예상 수익률**을 계산합니다.
        3. 설정한 **목표 수익률(%)**보다 높으면 사고(BUY), 낮으면 팝니다(SELL).
        4. 그 사이 구간이면 거래하지 않고 가만히 있습니다(HOLD).
        """)
    with c2:
        threshold_pct = st.slider("🎯 목표 수익률 설정 (%)", 1.0, 10.0, 5.0, 0.5)
        initial_capital = st.number_input("💰 초기 자본금 설정 (USD)", min_value=100, max_value=100000, value=10000, step=100)
        st.caption(f"💡 자본금 **${initial_capital:,.0f}**, 코인 **0개**로 시작합니다.")

    # 2. 실행 버튼
    if st.button("🚀 시뮬레이션 시작 (최근 180일)"):
        model = get_model(selected_model, selected_seq_len)
        if model:
            with st.spinner("과거 데이터를 바탕으로 투자를 진행중입니다..."):
                backtest_window = 180
                data_slice = df.tail(backtest_window + selected_seq_len).reset_index(drop=True)
                history_tensor = torch.tensor(scaler.transform(data_slice[features].values)).float()
                
                cash = float(initial_capital)
                coin = 0.0
                fee_rate = 0.0005 # 0.05% 수수료
                
                results = []
                portfolio_history = []
                buy_hold_history = []
                
                for i in range(backtest_window):
                    current_idx = i + selected_seq_len
                    input_seq = history_tensor[i : current_idx].unsqueeze(0)
                    with torch.no_grad():
                        pred_seq = model(input_seq).numpy()[0]
                    
                    pred_prices = []
                    for p in pred_seq:
                        dummy = np.zeros(len(features))
                        dummy[btc_idx] = p
                        pred_prices.append(scaler.inverse_transform([dummy])[0][btc_idx])
                    
                    avg_pred_7d = np.mean(pred_prices)
                    current_price = data_slice.iloc[current_idx-1]['BTC_Close']
                    current_date = data_slice.iloc[current_idx-1]['timestamp']
                    
                    exp_return = (avg_pred_7d - current_price) / current_price
                    exp_return_pct = exp_return * 100
                    
                    action = "HOLD"
                    if exp_return_pct >= threshold_pct:
                        if cash > 0:
                            amount_to_buy = cash * (1 - fee_rate)
                            coin += amount_to_buy / current_price
                            cash = 0
                            action = "BUY"
                    elif exp_return_pct <= -threshold_pct:
                        if coin > 0:
                            amount_to_get = coin * current_price * (1 - fee_rate)
                            cash += amount_to_get
                            coin = 0
                            action = "SELL"
                    
                    total_value = cash + (coin * current_price)
                    portfolio_history.append(total_value)
                    
                    if i == 0: start_price = current_price
                    buy_hold_value = (current_price / start_price) * initial_capital
                    buy_hold_history.append(buy_hold_value)
                    
                    results.append({
                        "Date": current_date,
                        "Price": current_price,
                        "Exp_Return(%)": round(exp_return_pct, 2),
                        "Action": action,
                        "Cash": round(cash, 2),
                        "Coins": round(coin, 4),
                        "Total_Value": round(total_value, 2)
                    })
                
                final_pf = portfolio_history[-1]
                final_bh = buy_hold_history[-1]
                
                pf_return = (final_pf - initial_capital) / initial_capital * 100
                bh_return = (final_bh - initial_capital) / initial_capital * 100
                
                c_res1, c_res2 = st.columns(2)
                c_res1.metric("Total Cumulative Return (누적 수익률)", f"{pf_return:.2f}%", f"${final_pf:,.0f}")
                c_res2.metric("Buy & Hold Return (단순 보유)", f"{bh_return:.2f}%", f"${final_bh:,.0f}")
                st.caption("ℹ️ **단순 보유(Buy & Hold)**: 비교군(Benchmark)은 시뮬레이션 시작일에 자본금으로 비트코인을 전량 매수하여 종료일까지 매도하지 않고 보유했을 경우의 가치입니다.")
                
                df_res = pd.DataFrame(results)
                df_res['BuyHold'] = buy_hold_history
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_res['Date'], y=df_res['Total_Value'], name="TOBIT Strategy", line=dict(color='#58a6ff', width=3)))
                fig.add_trace(go.Scatter(x=df_res['Date'], y=df_res['BuyHold'], name="Buy & Hold", line=dict(color='#8b949e', width=2, dash='dot')))
                # [수정] 높이 350으로 축소
                fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, title="Backtest Performance (Cumulative Value)", margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("##### 📜 상세 거래 내역 (Daily Asset Log)")
                st.dataframe(df_res[['Date', 'Price', 'Exp_Return(%)', 'Action', 'Cash', 'Coins', 'Total_Value']], use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#8b949e; font-size:12px;'>TOBIT v2.1 | Deep Learning Time Series Forecasting</div>", unsafe_allow_html=True)
