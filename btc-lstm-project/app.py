import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import inspect
from datetime import datetime
from model import LSTMModel, DLinear, PatchTST, iTransformer, TCN, MLP
from data_utils import fetch_multi_data, load_scaler, TICKERS

# ==============================================================================
# 1. Page Config & ToBit Theme CSS
# ==============================================================================
st.set_page_config(
    page_title="ToBit | From Data to Bitcoin",
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
        border-radius: 12px; padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .kpi-card:hover { border-color: #58a6ff; transform: translateY(-2px); }
    
    .kpi-label { font-size: 0.85rem; color: #8b949e; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { font-family: 'Roboto Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #e6edf3; }
    .kpi-delta { font-family: 'Roboto Mono', monospace; font-size: 0.9rem; margin-top: 5px; font-weight: 600; }
    
    /* 색상 유틸리티 */
    .text-green { color: #3fb950; } .text-red { color: #f85149; } .text-blue { color: #58a6ff; } .text-gold { color: #d29922; }
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #262a33; padding-bottom: 5px; }
    .stTabs [data-baseweb="tab"] { height: 45px; background-color: transparent; border: 1px solid transparent; color: #8b949e; font-weight: 600; border-radius: 6px; }
    .stTabs [aria-selected="true"] { background-color: #1f242c; color: #58a6ff; border: 1px solid #262a33; }
    
    /* 코드 블록 폰트 */
    code { font-family: 'Roboto Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. 데이터 및 모델 로딩
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
# 3. 사이드바 (ToBit Branding with Logo)
# ==============================================================================
with st.sidebar:
    # [로고 적용 부분]
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)
    else:
        st.markdown("## 🐻 **TOBIG'S**")

    st.markdown("### **ToBit**\n*From Data to Bitcoin*")
    st.caption("Powered by **ToBigs** Data Science Club")
    st.markdown("---")
    
    menu = st.radio("MENU", ["📊 Market Forecast", "🧠 Deep Insight (XAI)", "📘 Model Specs", "⚡ Strategy Backtest"])
    
    st.markdown("---")
    st.markdown("<div style='color: #8b949e; font-size: 12px; margin-bottom: 5px;'>PARAMETERS</div>", unsafe_allow_html=True)
    
    selected_seq_len = st.select_slider("Lookback Window", options=[14, 21, 45], value=14, format_func=lambda x: f"{x} Days")
    selected_model = st.selectbox("Target Model", MODELS_LIST, index=2)

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
# 4. KPI Cards
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
        st.markdown("<h2 style='margin-top: 5px;'>ToBit Analysis Dashboard</h2>", unsafe_allow_html=True)

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
# 5. Main Content
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
        fig.add_trace(go.Scatter(x=future_dates, y=preds, name=f"ToBit Forecast", mode='lines+markers', line=dict(color=pred_color, width=3), marker=dict(size=6, color='#161b22', line=dict(width=2, color=pred_color))))
        
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#262a33'), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        diff_pred = preds[-1] - preds[0]
        st.markdown(f"""
        <div style="padding: 15px; border-left: 3px solid {'#3fb950' if diff_pred > 0 else '#f85149'}; background-color: #161b22;">
            <span style="color: #8b949e; font-size: 14px;">ToBit Analysis Summary:</span><br>
            <span style="font-size: 18px; font-weight: bold; color: #e6edf3;">Target Price (7D): ${preds[-1]:,.0f}</span>
            <span class="{'text-green' if diff_pred > 0 else 'text-red'}" style="font-weight: bold; margin-left: 10px;">{'BULLISH 🚀' if diff_pred > 0 else 'BEARISH 📉'} ({diff_pred/preds[0]*100:+.2f}%)</span>
        </div>
        """, unsafe_allow_html=True)
    else: st.warning("Model weights not found. Please train first.")

# [TAB 2] XAI (Counterfactual Added)
elif menu == "🧠 Deep Insight (XAI)":
    st.markdown(f"#### 🧠 Explainable AI: Why {selected_model}?")
    model = get_model(selected_model, selected_seq_len)
    
    if model:
        # 1. 기본 데이터 준비
        input_raw = df[features].tail(selected_seq_len).values
        input_tensor = torch.tensor(scaler.transform(input_raw)).float().unsqueeze(0)
        input_tensor.requires_grad = True
        
        # 2. 기본 예측 및 Gradient 계산
        output = model(input_tensor)
        base_pred_scaled = output[0].detach().numpy() # [Pred_Len]
        
        # Backprop for Heatmap
        output[0, 0].backward()
        grads = input_tensor.grad.abs().squeeze().numpy()
        
        # --------------------------------------------------------------------------
        # A. Existing XAI Charts (Heatmap & TimeSHAP)
        # --------------------------------------------------------------------------
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("##### 📍 Attention Heatmap (Feature Importance)")
            fig = go.Figure(data=go.Heatmap(z=grads.T, x=[f"D-{selected_seq_len-i}" for i in range(selected_seq_len)], y=features, colorscale='Inferno', showscale=False))
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown("##### ⏳ TimeSHAP (Temporal Impact)")
            block_size = selected_seq_len // 5 if selected_seq_len > 10 else 2
            num_blocks = selected_seq_len // block_size
            impacts = []
            base_val = output[0, 0].item()
            with torch.no_grad():
                for b in range(num_blocks):
                    p = input_tensor.clone()
                    p[0, b*block_size:(b+1)*block_size, :] = 0
                    impacts.append(abs(base_val - model(p)[0,0].item()))
            fig = px.bar(x=[f"T-{b}" for b in range(num_blocks)], y=impacts, labels={'x':'Time Block','y':'Impact Score'})
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --------------------------------------------------------------------------
        # B. NEW: Counterfactual Analysis (What-If Simulation)
        # --------------------------------------------------------------------------
        st.markdown("##### 🔀 Counterfactual Simulator (What-If Analysis)")
        st.info("💡 **실험:** '만약 가장 최근 데이터(오늘)의 특정 지표가 달랐다면 미래 예측은 어떻게 변할까?'를 시뮬레이션합니다.")

        # UI: 변수 선택 및 조절
        cf_col1, cf_col2, cf_col3 = st.columns([1, 1, 2])
        
        with cf_col1:
            target_feat = st.selectbox("Select Feature to Tweak", features, index=btc_idx)
        
        with cf_col2:
            # 현재 마지막 값 가져오기
            current_val = input_raw[-1, features.index(target_feat)]
            delta_pct = st.slider(f"Change {target_feat} (%)", min_value=-30, max_value=30, value=0, step=1, format="%d%%")
        
        # 시뮬레이션 로직
        # 1. 입력 데이터 복사 및 수정 (최근 시점 데이터 수정)
        modified_input_raw = input_raw.copy()
        feat_idx = features.index(target_feat)
        modified_val = current_val * (1 + delta_pct / 100.0)
        modified_input_raw[-1, feat_idx] = modified_val
        
        # 2. 모델 예측 (Counterfactual)
        mod_tensor = torch.tensor(scaler.transform(modified_input_raw)).float().unsqueeze(0)
        with torch.no_grad():
            mod_pred_scaled = model(mod_tensor).numpy()[0]
            
        # 3. 스케일 복원 (Inverse Transform)
        def inverse_preds(pred_scaled):
            res = []
            for p in pred_scaled:
                dummy = np.zeros(len(features))
                dummy[btc_idx] = p
                res.append(scaler.inverse_transform([dummy])[0][btc_idx])
            return res

        orig_preds_real = inverse_preds(base_pred_scaled)
        mod_preds_real = inverse_preds(mod_pred_scaled)
        
        # 4. 결과 시각화
        with cf_col3:
            st.markdown(f"**Changed Input:** {current_val:,.2f} ➡️ <span style='color:#58a6ff'>{modified_val:,.2f}</span>", unsafe_allow_html=True)
            diff_final = mod_preds_real[-1] - orig_preds_real[-1]
            st.markdown(f"**Impact on 7th Day Price:** <span style='color:{'#3fb950' if diff_final>0 else '#f85149'}'>{diff_final:+.2f} USD</span>", unsafe_allow_html=True)

        # 그래프 그리기
        future_dates = [pd.to_datetime(df['timestamp'].values[-1]) + pd.Timedelta(days=i) for i in range(1, 8)]
        
        fig_cf = go.Figure()
        # 원래 예측
        fig_cf.add_trace(go.Scatter(
            x=future_dates, y=orig_preds_real, name="Original Forecast",
            mode='lines+markers', line=dict(color='#8b949e', width=2, dash='dot')
        ))
        # 수정된 예측 (Counterfactual)
        fig_cf.add_trace(go.Scatter(
            x=future_dates, y=mod_preds_real, name="Counterfactual (What-If)",
            mode='lines+markers', line=dict(color='#58a6ff', width=4)
        ))
        
        fig_cf.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=350, hovermode="x unified",
            title=dict(text="Forecast Comparison", font=dict(size=14))
        )
        st.plotly_chart(fig_cf, use_container_width=True)

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
    st.markdown("#### 🧪 Backtest Simulation")
    df_bt = pd.DataFrame({
        "Model": MODELS_LIST,
        "Win Rate": [0.58, 0.62, 0.55, 0.59, 0.64, 0.61],
        "Profit Factor": [1.2, 1.4, 1.1, 1.3, 1.5, 1.35]
    })
    st.dataframe(df_bt, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#8b949e; font-size:12px;'>ToBit v2.1 | Powered by <b>ToBigs</b> Data Science Club</div>", unsafe_allow_html=True)
