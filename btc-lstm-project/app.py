import streamlit as st
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
# 현재 app.py가 있는 폴더를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 기존 프로젝트 파일에서 모델 클래스와 데이터 처리 함수 임포트
# (사용자님의 model.py와 data_utils.py 내용에 따라 수정이 필요할 수 있습니다)
from model import LSTMModel, DLinearModel, PatchTSTModel, TCNModel, iTransformerModel
from data_utils import prepare_data, inverse_transform

# --- 1. 경로 설정 (가장 중요한 부분) ---
# app.py 파일이 위치한 디렉토리를 기준으로 절대 경로를 설정합니다.
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"

# --- 2. 모델 로드 함수 (캐싱 적용) ---
@st.cache_resource
def get_model(name):
    """모델 이름에 따라 객체를 생성하고 가중치를 로드합니다."""
    # 1. 모델 객체 생성 (사용자님의 model.py 정의에 맞춰 파라미터 수정 필요)
    if name == "LSTM":
        model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
    elif name == "DLinear":
        model = DLinearModel(seq_len=96, pred_len=24)
    elif name == "PatchTST":
        model = PatchTSTModel()
    elif name == "TCN":
        model = TCNModel()
    elif name == "iTransformer":
        model = iTransformerModel()
    else:
        # 기본 'model.pth' 처리
        model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)

    # 2. 가중치 파일 경로 확인 및 로드
    weight_path = WEIGHTS_DIR / f"{name}.pth"
    
    if not weight_path.exists():
        st.error(f"모델 파일을 찾을 수 없습니다: {weight_path}")
        return None

    try:
        # Streamlit Cloud 환경을 위해 cpu로 매핑하여 로드
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None

# --- 3. Streamlit UI 레이아웃 ---
st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")
st.title("📈 Bitcoin Price Prediction Dashboard")
st.sidebar.header("설정")

# 모델 선택 (GitHub의 weights 폴더 내 파일명 기준)
model_option = st.sidebar.selectbox(
    "사용할 모델을 선택하세요",
    ["LSTM", "DLinear", "PatchTST", "TCN", "iTransformer"]
)

# 데이터 불러오기 및 예측 버튼
if st.sidebar.button("예측 실행"):
    with st.spinner(f"{model_option} 모델로 예측 중..."):
        # 1. 모델 로드
        model = get_model(model_option)
        
        if model:
            # 2. 데이터 준비 (data_utils.py 활용)
            # 여기서는 예시 코드로 작성되었습니다. 실제 입력 텐서 준비 로직을 넣어주세요.
            # input_tensor = prepare_data() 
            
            # 임의의 데이터 시뮬레이션 (테스트용)
            input_tensor = torch.randn(1, 96, 1) 
            
            # 3. 예측 수행
            with torch.no_grad():
                preds_scaled = model(input_tensor).numpy()[0]
            
            # 4. 결과 시각화
            st.subheader(f"Results: {model_option}")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(preds_scaled, label='Predicted Price', color='orange')
            ax.set_title(f"Bitcoin Price Forecast ({model_option})")
            ax.legend()
            st.pyplot(fig)
            
            st.success("예측이 완료되었습니다!")

# --- 4. 디버깅 정보 (필요시 사이드바 하단에 표시) ---
if st.sidebar.checkbox("디버깅 경로 확인"):
    st.sidebar.write(f"BASE_DIR: {BASE_DIR}")
    st.sidebar.write(f"WEIGHTS_DIR: {WEIGHTS_DIR}")
    if WEIGHTS_DIR.exists():
        st.sidebar.write("존재하는 가중치 파일:", os.listdir(WEIGHTS_DIR))


