import sys
import os
from pathlib import Path

# --- 1. Python 경로 설정 ---
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 2. 로컬 모듈 임포트 (실제 클래스명과 일치시킴) ---
try:
    # model.py에 정의된 정확한 클래스 이름들입니다.
    from model import LSTMModel, DLinear, PatchTST, iTransformer, TCN
except ImportError as e:
    st.error(f"모듈 로드 실패: {e}")
    raise e

# --- 3. 모델 로드 함수 ---
@st.cache_resource
def get_model(name):
    # model.py의 __init__ 파라미터 기본값에 맞춰 생성합니다.
    if name == "LSTM":
        return LSTMModel(input_size=8, hidden_size=128, num_layers=3, output_size=7)
    elif name == "DLinear":
        return DLinear(seq_len=120, pred_len=7, input_size=8)
    elif name == "PatchTST":
        return PatchTST(input_size=8, seq_len=120, pred_len=7)
    elif name == "iTransformer":
        return iTransformer(seq_len=120, pred_len=7, input_size=8)
    elif name == "TCN":
        return TCN(input_size=8, output_size=7)
    return None

# --- 4. Streamlit UI ---
st.set_page_config(page_title="BTC Price Prediction", layout="wide")
st.title("📈 Bitcoin Price Prediction")

# 사이드바 설정
model_option = st.sidebar.selectbox(
    "모델 선택",
    ["LSTM", "DLinear", "PatchTST", "iTransformer", "TCN"]
)

# 경로 설정
WEIGHTS_DIR = BASE_DIR / "weights"

if st.sidebar.button("예측 실행"):
    with st.spinner(f"{model_option} 로딩 중..."):
        model = get_model(model_option)
        
        # 가중치 파일 로드
        weight_path = WEIGHTS_DIR / f"{model_option}.pth"
        if weight_path.exists():
            model.load_state_dict(torch.load(weight_path, map_location='cpu'))
            model.eval()
            st.success(f"{model_option} 모델 로드 완료!")
            
            # (여기에 예측 및 시각화 로직 추가 가능)
            st.info("예측 결과 시각화 준비 중...")
        else:
            st.error(f"가중치 파일을 찾을 수 없습니다: {weight_path.name}")

# 디버깅용 정보
if st.sidebar.checkbox("시스템 경로 확인"):
    st.write("BASE_DIR:", BASE_DIR)
    st.write("파일 목록:", os.listdir(BASE_DIR))
