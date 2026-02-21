# config.py
import os

"""
프로젝트의 모든 설정값을 관리하는 파일입니다.
"""
# --- API Keys for my dorm---
# flALAWxxGA8kjgvqMIExtk5PoHkxbc11T55WiyFR /// access
# 8cQH2cFbf8eYjE5uP42MiX9SbBdoW0PQO8CJIuQw /// secret

# --- API Keys (실제 투자 시 사용) ---
UPBIT_ACCESS_KEY = "EfFN67T9DQykLNCmgLFi18stzPfJPOwmEh6QKEFy"
UPBIT_SECRET_KEY = "X85xogyQzkEoe8VRbGjivL5cIgcKZtvt0umLGgud"

# --- 코인 선별 조건 파라미터 (예시) ---
MIN_VOLUME_KRW = 5_000_000_000
MIN_VOLATILITY_PCT = 1.5
MIN_ADX = 20

# --- DB 설정 ---
DB_PATH = "trading_bot.db"

# 데이터 봉 인터벌 ('minute30', 'minute60', 'day' 등)
INTERVAL = "minute30"

# 데이터 조회 기간 (INTERVAL이 'minute30'일 경우, 48 * 30 = 30일치 데이터)
DATA_COUNT = 48 * 30

# 초기 자본금
INITIAL_BALANCE = 10000000

# --- Strategy Parameters ---
# 변동성 돌파 전략 K값 범위
K_MIN = 0.3
K_MAX = 1.2

TRADE_RATIO_1ST = 0.6  # 매수 비중 (0.0 ~ 1.0)
TRADE_RATIO_2ND = 0.5
TRADE_RATIO_3RD = 1.0

ADD_BUY_PCT_2ND = 1.0  # 2차 매수 진입 조건 (%)
ADD_BUY_PCT_3RD = 2.0  # 3차 매수 진입 조건 (%)

# 거래량 비율
VOLUME_RATIO = 1.2

# ATR Trailing Stop 배수
ATR_MULTIPLIER = 1.0

# 고정 손절매 비율 (%)
STOP_LOSS_PERCENT = 1.5

# 1차 매수 후 익절/수익보호 관련 비율 (%)
PROFIT_TARGET_PERCENT_1ST = 4.0  # 1차 매수 후 목표 익절
PROFIT_LOCK_START_PERCENT_1ST = 0.0 # 수익 보호 Trailing 시작 시점
PROFIT_LOCK_TRAILING_PERCENT_1ST = 0.5 # 수익 보호 Trailing 폭

# 동시에 보유할 수 있는 최대 코인(포지션)의 수
MAX_POSITIONS = 1

# --- RL 모델 설정 ---
RL_MODEL_PATH = "1_model/final_model.zip"
RL_STATS_PATH = "1_model/obs_stats_btc.pkl"
RL_WINDOW_SIZE = 10 
INITIAL_BALANCE = 300000.0 # ❗️ [필수] 훈련 시 사용한 self.initial_balance 값
OBS_COLS = [
    '30_to_60_Close_ratio', '60_OBV', 'day_of_week', 
    '30_ATR', '30_Keltner_lband', '60_ADX'
]