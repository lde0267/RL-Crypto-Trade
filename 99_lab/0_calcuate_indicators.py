import pandas as pd
import ta
import numpy as np
import os
import talib

# --- 0. 데이터 로드 ---
data_file = os.path.join("0_data", "btc_ohlcv_30min.csv")
if not os.path.exists(data_file):
    print(f"'{data_file}' 파일을 찾을 수 없습니다. 'get_data.py'를 먼저 실행하세요.")
    exit()

print(f"'{data_file}' 파일을 불러옵니다.")
df = pd.read_csv(data_file, parse_dates=['datetime'])
df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
df = df.set_index('datetime')

# --- 1. 60분봉 리샘플링 및 병합 ---
print("60분봉 데이터를 리샘플링하여 병합합니다.")
logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
df_60 = df.resample('60min', closed='left', label='left').agg(logic).dropna()
df_60 = df_60.add_prefix('60_')
df_merged = pd.merge_asof(df, df_60, left_index=True, right_index=True, direction='backward')
# ffill()을 사용하여 병합 시 발생할 수 있는 초기 NaN 값을 채웁니다.
df = df_merged.fillna(method='ffill') 

# --- 2. Target(Y) 정의 (트리플 배리어 방식) ---
print("Target(Y)을 '트리플 배리어' 방식으로 정의합니다.")
PROFIT_TARGET = 0.01  # +1% 익절
STOP_LOSS = 0.032      # -3.2% 손절
N_PERIODS = 48          # 24시간 (30분 * 24) 시간 제한

entry_prices = df['Close']
upper_barriers = entry_prices * (1 + PROFIT_TARGET)
lower_barriers = entry_prices * (1 - STOP_LOSS)
outcomes = pd.Series(0, index=df.index) # 변수 초기화

# shift(-i)를 사용하여 미래 데이터를 조회합니다.
for i in range(1, N_PERIODS + 1):
    future_high = df['High'].shift(-i)
    future_low = df['Low'].shift(-i)
    
    # 아직 결과(outcomes)가 0인 경우에만 업데이트
    loss_hit = (future_low <= lower_barriers) & (outcomes == 0)
    outcomes[loss_hit] = -1
    profit_hit = (future_high >= upper_barriers) & (outcomes == 0)
    outcomes[profit_hit] = 1

df['Target'] = (outcomes == 1).astype(int)
print(f"Target(Y) 정의 완료. (성공률: {df['Target'].mean():.2%})")

# --- 3A. 기존 'Elite' 지표 계산 ---
print("기존 'Elite' 지표들을 계산합니다...")
# (A) 모델 1
df['30_Close'] = df['Close']
df['30_relative_volume'] = df['Volume'] / (df['Volume'].rolling(window=20).mean() + 1e-6)
df['60_ATR'] = ta.volatility.average_true_range(df['60_High'], df['60_Low'], df['60_Close'], window=14)
df['30_Close_t-1'] = df['30_Close'].shift(1)
# (B) 모델 2
df['ha_close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
df['ha_open'] = np.nan
df.iloc[0, df.columns.get_loc('ha_open')] = (df.iloc[0]['Open'] + df.iloc[0]['Close']) / 2
for i in range(1, len(df)):
    df.iloc[i, df.columns.get_loc('ha_open')] = (df.iloc[i-1]['ha_open'] + df.iloc[i-1]['ha_close']) / 2
aroon = ta.trend.AroonIndicator(df['High'], df['Low'], window=25) 
df['30_AROON_down'] = aroon.aroon_down()
df['30_RSI'] = ta.momentum.rsi(df['Close'], window=14)
df['30_MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)
df['30_Keltner_hband'] = keltner.keltner_channel_hband()
df['30_Keltner_lband'] = keltner.keltner_channel_lband()
df['30_Force_Index'] = ta.volume.force_index(df['Close'], df['Volume'], window=13)
df['60_RSI'] = ta.momentum.rsi(df['60_Close'], window=14)
df['60_ADX'] = ta.trend.adx(df['60_High'], df['60_Low'], df['60_Close'], window=14)
print("기존 'Elite' 특성(X) 계산 완료.")

# --- 3B. 50+ 확장 지표 추가 ---
print("50개 이상의 확장 지표 계산을 시작합니다...")

# === (1) 캔들 패턴 지표 (30분봉) ===
print("... (1/6) 캔들 패턴 지표 계산 중 ...")

# [수정] 캔들 패턴(CDL...)을 캔들의 '수치적 특성'으로 분해합니다.
# 이는 '긴 꼬리', '짧은 꼬리', '몸통 크기' 등을 숫자로 표현합니다.
candle_range = df['High'] - df['Low'] + 1e-9 # (0으로 나누기 방지를 위해 아주 작은 값(epsilon)을 더함)
body_size_abs = (df['Close'] - df['Open']).abs() # 1. 캔들 범위 대비 몸통(Body) 크기 비율 (0.0 ~ 1.0)
df['30_body_ratio'] = body_size_abs / candle_range

upper_wick = df['High'] - np.maximum(df['Open'], df['Close']) # 2. 캔들 범위 대비 위 꼬리(Upper Wick) 비율 (0.0 ~ 1.0)
df['30_upper_wick_ratio'] = upper_wick / candle_range

lower_wick = np.minimum(df['Open'], df['Close']) - df['Low'] # 3. 캔들 범위 대비 아래 꼬리(Lower Wick) 비율 (0.0 ~ 1.0)
df['30_lower_wick_ratio'] = lower_wick / candle_range

df['30_body_direction'] = np.sign(df['Close'] - df['Open']) # 4. 몸통의 방향 (+1: 상승, -1: 하락)

sma_20 = ta.trend.sma_indicator(df['Close'], window=20) # [추가] '이격도' (Disparity) - 사용자가 언급한 용어
df['30_disparity_20'] = (df['Close'] - sma_20) / (sma_20 + 1e-9)

# === (2) 모멘텀 지표 ===
print("... (2/6) 모멘텀 지표 계산 중 ...")
df['30_Awesome_Oscillator'] = ta.momentum.awesome_oscillator(df['High'], df['Low']) 
stoch_rsi = ta.momentum.StochRSIIndicator(df['Close'], window=14, smooth1=3, smooth2=3)
df['30_Stoch_RSI'] = stoch_rsi.stochrsi() 
df['30_Stoch_RSI_K'] = stoch_rsi.stochrsi_k() 
df['30_Stoch_RSI_D'] = stoch_rsi.stochrsi_d() 
stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
df['30_Stoch'] = stoch.stoch() 
df['30_Stoch_Signal'] = stoch.stoch_signal() 
df['30_Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14) 
df['30_Ultimate_Oscillator'] = ta.momentum.ultimate_oscillator(df['High'], df['Low'], df['Close']) 
df['30_ROC'] = ta.momentum.roc(df['Close'], window=12) 
df['60_Stoch_RSI'] = ta.momentum.stochrsi(df['60_Close'], window=14) 
stoch_60 = ta.momentum.StochasticOscillator(df['60_High'], df['60_Low'], df['60_Close'], window=14, smooth_window=3)
df['60_Stoch'] = stoch_60.stoch() 
df['60_Stoch_Signal'] = stoch_60.stoch_signal() 
df['60_Williams_R'] = ta.momentum.williams_r(df['60_High'], df['60_Low'], df['60_Close'], lbp=14) 

# === (3) 거래량 지표 ===
print("... (3/6) 거래량 지표 계산 중 ...")
df['30_OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume']) 
df['30_CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20) 
df['30_EOM'] = ta.volume.ease_of_movement(df['High'], df['Low'], df['Volume'], window=14) 
df['30_ADI'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume']) 
df['30_VPT'] = ta.volume.volume_price_trend(df['Close'], df['Volume']) 
df['60_OBV'] = ta.volume.on_balance_volume(df['60_Close'], df['60_Volume']) 
df['60_CMF'] = ta.volume.chaikin_money_flow(df['60_High'], df['60_Low'], df['60_Close'], df['60_Volume'], window=20) 

# === (4) 추세 지표 ===
print("... (4/6) 추세 지표 계산 중 ...")
macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
df['30_MACD'] = macd.macd() 
df['30_MACD_Signal'] = macd.macd_signal() 
df['30_MACD_Hist'] = macd.macd_diff() 
df['30_Aroon_Up'] = aroon.aroon_up() 
df['30_Aroon_Indicator'] = aroon.aroon_indicator()
df['30_CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20) 
df['30_DPO'] = ta.trend.dpo(df['Close'], window=20) 
macd_60 = ta.trend.MACD(df['60_Close'], window_slow=26, window_fast=12, window_sign=9)
df['60_MACD'] = macd_60.macd() 
df['60_MACD_Signal'] = macd_60.macd_signal() 
df['60_CCI'] = ta.trend.cci(df['60_High'], df['60_Low'], df['60_Close'], window=20) 

# === (5) 변동성 지표 ===
print("... (5/6) 변동성 지표 계산 중 ...")
df['30_ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14) 
bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['30_BB_High'] = bb.bollinger_hband() 
df['30_BB_Low'] = bb.bollinger_lband() 
df['30_BB_Width'] = bb.bollinger_wband() 
dc = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'], window=20)
df['30_Donchian_High'] = dc.donchian_channel_hband() 
df['30_Donchian_Low'] = dc.donchian_channel_lband() 
bb_60 = ta.volatility.BollingerBands(df['60_Close'], window=20, window_dev=2)
df['60_BB_High'] = bb_60.bollinger_hband() 
df['60_BB_Low'] = bb_60.bollinger_lband() 
df['60_BB_Width'] = bb_60.bollinger_wband() 

# === (6) 수동/가격 기반 지표 ===
print("... (6/6) 수동/가격/시간 지표 계산 중 ...")
df['30_return_1'] = df['Close'].pct_change(1) 
df['30_return_3'] = df['Close'].pct_change(3) 
df['30_return_6'] = df['Close'].pct_change(6) 
df['30_to_60_Close_ratio'] = df['Close'] / (df['60_Close'] + 1e-6) 
df['30_high_low_spread'] = (df['High'] - df['Low']) / (df['Close'] + 1e-6) 
df['30_close_open_spread'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-6) 
df['hour_of_day'] = df.index.hour 
df['day_of_week'] = df.index.dayofweek 

print("모든 확장 특성(X) 계산 완료.")

# --- 💡 [수정] NaN 값 디버깅 코드 추가 ---
print("\n--- [NaN 값 디버깅 시작] ---")
total_rows = len(df)
print(f"NaN 제거 전 원본 데이터 행 수: {total_rows}")

# 1. 각 열의 NaN 개수 계산
nan_counts = df.isna().sum()

# 2. 100% NaN인 열 (범인) 찾기
all_nan_cols = nan_counts[nan_counts == total_rows].index.tolist()

if len(all_nan_cols) > 0:
    print(f"\n[!!!] 치명적 오류: 다음 {len(all_nan_cols)}개 열은 100% NaN입니다. (전체 행: {total_rows}개)")
    print("이 지표들의 계산 로직을 확인하거나 주석 처리하세요:")
    for col in all_nan_cols:
        print(f"- {col}")
    
    # (참고) 100% NaN은 아니지만 NaN이 많은 상위 10개 열
    print("\n(참고) NaN이 많은 상위 10개 열:")
    print(nan_counts.sort_values(ascending=False).head(10))
    
    print("\n디버깅을 위해 스크립트를 중단합니다.")
    exit() # <-- 여기서 중단하여 범인을 확인
else:
    print("✓ 100% NaN인 열을 찾지 못했습니다. 일반 NaN 제거를 계속합니다.")
# --- [디버깅 코드 끝] ---


# (기존 코드) 모든 지표 계산이 완료된 후, NaN을 포함한 행을 모두 제거합니다.
df = df.dropna()
print(f"NaN 제거 후 최종 데이터 행 수: {len(df)}")

if len(df) == 0:
    # 이 메시지가 보인다면, 100% NaN인 열은 없지만,
    # 여러 열의 NaN이 조합되어 모든 행이 삭제되었다는 의미입니다.
    print("치명적 오류: 100% NaN 열은 없었으나, NaN 조합으로 인해 모든 행이 제거되었습니다.")
    print("데이터 시작 부분의 NaN이 너무 많을 수 있습니다.")
    print("\n(참고) NaN이 많은 상위 10개 열:")
    print(nan_counts.sort_values(ascending=False).head(10))
    exit()

# --- 4. 데이터 저장 ---
output_file = "0_data/updated.csv"
df.to_csv(output_file, index=True) 
print(f"모든 특성과 Target이 포함된 데이터프레임을 '{output_file}' 파일로 저장했습니다.")