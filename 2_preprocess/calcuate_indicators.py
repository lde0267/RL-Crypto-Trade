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
STOP_LOSS = 0.01      # -1% 손절
N_PERIODS = 12          # 6시간 (30분 * 12) 시간 제한

entry_prices = df['Close']
upper_barriers = entry_prices * (1 + PROFIT_TARGET)
lower_barriers = entry_prices * (1 - STOP_LOSS)
outcomes = pd.Series(0, index=df.index)

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

keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)

df['30_Keltner_lband'] = keltner.keltner_channel_lband()
df['60_ADX'] = ta.trend.adx(df['60_High'], df['60_Low'], df['60_Close'], window=14)
df['60_OBV'] = ta.volume.on_balance_volume(df['60_Close'], df['60_Volume']) 
df['30_ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14) 
df['30_to_60_Close_ratio'] = df['Close'] / (df['60_Close'] + 1e-6) 
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
output_file = "0_data/btc_updated_6indi.csv"
df.to_csv(output_file, index=True) 
print(f"모든 특성과 Target이 포함된 데이터프레임을 '{output_file}' 파일로 저장했습니다.")