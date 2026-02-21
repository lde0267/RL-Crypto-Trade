import pandas as pd
import statsmodels.api as sm
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# --- 0. 전처리된 데이터 로드 ---
data_file = "0_data/btc_updated.csv"
if not os.path.exists(data_file):
    print(f"'{data_file}' 파일을 찾을 수 없습니다. 'calculate_features.py'를 먼저 실행하세요.")
    exit()

print(f"'{data_file}' 파일을 불러옵니다.")
# ★ 중요: 저장 시 'datetime' 인덱스가 파일에 포함되었으므로 index_col로 지정합니다.
df = pd.read_csv(data_file, index_col='datetime', parse_dates=True)

# --- 1. 통계 분석 (RL 환경 모델) ---
print("\n1. 통계 분석 (RL 환경에 사용할 8개 지표)")

# ★ 수정: 요청하신 6개의 지표 리스트
rl_model_cols = [
            '30_to_60_Close_ratio', 
            '60_OBV', 
            'day_of_week', 
            '30_ATR', 
            '30_Keltner_lband', 
            '60_ADX'
        ]  

# ★ 수정: 6개 지표 + Target을 기준으로 NaN 및 inf 값 제거
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=rl_model_cols + ['Target'])

X = df_clean[rl_model_cols]
Y = df_clean['Target']

# (★ 2. 스케일링 적용 ★)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# (★ 3. 상수항 추가 ★)
# statsmodels는 상수항(절편)을 수동으로 추가
X_with_const = sm.add_constant(X_scaled)
# ★ 수정: 8개 지표 기준으로 컬럼 이름 복원
X_with_const = pd.DataFrame(X_with_const, columns=['const'] + rl_model_cols, index=Y.index)


print(f"총 {len(Y)}개의 샘플로 통계 분석을 실행합니다. (선택된 지표: {len(rl_model_cols)}개, 스케일링 완료)")

try:
    # (★ 4. 로지스틱 회귀 모델 피팅 ★)
    model = sm.Logit(Y, X_with_const.astype(float))
    results = model.fit(method='lbfgs') # lbfgs solver 사용 (수렴 안정성)
    print(results.summary())
except Exception as e:
    print(f"통계 모델 피팅 중 오류 발생: {e}")