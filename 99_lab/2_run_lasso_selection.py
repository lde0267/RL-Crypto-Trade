import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

# 경고 메시지 무시 (스케일링 관련)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 0. 데이터 로드 및 전처리 ---
data_file = os.path.join("0_data", "updated.csv")
if not os.path.exists(data_file):
    print(f"'{data_file}' 파일을 찾을 수 없습니다. 'calculate_features.py'를 먼저 실행하세요.")
    exit()

print(f"'{data_file}' 파일을 불러옵니다.")
df = pd.read_csv(data_file, index_col='datetime', parse_dates=True)

# 24시간/1.0% 모델의 Top 20 리스트
top_20_list = ['60_OBV', '60_BB_Width', '30_VPT', '30_ADI', 'day_of_week', 
               '30_OBV', '30_to_60_Close_ratio', '30_Donchian_Low', 
               '30_BB_Low', '30_Donchian_High', '30_Keltner_lband', 
               '60_ATR', '30_Keltner_hband', 'ha_close', '60_BB_Low', 
               '30_BB_High', '30_ATR', '30_Close', '60_BB_High', 
               '30_BB_Width', 'ha_open', '30_Close_t-1', '30_MACD_Signal',
                 '60_MACD_Signal', '60_ADX']
Y_col = 'Target'

# NaN, Inf 값 처리
df_clean = df.replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.dropna(subset=top_20_list + [Y_col])

Y = df_clean[Y_col]
X_raw = df_clean[top_20_list]

# --- 1. 스케일링 (로지스틱 회귀에는 필수!) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_scaled_df = pd.DataFrame(X_scaled, columns=top_20_list, index=Y.index)

print(f"총 {len(X_scaled_df)}개의 샘플로 L1(Lasso) 규제 탐색을 시작합니다.\n")

# --- 2. L1(Lasso) 규제를 통한 반복 탐색 ---
# 'C' 값은 규제의 역수입니다. (C가 작을수록 규제가 강해져 변수가 적게 남습니다)
# C=1.0 (약한 규제) -> C=0.01 (강한 규제) 순으로 테스트
c_values_to_test = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]

print("--- L1(Lasso) 페널티 강도별 선택된 특성 ---")
for c_val in c_values_to_test:
    # L1(Lasso) 로지스틱 회귀 모델 선언
    # penalty='l1' : L1 규제 사용
    # solver='liblinear' : L1 규제를 지원하는 솔버
    # C=c_val : 규제 강도 설정
    lasso_model = LogisticRegression(
        penalty='l1', 
        solver='liblinear', 
        C=c_val, 
        random_state=42
    )
    
    # 훈련
    lasso_model.fit(X_scaled_df, Y)
    
    # 결과 추출 (계수(coef_)가 0이 아닌 변수들만 "살아남음")
    coefficients = lasso_model.coef_[0]
    
    # 0이 아닌 계수를 가진 변수들만 필터링
    selected_features = [
        name for name, coef in zip(top_20_list, coefficients) if coef != 0
    ]
    
    print(f"\n[C = {c_val:.2f}] (규제강도: {1/c_val:.1f}x)")
    print(f"  선택된 특성 개수: {len(selected_features)}")
    if selected_features:
        print(f"  선택된 특성 목록: {selected_features}")
    else:
        print("  선택된 특성 목록: (없음 - 규제가 너무 강합니다)")

print("\n--- 탐색 완료 ---")
print("C값을 조절하여 원하는 개수의 최적 조합을 찾을 수 있습니다.")
print("(예: 5~7개 정도의 변수가 남는 C값을 선택)")