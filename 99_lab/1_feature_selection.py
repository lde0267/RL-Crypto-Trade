import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# --- 0. 데이터 로드 및 전처리 ---
data_file = os.path.join("0_data", "updated.csv")
if not os.path.exists(data_file):
    print(f"'{data_file}' 파일을 찾을 수 없습니다. 'calculate_features.py'를 먼저 실행하세요.")
    exit()

print(f"'{data_file}' 파일을 불러옵니다.")
df = pd.read_csv(data_file, index_col='datetime', parse_dates=True)

# --- 1. X, Y 정의 및 데이터 클리닝 ---

# X(특성)에 사용하지 않을 컬럼들
# (Target과 원본 OHLCV, 60분봉 OHLCV 등)
drop_cols = [
    'Target', 
    'Open', 'High', 'Low', 'Close', 'Volume',
    '60_Open', '60_High', '60_Low', '60_Close', '60_Volume'
]

# 위 컬럼을 제외한 모든 컬럼을 특성(X)으로 간주
X_cols = [col for col in df.columns if col not in drop_cols]
Y_col = 'Target'

# NaN, Inf 값 처리
df_clean = df.replace([np.inf, -np.inf], np.nan)
# X와 Y 컬럼에 NaN이 있는 행 전체 삭제
df_clean = df_clean.dropna(subset=X_cols + [Y_col])

X = df_clean[X_cols]
Y = df_clean[Y_col]

# 데이터가 충분한지 확인
if len(X) < 1000:
    print(f"경고: NaN/Inf 제거 후 샘플 수가 {len(X)}개로 너무 적습니다.")
    # exit() # (실행은 계속하도록 주석 처리)

print(f"총 {len(X_cols)}개의 특성으로 {len(X)}개의 샘플을 분석합니다.")

# (선택 사항) 스케일링: Tree 모델은 스케일링이 필수는 아니지만, 
# XGBoost의 경우 수렴 속도에 도움을 줄 수 있습니다.
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled = pd.DataFrame(X_scaled, columns=X_cols, index=X.index)

# (스케일링 없이 원본 X 사용)

# --- 2. 훈련/테스트 데이터 분리 ---
# 중요도 계산은 훈련 데이터로만 수행하는 것이 정석입니다.
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# --- 3. 모델 훈련 ---
print("\nRandom Forest 모델 훈련 중...")
# n_estimators: 나무의 개수, n_jobs=-1: 모든 CPU 코어 사용
rf_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1,
    # class_weight='balanced' # Target 불균형이 심할 경우 사용
)
rf_model.fit(X_train, y_train)

print("XGBoost 모델 훈련 중...")
# scale_pos_weight: Target 불균형 처리 (Y=0 개수 / Y=1 개수)
# (Y=1(성공)이 매우 적으므로 불균형 처리를 권장합니다)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = XGBClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1,
    eval_metric='logloss', # 경고 메시지 제거
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight 
)
xgb_model.fit(X_train, y_train)

print("모델 훈련 완료.")

# --- 4. 특성 중요도 추출 및 취합 ---

# 1. 랜덤 포레스트 중요도
rf_imp = pd.Series(rf_model.feature_importances_, index=X_cols)

# 2. XGBoost 중요도
xgb_imp = pd.Series(xgb_model.feature_importances_, index=X_cols)

# 3. 데이터프레임으로 합치기
imp_df = pd.DataFrame({
    'RF_Importance': rf_imp,
    'XGB_Importance': xgb_imp
})

# 4. 정규화 (Min-Max Scaling): 두 모델의 스케일이 다르므로 0~1 사이로 통일
imp_df['RF_Norm'] = (imp_df['RF_Importance'] - imp_df['RF_Importance'].min()) / \
                    (imp_df['RF_Importance'].max() - imp_df['RF_Importance'].min())
imp_df['XGB_Norm'] = (imp_df['XGB_Importance'] - imp_df['XGB_Importance'].min()) / \
                     (imp_df['XGB_Importance'].max() - imp_df['XGB_Importance'].min())

# 5. 평균 중요도 계산
imp_df['Average_Norm'] = (imp_df['RF_Norm'] + imp_df['XGB_Norm']) / 2

# --- 5. 최종 결과 출력 ---

# 평균 중요도 기준으로 상위 20개 선별
top_20_features = imp_df.sort_values(by='Average_Norm', ascending=False).head(25)

print("\n--- [최종 특성 중요도 TOP 20 (RF + XGB 평균)] ---")
print(top_20_features)

# (참고) 각 모델별 Top 20
print("\n--- [참고: RF 개별 Top 20] ---")
print(imp_df.sort_values(by='RF_Importance', ascending=False).head(20)['RF_Importance'])
print("\n--- [참고: XGB 개별 Top 20] ---")
print(imp_df.sort_values(by='XGB_Importance', ascending=False).head(20)['XGB_Importance'])

# 상위 20개 리스트만 따로 추출
top_20_list = top_20_features.index.tolist()
print(f"\n[RL 모델에 사용할 상위 20개 특성 리스트]:\n{top_20_list}")

# (선택 사항) 그래프 그리기
plt.figure(figsize=(10, 8))
top_20_features['Average_Norm'].sort_values(ascending=True).plot(kind='barh')
plt.title('Top 20 Feature Importance (RF + XGBoost Average)')
plt.xlabel('Normalized Importance Score')
plt.show()