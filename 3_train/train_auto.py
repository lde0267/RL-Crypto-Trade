import os
import pandas as pd
import numpy as np
import pickle 
from stable_baselines3.common.vec_env import DummyVecEnv
# ⚠️ [삭제] VecNormalize는 더 이상 사용하지 않음
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from sb3_contrib import RecurrentPPO
from env_auto import TradingEnv # TradingEnv 클래스가 있는 모듈을 임포트

def train_agent():
    """강화학습 에이전트 학습 및 평가 파이프라인"""

    # ===== 1. 설정 (Configuration) =====
    # --- 파일 경로 ---
    data_path = "0_data/updated.csv"
    log_dir = "./0_logs/recurrent_ppo/" # 텐서보드 로그 경로를 모델 이름에 맞게 수정
    best_model_save_path = "./1_model/best_model_auto" # 모델 파일 이름 수정
    final_model_save_path = "./1_model/final_auto" # 모델 파일 이름 수정
    
    # ✅ [수정] 통계 파일 저장 경로
    stats_save_path = os.path.join(os.path.dirname(best_model_save_path), "obs_stats_btc_auto.pkl")

    # --- 학습 파라미터 ---
    total_timesteps = 1_000_000 
    episode_length = 48 
    window_size = 10 
    train_test_split_ratio = 0.8
    
    # --- 모델 하이퍼파라미터 ---
    policy_kwargs = dict(
        lstm_hidden_size=256,
        net_arch=dict(pi=[256], vf=[256]) 
    )
    model_params = {
        'policy': "MlpLstmPolicy",
        'learning_rate': 3e-5,
        'n_steps': 1024, 
        'batch_size': 64,
        'ent_coef': 0.01, 
        'verbose': 1,
        'tensorboard_log': log_dir,
        'policy_kwargs': policy_kwargs
    }
    
    # ✅ [수정] (Lasso C=0.01)로 선택된 9개 지표
    OBS_COLS = ['60_BB_Width', 
                '30_VPT', 
                '30_ADI', 
                'day_of_week', 
                '30_OBV', 
                '30_to_60_Close_ratio', 
                '30_BB_High', 
                '30_ATR', 
                '60_ADX']

    # ===== 2. 데이터 로드 및 분리 =====
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    
    # ✅ [수정] 결측치 제거
    df = df.dropna(subset=OBS_COLS + ['Close']) 
    df = df.reset_index(drop=True) # TradingEnv는 reset_index된 df를 사용함

    split_point = int(len(df) * train_test_split_ratio)
    
    train_df = df[:split_point].copy() # 불필요한 SettingWithCopyWarning 방지를 위해 copy() 추가
    eval_df = df[split_point:].copy() # 불필요한 SettingWithCopyWarning 방지를 위해 copy() 추가

    print(f"학습 데이터 크기: {len(train_df)}")
    print(f"검증 데이터 크기: {len(eval_df)}")

    # ===== 3. ✅ [수정] 정규화 통계 계산 및 저장 (훈련 데이터로만!) =====
    print("훈련 데이터(train_df)를 기준으로 정규화 통계를 계산합니다...")
    obs_means = train_df[OBS_COLS].mean()
    # 0으로 나누는 것을 방지하고, Series 형태로 유지
    obs_stds = train_df[OBS_COLS].std().replace(0, 1e-6) 
    
    # 계산된 통계 저장 
    stats = {'means': obs_means.to_dict(), 'stds': obs_stds.to_dict()} # 딕셔너리로 저장하여 로드 시 편리하게 함
    with open(stats_save_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"정규화 통계가 '{stats_save_path}'에 저장되었습니다.")

    # ===== 4. ✅ [수정] 환경 생성 (VecNormalize 제거 및 통계 주입) =====
    
    # --- 학습 환경 ---
    # DummyVecEnv만 사용하며, 계산된 통계를 TradingEnv에 직접 주입
    train_env_fn = lambda: TradingEnv(
        train_df, 
        obs_means=obs_means, 
        obs_stds=obs_stds, 
        window_size=window_size, 
        episode_length=episode_length
    )
    train_env = DummyVecEnv([train_env_fn]) 
    
    # --- 검증 환경 ---
    # ⚠️ [중요] 검증 환경에도 '훈련 데이터의 통계'를 동일하게 주입
    eval_env_fn = lambda: TradingEnv(
        eval_df, 
        obs_means=obs_means, 
        obs_stds=obs_stds, 
        window_size=window_size, 
        episode_length=len(eval_df) # 검증은 전체 기간을 한 에피소드로
    )
    eval_env = DummyVecEnv([eval_env_fn]) 
    
    print("환경 생성 완료. (VecNormalize 대신 환경 내부 정규화 및 통계 주입 사용)")

    # ===== 5. 콜백(Callback) 설정 =====
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=20, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path=log_dir,
        eval_freq=model_params['n_steps'] * 10,
        n_eval_episodes=1, 
        deterministic=True,
        render=False,
        callback_on_new_best=stop_train_callback 
    )

    # ===== 6. 모델 생성 및 학습 =====
    model = RecurrentPPO(env=train_env, **model_params)
    
    print("🚀 학습을 시작합니다...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback 
    )

    # ===== 7. 최종 모델 저장 =====
    print("✅ 학습이 완료되었습니다. 최종 모델을 저장합니다.")
    model.save(final_model_save_path)
    
    print(f"최종 모델: '{final_model_save_path}.zip'")
    print(f"훈련 통계: '{stats_save_path}'")


if __name__ == '__main__': 
    # 디렉토리 생성
    os.makedirs("./1_model", exist_ok=True)
    os.makedirs("./0_logs/recurrent_ppo2", exist_ok=True)
    os.makedirs("./0_data", exist_ok=True) 
    
    train_agent()