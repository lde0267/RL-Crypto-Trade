import os
import pandas as pd
import numpy as np
import pickle 
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from env2 import TradingEnv # TradingEnv 클래스가 있는 모듈을 임포트

# --- 경로 설정 (학습 코드와 동일) ---
data_path = "0_data/updated.csv"
stats_save_path = "./1_model/obs_stats_btc2.pkl"
# 가장 성능이 좋았던 모델을 로드하는 것이 일반적입니다.
model_path = "./1_model/final_model_rnn_ppo.zip" 
# "./1_model/best_model/best_model.zip"

# --- 학습 파라미터 (학습 코드와 동일) ---
train_test_split_ratio = 0.8
window_size = 10 
OBS_COLS = ['60_BB_Width', '30_VPT', '30_ADI', 
            'day_of_week', '30_OBV', '30_to_60_Close_ratio', 
            '30_BB_High', '30_ATR', '60_ADX']

def evaluate_agent():
    # 1. 데이터 로드 및 분리 (학습 시점과 동일하게)
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    df = df.dropna(subset=OBS_COLS + ['Close']) 
    df = df.reset_index(drop=True) 

    split_point = int(len(df) * train_test_split_ratio)
    eval_df = df[split_point:].copy() # 검증/테스트 데이터만 사용

    print(f"평가 데이터 크기: {len(eval_df)}")

    # 2. 정규화 통계 로드
    with open(stats_save_path, 'rb') as f:
        stats = pickle.load(f)
    
    obs_means = pd.Series(stats['means'])
    obs_stds = pd.Series(stats['stds'])
    print("정규화 통계 로드 완료.")

    # 3. 평가 환경 생성
    # ⚠️ [중요] 평가 기간 전체를 하나의 에피소드로 설정 (episode_length=len(eval_df))
    eval_env_fn = lambda: TradingEnv(
        eval_df, 
        obs_means=obs_means, 
        obs_stds=obs_stds, 
        window_size=window_size, 
        episode_length=len(eval_df)
    )
    eval_env = DummyVecEnv([eval_env_fn]) 
    print("평가 환경 생성 완료.")
    
    # 4. 모델 로드
    try:
        model = RecurrentPPO.load(model_path, env=eval_env)
        print(f"모델 로드 완료: {model_path}")
    except FileNotFoundError:
        print(f"⚠️ 모델 파일이 없습니다: {model_path}. 경로를 확인해주세요.")
        return
    # 5. 평가 실행 (이 부분이 누락되어 있었으므로 추가합니다)
    initial_close = eval_df['Close'].iloc[window_size] 
    final_close = eval_df['Close'].iloc[-1]
    
    print("벤치마크 데이터 준비 완료.")

    # 6. 평가 실행 (필수 인수인 initial_close와 final_close를 전달)
    evaluate(model, eval_env, initial_close, final_close)
    
def evaluate(model, env, initial_close, final_close, num_episodes=1):
    """
    에이전트를 평가하고 에피소드 결과를 반환하며 B&H 벤치마크를 출력
    """
    episode_rewards = []
    
    # Buy & Hold (B&H) 수익률 계산
    b_and_h_return = (final_close / initial_close) - 1

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        lstm_states = None 
        
        actions = []
        rewards = []
        
        # ⚠️ [추가] 포트폴리오 가치 히스토리를 저장할 리스트
        portfolio_values = []
        
        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
            
            # 4개 항목을 받습니다. (obs, reward, done, info)
            obs, reward, done_vec, info = env.step(action)
            
            done = done_vec[0] 
            total_reward += reward[0]
            
            # ⚠️ [추가] TradingEnv의 info에서 현재 포트폴리오 가치를 추출하여 저장
            if 'current_value' in info[0]:
                 portfolio_values.append(info[0]['current_value'])
                 
            actions.append(action[0])
            rewards.append(reward[0])
            
        episode_rewards.append(total_reward)

    # 6. 결과 분석
    print("\n--- 💰 평가 결과 분석 ---")
    print(f"평균 누적 보상 (Total Reward): {np.mean(episode_rewards):.2f}")
    print("-----------------------------------")
    
    if portfolio_values:
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        agent_return = (final_value / initial_value) - 1
        
        # 최대 낙폭 계산 (단순화된 형태, TradingEnv의 metrics가 있다면 그것을 사용하는 것이 정확합니다)
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdown = (cumulative_max - portfolio_values) / cumulative_max
        max_drawdown = np.max(drawdown)
        
        print(f"📈 에이전트 최종 수익률 (Agent Return): {agent_return:.2%}")
        print(f"📉 에이전트 최대 낙폭 (Max Drawdown): {max_drawdown:.2%}")
        print(f"💰 최종 포트폴리오 가치 (Final Value): ${final_value:,.2f}")
    
    # 벤치마크 결과 출력
    print("\n--- ⚖️ 벤치마크 비교 (Buy & Hold) ---")
    print(f"📊 B&H 수익률: {b_and_h_return:.2%}")
    print("-----------------------------------")
    
    return episode_rewards
# 5. 평가 실행
if __name__ == '__main__': 
    evaluate_agent()