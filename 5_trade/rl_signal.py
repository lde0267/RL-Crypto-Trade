# rl_signal.py
import numpy as np
from stable_baselines3 import PPO
import config

# --- 전역 변수로 모델 로드 (봇 시작 시 1회만) ---
RL_MODEL = None

def load_rl_model():
    """봇 시작 시 RL 모델을 메모리에 로드합니다."""
    global RL_MODEL
    try:
        RL_MODEL = PPO.load(config.RL_MODEL_PATH)
        print(f"✅ RL 모델 '{config.RL_MODEL_PATH}' 로드 성공.")
    except Exception as e:
        print(f"🚨 치명적 오류: RL 모델 로드 실패: {e}")
        RL_MODEL = None

def generate_rl_buy_signal(observation):
    """
    정규화된 관측값(observation)을 받아 RL 모델의 매수 신호를 반환합니다.
    """
    if RL_MODEL is None:
        print("🚨 RL 모델이 로드되지 않아 매수 신호를 확인할 수 없습니다.")
        return False
        
    if observation is None:
        print("🚨 관측값이(Observation)가 None입니다. 매수 신호를 확인할 수 없습니다.")
        return False

    try:
        # ❗️ [중요] 훈련 시 정의한 Action Space에 따라 수정해야 합니다.
        # 예: 0 = HOLD, 1 = BUY, (2 = SELL)
        # 여기서는 1번 행동(action)이 'BUY'라고 가정합니다.
        BUY_ACTION_VALUE = 1 
        
        action, _states = RL_MODEL.predict(observation, deterministic=True)
        
        if action == BUY_ACTION_VALUE:
            # RL 모델이 매수 신호를 결정했습니다.
            return True
        else:
            # RL 모델이 홀드(또는 매도) 신호를 결정했습니다.
            return False
            
    except Exception as e:
        print(f"🚨 RL 모델 예측(predict) 중 오류 발생: {e}")
        return False