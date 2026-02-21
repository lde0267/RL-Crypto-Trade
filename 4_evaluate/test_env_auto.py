import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
from typing import Optional, Tuple, Any

class TradingEnv(gym.Env):
    """
    [훈련용 환경 - env2.py 기반 수정됨]
    - SB3 RecurrentPPO 학습에 사용하기 위해 설계
    - 정규화 통계(means/stds)를 외부에서 주입받음.
    - 행동 공간: 0:유지/미보유, 1:매수, 2:매도
    - ✅ 24시간(48 스텝) 강제 청산 로직 및 보유 기간 관측치 추가
    """
    metadata = {"render_modes": ["human"]} 
    
    # ⭐️ [추가] 최대 보유 기간 상수
    MAX_HOLD_STEPS = 48 # 24시간 = 48 스텝 (30분봉 기준)

    def __init__(self, df: pd.DataFrame, 
                 obs_means: pd.Series, 
                 obs_stds: pd.Series,
                 window_size: int = 10, 
                 episode_length: int = 96,
                 trade_ratio: float = 0.5):
        
        super().__init__()
        self.df = df.copy() 
        self.window_size = window_size
        self.episode_length = episode_length
        self.random_start_margin = 120 

        # --- Trading Parameters ---
        self.initial_balance = 300_000.0
        self.min_trade_krw = 5000.0
        self.fee = 0.0005
        self.trade_ratio = trade_ratio
        
        # --- Rewards and Penalties ---
        self.reward_scaling = 100.0
        self.profit_bonus = 5.0
        self.loss_penalty = -5.0
        self.shaping_scaling = 50.0

        # --- Observation and Action Spaces ---
        self.obs_cols = ['60_BB_Width', '30_VPT', '30_ADI', 'day_of_week', '30_OBV', 
                         '30_to_60_Close_ratio', '30_BB_High', '30_ATR', '60_ADX']
        
        self.portfolio_info_len = 5 
        num_features = len(self.obs_cols) + self.portfolio_info_len
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, num_features), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3) 

        self.obs_means = obs_means
        self.obs_stds = obs_stds
        
        self.df_index = self.df.index 
        
        self.max_episode_start_idx = len(self.df) - self.episode_length - self.window_size - self.random_start_margin

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        if self.max_episode_start_idx < 0:
             self.start_idx = 0 
        else:
             self.start_idx = self.np_random.integers(0, self.max_episode_start_idx + 1)
        
        self.current_step_df_idx = self.start_idx + self.window_size + self.random_start_margin
        
        self.step_idx = 0
        self.balance = float(self.initial_balance)
        self.coin_holdings = 0.0
        self.avg_buy_price = 0.0
        self.steps_since_buy = 0 # ⭐️ [추가] 보유 기간 카운터 초기화

        self.previous_price = self.df.iloc[self.current_step_df_idx]['Close']
        
        info = self._get_info()
        return self._get_obs(), info

    def _get_obs(self) -> np.ndarray:
        end_iloc = self.current_step_df_idx + 1
        start_iloc = end_iloc - self.window_size
        
        if start_iloc < 0:
             start_iloc = 0
             
        window_df = self.df.iloc[start_iloc:end_iloc]
        
        norm_obs_window = (window_df[self.obs_cols] - self.obs_means) / self.obs_stds
        
        current_price = self.df.iloc[self.current_step_df_idx]['Close']
        is_holding = 1.0 if self.coin_holdings > 0 else 0.0
        unrealized_pnl = (current_price - self.avg_buy_price) / (self.avg_buy_price + 1e-9) if is_holding else 0.0
        
        # ⭐️ [수정] 에피소드 진행률 대신 '보유 시간 진행률'을 관측치에 추가합니다.
        holding_time_ratio = (self.steps_since_buy / self.MAX_HOLD_STEPS) if is_holding else 0.0

        portfolio_info = np.array([
            (self.balance - self.initial_balance) / (self.initial_balance * 0.5), 
            (self.coin_holdings * current_price) / self.initial_balance,           
            is_holding, 
            unrealized_pnl, 
            holding_time_ratio # ✅ 보유 시간 비율
        ])
        
        portfolio_info_tiled = np.tile(portfolio_info, (self.window_size, 1))
        
        current_window_len = len(norm_obs_window)
        if current_window_len < self.window_size:
            padding = np.zeros((self.window_size - current_window_len, len(self.obs_cols)))
            norm_obs_window_values = np.concatenate([padding, norm_obs_window.values], axis=0)
        else:
            norm_obs_window_values = norm_obs_window.values

        obs_array = np.concatenate([norm_obs_window_values, portfolio_info_tiled], axis=1)
        return obs_array.astype(np.float32)

    def _get_info(self) -> dict:
        current_price = self.df.iloc[self.current_step_df_idx]['Close']
        total_asset = self.balance + self.coin_holdings * current_price
        
        info = {
            'date': self.df.index[self.current_step_df_idx],
            'current_value': total_asset, 
            'balance': self.balance,
            'holdings': self.coin_holdings,
            'avg_buy_price': self.avg_buy_price,
            'is_holding': self.coin_holdings > 0,
        }
        return info

    def _calculate_reward(self, realized_pnl: float, is_stop_loss: bool, is_take_profit: bool) -> float:
        reward = 0.0
        if realized_pnl != 0:
            reward += realized_pnl * self.reward_scaling
            if is_take_profit:
                reward += self.profit_bonus
            elif is_stop_loss:
                reward += self.loss_penalty
        return float(np.clip(reward, -20.0, 20.0))

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        current_price = self.df.iloc[self.current_step_df_idx]['Close']
        is_holding = self.coin_holdings > 0
        is_stop_loss, is_take_profit, traded = False, False, False
        realized_pnl = 0.0 
        
        shaping_reward = 0.0
        
        # ⭐️⭐️⭐️ [1. 보유 기간(24시간) 강제 청산 로직] ⭐️⭐️⭐️
        if is_holding:
            self.steps_since_buy += 1 # 보유 기간 1 스텝 증가
            
            if self.steps_since_buy >= self.MAX_HOLD_STEPS:
                # 24시간이 지나면 강제 매도 실행
                _, r_pnl = self._sell(self.coin_holdings, current_price)
                realized_pnl = r_pnl 
                traded = True
                
                # 강제 청산 결과 플래그 설정
                if realized_pnl > 0:
                    is_take_profit = True
                else:
                    is_stop_loss = True
                
                # 강제 청산이 발생했으므로, 에이전트의 현재 action을 무시 (0:유지)
                action = 0 
                shaping_reward -= 10.0 # 강제 청산에 대한 페널티
        # ⭐️⭐️⭐️ [로직 추가 끝] ⭐️⭐️⭐️

        # --- 2. 보상 쉐이핑 (Dense Reward) ---
        price_change_pct = (current_price - self.previous_price) / (self.previous_price + 1e-9)

        if is_holding:
            shaping_reward += price_change_pct 
        else:
            shaping_reward += -price_change_pct 
        
        if action == 1 and is_holding: shaping_reward += -0.1 
        elif action == 2 and not is_holding: shaping_reward += -0.1

        shaping_reward *= self.shaping_scaling
        # -----------------------------------------------

        # --- 3. 에이전트 행동 로직 (강제 청산이 아닐 경우) ---

        # 1. 매수 로직 (Action == 1)
        if action == 1 and not is_holding:
            cost_to_spend = self.balance * self.trade_ratio
            buy_qty = cost_to_spend / current_price if current_price > 0 else 0
            cost = self._buy(buy_qty, current_price)
            if cost > 0: traded = True

        # 2. 매도 로직 (Action == 2)
        elif action == 2 and is_holding:
            _, r_pnl = self._sell(self.coin_holdings, current_price)
            realized_pnl += r_pnl
            traded = True
            
            if realized_pnl > 0:
                is_take_profit = True
                shaping_reward += 5.0 
            else:
                is_stop_loss = True
                shaping_reward += -2.0 

        # --- 4. Calculate reward and move to the next step ---
        realized_reward = self._calculate_reward(realized_pnl, is_stop_loss, is_take_profit)
        step_reward = realized_reward + shaping_reward 

        self.previous_price = current_price
        
        self.current_step_df_idx += 1
        self.step_idx += 1
        
        terminated = self.current_step_df_idx >= (len(self.df) - 1)
        truncated = self.step_idx >= self.episode_length
        
        # --- 정보 및 관측치 반환 ---
        info = self._get_info()
        info['traded'] = traded 

        if terminated or truncated:
            # 에피소드 종료 시 강제 청산 (종료 보상)
            if self.coin_holdings > 0:
                _, end_pnl = self._sell(self.coin_holdings, current_price)
                step_reward += self._calculate_reward(end_pnl, end_pnl < 0, end_pnl > 0)
                
            info = self._get_info() 
        
        obs = self._get_obs() 
        
        return obs, step_reward, terminated, truncated, info

    def _buy(self, qty, price):
        if qty <= 0 or price <= 0: return 0.0
        cost = qty * price * (1 + self.fee)
        if cost < self.min_trade_krw or cost > self.balance: return 0.0
        
        self.avg_buy_price = price
        self.coin_holdings = qty
        self.balance -= cost
        self.steps_since_buy = 0 # ⭐️ [추가] 매수 시 카운터 초기화
        return cost

    def _sell(self, qty, price):
        if qty <= 0 or price <= 0 or self.coin_holdings <= 0: return 0.0, 0.0
        
        revenue = qty * price * (1 - self.fee)
        realized_pnl = (price - self.avg_buy_price) / (self.avg_buy_price + 1e-9)
        
        self.coin_holdings = 0.0
        self.avg_buy_price = 0.0
        self.balance += revenue
        self.steps_since_buy = 0 # ⭐️ [추가] 매도 시 카운터 초기화
        return revenue, realized_pnl

    def render(self, mode="human"):
        current_price = self.df.iloc[self.current_step_df_idx]['Close']
        total_asset = self.balance + self.coin_holdings * current_price
        is_holding = self.coin_holdings > 0
        
        print(f"📈 [Step {self.step_idx:03d}] Asset: {total_asset:,.0f} | "
              f"Holdings: {self.coin_holdings:.4f} | "
              f"Balance: {self.balance:,.0f} | "
              f"Holding: {'O' if is_holding else 'X'}")

    def close(self):
        pass