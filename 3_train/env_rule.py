import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    [수정됨]
    - 정규화 통계(means/stds)를 내부에서 계산하지 않고,
    - 외부에서 파라미터로 주입받아 데이터 유출(Lookahead Bias)을 방지합니다.
    """
    metadata = {"render.modes": ["human"]}

    # ✅ [수정] __init__ 시그니처 변경: obs_means와 obs_stds를 파라미터로 받음
    def __init__(self, df: pd.DataFrame, 
                 obs_means: pd.Series, 
                 obs_stds: pd.Series,
                 window_size: int = 10, 
                 episode_length: int = 96,
                 trailing_stop_pct: float = 0.01,
                 trade_ratio: float = 0.5):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        
        self.window_size = window_size
        self.episode_length = episode_length

        # --- Trading Parameters ---
        self.initial_balance = 300_000.0
        self.min_trade_krw = 5000.0
        self.fee = 0.0005
        self.trade_ratio = trade_ratio
        self.trailing_stop_pct = trailing_stop_pct
        
        # --- Rewards and Penalties ---
        self.reward_scaling = 100.0
        self.profit_bonus = 5.0
        self.loss_penalty = -5.0
        self.shaping_scaling = 1.0 # 너무 크다!

        # --- Observation and Action Spaces ---
        
        # Lasso (C=0.01)로 선택된 6개의 핵심 지표
        self.obs_cols = [
            '30_to_60_Close_ratio', 
            '60_OBV', 
            'day_of_week', 
            '30_ATR', 
            '30_Keltner_lband', 
            '60_ADX'
        ]  
        
        self.portfolio_info_len = 5 
        num_features = len(self.obs_cols) + self.portfolio_info_len
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, num_features), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        # ✅ [수정] 정규화 통계를 외부에서 주입받음
        # (데이터 유출 방지)
        self.obs_means = obs_means
        self.obs_stds = obs_stds
        
        self.highest_price_since_buy = 0.0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 60분봉 지표(60_OBV, 60_ADX) 등을 위한 안전 마진 (120이면 충분)
        safe_start_margin = self.window_size + 120 
        
        # ✅ [수정] max_start 계산 시 window_size만 빼도 되지만, 
        #           지표 계산 마진을 고려해 safe_start_margin을 사용
        max_start = len(self.df) - self.episode_length - safe_start_margin
        
        self.start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        self.current_step = self.start_idx + safe_start_margin
        
        self.step_idx = 0
        self.balance = float(self.initial_balance)
        self.coin_holdings = 0.0
        self.avg_buy_price = 0.0
        self.highest_price_since_buy = 0.0

        self.previous_price = self.df.loc[self.current_step, 'Close']
        
        return self._get_obs(), {}

    def _get_obs(self):
        start = self.current_step - self.window_size + 1
        end = self.current_step + 1
        
        # ⚠️ window_df가 비어있는 극단적인 경우 방지
        if start < 0:
            start = 0
            # obs_array가 (window_size, num_features)가 아닐 수 있으므로 경고
            # 하지만 reset() 로직상 이 경우는 거의 발생하지 않음
            
        window_df = self.df.iloc[start:end]
        
        # ✅ [수정] 정규화 시 (미래 데이터가 아닌) 주입받은 통계(self.obs_means) 사용
        norm_obs_window = (window_df[self.obs_cols] - self.obs_means) / self.obs_stds
        
        current_price = self.df.loc[self.current_step, 'Close']
        is_holding = 1.0 if self.coin_holdings > 0 else 0.0
        unrealized_pnl = (current_price - self.avg_buy_price) / (self.avg_buy_price + 1e-9) if is_holding else 0.0
        
        portfolio_info = np.array([
            (self.balance - self.initial_balance) / (self.initial_balance * 0.5),
            (self.coin_holdings * current_price) / self.initial_balance,
            is_holding, 
            unrealized_pnl, 
            self.step_idx / max(1, self.episode_length)
        ])
        
        portfolio_info_tiled = np.tile(portfolio_info, (self.window_size, 1))
        
        # ✅ [수정] window_df가 window_size보다 작을 경우 패딩 처리 (에피소드 극초반)
        current_window_len = len(norm_obs_window)
        if current_window_len < self.window_size:
            padding = np.zeros((self.window_size - current_window_len, len(self.obs_cols)))
            norm_obs_window_values = np.concatenate([padding, norm_obs_window.values], axis=0)
        else:
            norm_obs_window_values = norm_obs_window.values

        obs_array = np.concatenate([norm_obs_window_values, portfolio_info_tiled], axis=1)
        return obs_array.astype(np.float32)

    # ... ( _calculate_reward, step, _buy, _sell, render 메서드는 동일하므로 생략 )...
    
    def _calculate_reward(self, realized_pnl, is_stop_loss, is_take_profit):
        reward = 0.0
        if realized_pnl != 0:
            reward += realized_pnl * self.reward_scaling
            if is_take_profit:
                reward += self.profit_bonus
            elif is_stop_loss:
                reward += self.loss_penalty
        return float(np.clip(reward, -20.0, 20.0))

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        is_holding = self.coin_holdings > 0
        is_stop_loss, is_take_profit, traded = False, False, False
        realized_pnl = 0.0
        
        # ✅ --- [추가] 보상 쉐이핑 (Dense Reward) ---
        shaping_reward = 0.0

        # 1. 가격 변동률 계산 (1스텝 전 대비)
        # (0으로 나누기 방지)
        price_change_pct = (current_price - self.previous_price) / (self.previous_price + 1e-9)

        if is_holding:
            # 1. (홀딩 중) 가격이 오르면 보상, 내리면 페널티
            # (미실현 손익의 '변화량'을 보상으로 줌)
            shaping_reward = price_change_pct
        else:
            # 2. (미보유 중) 가격이 오르면 페널티(기회비용), 내리면 보상(손실 회피)
            # (말씀하신 "가지고 있지 않은 동안 오르면 페널티")
            shaping_reward = -price_change_pct

        # 쉐이핑 보상에 가중치 적용
        shaping_reward *= self.shaping_scaling
        # -----------------------------------------------

        # 1. 자동 매도 로직 (트레일링 스탑)
        if is_holding:
            self.highest_price_since_buy = max(self.highest_price_since_buy, current_price)
            trailing_stop_price = self.highest_price_since_buy * (1 - self.trailing_stop_pct)
            
            if current_price <= trailing_stop_price:
                _, r_pnl = self._sell(self.coin_holdings, current_price)
                realized_pnl += r_pnl
                traded = True
                
                if realized_pnl > 0:
                    is_take_profit = True
                else:
                    is_stop_loss = True

        # 2. 에이전트의 매수 로직
        elif not is_holding and action == 1:
            cost_to_spend = self.balance * self.trade_ratio
            buy_qty = cost_to_spend / current_price if current_price > 0 else 0
            cost = self._buy(buy_qty, current_price)
            if cost > 0:
                traded = True

        # --- Calculate reward and move to the next step ---
        # ✅ [수정] 최종 스텝 보상 = (매도 보상) + (쉐이핑 보상)
        realized_reward = self._calculate_reward(realized_pnl, is_stop_loss, is_take_profit)
        step_reward = realized_reward + shaping_reward # 👈 두 보상을 합산

        # ✅ [추가] 다음 스텝을 위해 현재 가격을 '이전 가격'으로 저장
        self.previous_price = current_price
        
        self.current_step += 1
        self.step_idx += 1
        
        # ✅ [수정] 종료 조건 강화: 데이터 끝에 도달하기 최소 1스텝 전에 종료
        terminated = self.current_step >= (len(self.df) - 1)
        truncated = self.step_idx >= self.episode_length
        
        # 종료/중단 시 obs를 가져오지 않고 빈 dict와 함께 리셋 obs 반환 준비
        if terminated or truncated:
            obs = self._get_obs() # 마지막 obs를 가져오긴 하지만...
            info = {'asset': self.balance + self.coin_holdings * current_price}
            # ... 다음 reset()에서 새 obs가 나갈 것임
        else:
            obs = self._get_obs()
            info = {'asset': self.balance + self.coin_holdings * current_price}
            
        return obs, step_reward, terminated, truncated, info

    def _buy(self, qty, price):
        if qty <= 0 or price <= 0: return 0.0
        cost = qty * price * (1 + self.fee)
        if cost < self.min_trade_krw or cost > self.balance: return 0.0
        
        self.avg_buy_price = price
        self.coin_holdings = qty
        self.balance -= cost
        self.highest_price_since_buy = price
        return cost

    def _sell(self, qty, price):
        if qty <= 0 or price <= 0 or self.coin_holdings <= 0: return 0.0, 0.0
        
        revenue = qty * price * (1 - self.fee)
        realized_pnl = (price - self.avg_buy_price) / (self.avg_buy_price + 1e-9)
        
        self.coin_holdings = 0.0
        self.avg_buy_price = 0.0
        self.balance += revenue
        self.highest_price_since_buy = 0.0
        return revenue, realized_pnl

    def render(self, mode="human"):
        current_price = self.df.loc[self.current_step, 'Close']
        total_asset = self.balance + self.coin_holdings * current_price
        
        is_holding = self.coin_holdings > 0
        trailing_stop_price = self.highest_price_since_buy * (1 - self.trailing_stop_pct) if is_holding else 0
        
        print(f"[Step {self.step_idx:03d}] Asset: {total_asset:,.0f} | "
              f"Holdings: {self.coin_holdings:.4f} | "
              f"Balance: {self.balance:,.0f} | "
              f"Current Stop: {trailing_stop_price:,.0f}")