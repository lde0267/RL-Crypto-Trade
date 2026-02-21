import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    [수정됨]
    - 정규화 통계(means/stds)를 내부에서 계산하지 않고,
      외부에서 파라미터로 주입받아 데이터 유출(Lookahead Bias)을 방지합니다.
    - [백테스트용 수정] step()의 info 딕셔너리에
      'portfolio_value', 'date', 'traded'를 추가합니다.
    """
    metadata = {"render.modes": ["human"]}

    # ✅ [수정] __init__ 시그니처 변경: obs_means와 obs_stds를 파라미터로 받음
    def __init__(self, df: pd.DataFrame, 
                 obs_means: pd.Series, 
                 obs_stds: pd.Series,
                 window_size: int = 10, 
                 episode_length: int = 96,
                 trailing_stop_pct: float = 0.001,
                 trade_ratio: float = 0.5):
        
        super().__init__()
        
        # ❗️ 중요: 'datetime' 컬럼이 df에 포함되어 있어야 합니다.
        #    백테스트 리포팅에 이 날짜/시간 정보를 사용합니다.
        if 'datetime' not in df.columns:
            raise ValueError("DataFrame 'df' must contain a 'datetime' column for backtesting.")
            
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
        self.shaping_scaling = 50.0

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
        self.obs_means = obs_means
        self.obs_stds = obs_stds
        
        self.highest_price_since_buy = 0.0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 60분봉 지표(60_OBV, 60_ADX) 등을 위한 안전 마진 (120이면 충분)
        self.safe_start_margin = self.window_size + 120 
        
        max_start = len(self.df) - self.episode_length - self.safe_start_margin
        
        # 백테스트 모드(episode_length가 전체 길이일 경우) max_start가 0이 되어
        # 항상 0에서 시작하게 됩니다.
        self.start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        self.current_step = self.start_idx + self.safe_start_margin
        
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
        
        if start < 0:
            start = 0
            
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
        
        current_window_len = len(norm_obs_window)
        if current_window_len < self.window_size:
            padding = np.zeros((self.window_size - current_window_len, len(self.obs_cols)))
            norm_obs_window_values = np.concatenate([padding, norm_obs_window.values], axis=0)
        else:
            norm_obs_window_values = norm_obs_window.values

        obs_array = np.concatenate([norm_obs_window_values, portfolio_info_tiled], axis=1)
        return obs_array.astype(np.float32)

    
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
        price_change_pct = (current_price - self.previous_price) / (self.previous_price + 1e-9)

        if is_holding:
            shaping_reward = price_change_pct
        else:
            shaping_reward = -price_change_pct
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
        realized_reward = self._calculate_reward(realized_pnl, is_stop_loss, is_take_profit)
        step_reward = realized_reward + shaping_reward # 👈 두 보상을 합산

        self.previous_price = current_price
        
        self.current_step += 1
        self.step_idx += 1
        
        terminated = self.current_step >= (len(self.df) - 1)
        truncated = self.step_idx >= self.episode_length
        
        # ⭐️⭐️⭐️ [백테스트용 핵심 수정] ⭐️⭐️⭐️
        # quantstats가 요구하는 '포트폴리오 가치'와 '날짜'를 info에 담아 반환합니다.
        # 'datetime' 컬럼이 self.df에 존재해야 합니다.
        
        current_total_asset = self.balance + self.coin_holdings * current_price
        
        info = {
            # 1. 포트폴리오 가치 (이전 스크립트 호환을 위해 'portfolio_value' 사용)
            'portfolio_value': current_total_asset,
            
            # 2. 현재 날짜/시간 (df에서 'datetime' 컬럼 조회)
            'date': self.df.loc[self.current_step - 1, 'datetime'], # (step이 증가했으므로 -1)
            
            # 3. 거래 발생 여부 (총 거래 횟수 카운트용)
            'traded': traded,
            
            # (기타 정보)
            'realized_pnl_pct': realized_pnl,
            'is_holding': is_holding
        }
        
        # ⭐️⭐️⭐️ [수정 끝] ⭐️⭐️⭐️

        if terminated or truncated:
            obs = self._get_obs() # 마지막 obs (사용되지는 않음)
        else:
            obs = self._get_obs()
            
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
        # render는 백테스트 시 호출하지 않으므로 생략 가능
        pass