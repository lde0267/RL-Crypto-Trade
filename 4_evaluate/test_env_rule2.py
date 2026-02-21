import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any

class TradingEnv(gym.Env):
    """
    [룰 기반 매도 전략 반영]
    - 매수 결정: RL 모델 (Action 1)
    - 매도 결정: 룰 기반 (Stop Loss, Take Profit, Trailing Stop)
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, 
                 obs_means: pd.Series, 
                 obs_stds: pd.Series,
                 window_size: int = 10, 
                 episode_length: int = 96,
                 trailing_stop_pct: float = 0.001,
                 trade_ratio: float = 0.5,
                 # ⭐️ [추가] 새로운 매도 룰 파라미터
                 stop_loss_pct: float = 0.02, 
                 take_profit_pct: float = 0.03):
        
        super().__init__()
        
        if 'datetime' not in df.columns:
            # df.reset_index(drop=True)를 사용하므로, datetime 컬럼이 있어야 합니다.
            raise ValueError("DataFrame 'df' must contain a 'datetime' column for backtesting.")
            
        self.df = df.reset_index(drop=True).copy()
        
        self.window_size = window_size
        self.episode_length = episode_length

        # --- Trading Parameters ---
        self.initial_balance = 300_000.0
        self.min_trade_krw = 5000.0
        self.fee = 0.0005
        self.trade_ratio = trade_ratio
        
        # ⭐️ [수정] 매도 룰 파라미터 저장
        self.trailing_stop_pct = trailing_stop_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # --- Rewards and Penalties ---
        self.reward_scaling = 100.0
        self.profit_bonus = 5.0
        self.loss_penalty = -5.0
        self.shaping_scaling = 50.0

        # --- Observation and Action Spaces ---
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
        self.action_space = spaces.Discrete(2) # 0:유지/미보유, 1:매수

        self.obs_means = obs_means
        self.obs_stds = obs_stds
        
        self.highest_price_since_buy = 0.0
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        self.safe_start_margin = self.window_size + 120 
        
        max_start = len(self.df) - self.episode_length - self.safe_start_margin
        
        self.start_idx = self.np_random.integers(0, max_start + 1) if max_start > 0 else 0
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

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        current_price = self.df.loc[self.current_step, 'Close']
        is_holding = self.coin_holdings > 0
        is_stop_loss, is_take_profit, traded = False, False, False
        realized_pnl = 0.0
        
        # --- 보상 쉐이핑 ---
        shaping_reward = 0.0
        price_change_pct = (current_price - self.previous_price) / (self.previous_price + 1e-9)

        if is_holding:
            shaping_reward = price_change_pct
        else:
            shaping_reward = -price_change_pct
        shaping_reward *= self.shaping_scaling
        # -------------------

        # 1. 자동 매도 로직 (룰 기반)
        sell_trigger = None
        
        if is_holding:
            self.highest_price_since_buy = max(self.highest_price_since_buy, current_price)

            # ⭐️ 1-1. 손절매 (Stop Loss) 검사: 매수 가격 대비 하락률
            stop_loss_price = self.avg_buy_price * (1.0 - self.stop_loss_pct)
            
            # ⭐️ 1-2. 익절 목표 (Take Profit) 검사: 매수 가격 대비 상승률
            take_profit_price = self.avg_buy_price * (1.0 + self.take_profit_pct)
            
            # ⭐️ 1-3. 트레일링 스톱 (Trailing Stop) 검사: 최고가 대비 하락률
            trailing_stop_price = self.highest_price_since_buy * (1.0 - self.trailing_stop_pct)

            # ✅ [매도 룰 우선순위 적용] SL > TP > TS
            if current_price <= stop_loss_price:
                sell_trigger = 'SL' # 손절매 (최우선)
            elif current_price >= take_profit_price:
                sell_trigger = 'TP' # 익절 목표
            elif current_price <= trailing_stop_price:
                sell_trigger = 'TS' # 트레일링 스톱

            if sell_trigger:
                # 매도 실행
                _, r_pnl = self._sell(self.coin_holdings, current_price)
                realized_pnl += r_pnl
                traded = True
                
                # 보상 플래그 설정
                if r_pnl > 0:
                    is_take_profit = True
                else: 
                    is_stop_loss = True # 손절매 또는 익절/트레일링 후 손실 확정

        # 2. 에이전트의 매수 로직
        elif not is_holding and action == 1:
            cost_to_spend = self.balance * self.trade_ratio
            buy_qty = cost_to_spend / current_price if current_price > 0 else 0
            cost = self._buy(buy_qty, current_price)
            if cost > 0:
                traded = True

        # --- Calculate reward and move to the next step ---
        realized_reward = self._calculate_reward(realized_pnl, is_stop_loss, is_take_profit)
        step_reward = realized_reward + shaping_reward 

        self.previous_price = current_price
        
        self.current_step += 1
        self.step_idx += 1
        
        terminated = self.current_step >= (len(self.df) - 1)
        truncated = self.step_idx >= self.episode_length
        
        # --- Info 딕셔너리 구성 ---
        current_total_asset = self.balance + self.coin_holdings * current_price
        
        info = {
            'portfolio_value': current_total_asset,
            # 현재 스텝의 날짜/시간 (current_step이 이미 1 증가했으므로 -1)
            'date': self.df.loc[self.current_step - 1, 'datetime'], 
            'traded': traded,
            'realized_pnl_pct': realized_pnl,
            'is_holding': self.coin_holdings > 0,
            # ⭐️ [추가] 어떤 룰로 매도되었는지 확인용 (백테스트 로깅에 유용)
            'sell_rule': sell_trigger if traded and sell_trigger else None
        }
        
        if terminated or truncated:
             # 에피소드 종료 시 마지막 청산
            if self.coin_holdings > 0:
                _, end_pnl = self._sell(self.coin_holdings, current_price)
                step_reward += self._calculate_reward(end_pnl, end_pnl < 0, end_pnl > 0)
                info['portfolio_value'] = self.balance
                info['is_holding'] = False
                info['traded'] = True
                info['sell_rule'] = 'END'
            
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
        pass