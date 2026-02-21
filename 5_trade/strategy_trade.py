# strategy_trade.py
import pandas as pd
import config # (추가매수 % 로드를 위해 config 임포트)

"""
매매 신호가 발생했을 때, 분할매수/매도, 익절/손절 등 
'어떻게 거래할 것인가'에 대한 구체적인 전략을 정의합니다.
(제공된 코드와 동일)
"""

# -----------------------------------------------------------------------------
# 1. 추가 매수 (Scaling-in) 전략
# -----------------------------------------------------------------------------
def should_add_buy(trade_state, current_price):
    """추가 매수(물타기) 조건을 확인합니다."""
    initial_price = trade_state['initial_entry_price']
    buy_count = trade_state['buy_count']

    # 2차 매수 조건
    # ❗️ [수정] config에서 비율을 가져오도록 수정 (하드코딩 제거)
    if buy_count == 1 and current_price <= initial_price * (1 - config.ADD_BUY_PCT_2ND / 100):
        print(f"✅ 추가 매수 조건 충족: 2차 매수 (현재가 {current_price:.2f})")
        return True, 2 # 2차 매수 신호
    
    # 3차 매수 조건
    if buy_count == 2 and current_price <= initial_price * (1 - config.ADD_BUY_PCT_3RD / 100):
        print(f"✅ 추가 매수 조건 충족: 3차 매수 (현재가 {current_price:.2f})")
        return True, 3 # 3차 매수 신호
        
    return False, None

# -----------------------------------------------------------------------------
# 2. 매도 (Exit) 전략 (원본 로직)
# -----------------------------------------------------------------------------
def check_exit_conditions(df, i, trade_state, stop_loss_pct, profit_target_pct, lock_start_pct, lock_trail_pct, atr_multiplier):
    """매도(익절/손절) 조건을 통합하여 확인합니다."""
    # ❗️ [수정] df.iloc[i]['ATR'] -> df.iloc[i].get('ATR', 0) 등으로 방어 코드 추가
    #           rl_data_processor에서 ATR 계산이 실패했을 경우를 대비
    current_price = df.iloc[i]['close']
    current_high = df.iloc[i]['high']
    current_atr = df.iloc[i].get('ATR', 0) # ❗️ .get()으로 안전하게 접근
    avg_price = trade_state['avg_price']
    
    # 1. 최우선 순위: 고정 손절
    if current_price < avg_price * (1 - stop_loss_pct / 100):
        print(f"✅ 손절 조건 충족: 손절매 (현재가 {current_price:.2f})")
        return True, current_price, f"Stop Loss (-{stop_loss_pct}%)"

    # 2. 익절/수익보호 로직 (매수 횟수 기준 분기)
    
    # 시나리오 1: 1차 매수 상태 (빠른 익절 및 수익 보호)
    if trade_state['buy_count'] == 1:
        initial_price = trade_state['initial_entry_price']
        
        # 1-1. 목표 익절
        if current_price >= initial_price * (1 + profit_target_pct / 100):
            print(f"✅ 익절 조건 충족: 목표 익절 (현재가 {current_price:.2f})")
            return True, current_price, f"Take Profit (+{profit_target_pct}%)"
        
        # 1-2. 수익 보호 (Trailing Stop)
        if not trade_state.get('profit_lock_activated') and current_price >= initial_price * (1 + lock_start_pct / 100):
            trade_state['profit_lock_activated'] = True
            start_stop_price = initial_price * (1 + (lock_start_pct - lock_trail_pct)/100)
            trade_state['trailing_stop_price'] = start_stop_price 
        
        if trade_state.get('profit_lock_activated'):
            new_stop_price = current_price * (1 - lock_trail_pct / 100)
            
            if new_stop_price > trade_state.get('trailing_stop_price', 0):
                trade_state['trailing_stop_price'] = new_stop_price
            
            if current_price < trade_state.get('trailing_stop_price', 0):
                print(f"✅ 수익 보호 조건 충족: Trailing Stop (현재가 {current_price:.2f})")
                return True, current_price, "Profit-locking Stop"

    # 시나리오 2 & 3: 2차 또는 3차 매수 완료 상태 (ATR 기반 수익 추적)
    else:
        # ❗️ ATR이 0이면 (계산 실패) 이 로직이 작동하지 않도록 방어
        if current_atr == 0:
            return False, None, None # ATR 없으면 Trailing Stop 불가

        highest_price = max(trade_state.get('highest_price_after_entry', 0), current_high)
        trade_state['highest_price_after_entry'] = highest_price
        
        new_stop_price = highest_price - (current_atr * atr_multiplier)
        old_stop_price = trade_state.get('atr_trailing_stop', 0)
        
        if new_stop_price > old_stop_price:
            trade_state['atr_trailing_stop'] = new_stop_price
        
        if old_stop_price > 0 and current_price < trade_state['atr_trailing_stop']:
            print(f"✅ ATR Trailing Stop 조건 충족: 손실최소화 (현재가 {current_price:.2f})")
            return True, current_price, "ATR Trailing Stop"
            
    return False, None, None