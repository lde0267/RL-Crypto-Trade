import pandas as pd
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# --- RL 모듈 임포트 ---
import rl_data_processor  # 1. 데이터 처리기
import rl_signal          # 2. RL 신호 생성기
import strategy_trade     # 3. 매매 관리 전략 (기존)

import config
import notifier
import db_handler
import upbit_client as uc

# ⭐️ 매도 후 쿨다운 시간 설정 (30분봉 1틱 분량)
COOLDOWN_DURATION = timedelta(minutes=30) 

client = None
KST = ZoneInfo('Asia/Seoul')
trade_states = {}
target_tickers = ["KRW-BTC"]

def log(message):
    """로그 메시지를 시간과 함께 출력하고, notifier를 통해 텔레그램으로 전송"""
    log_message = f"[{datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(log_message)
    notifier.send_message(log_message)

def _create_initial_state():
    """한 종목에 대한 초기 거래 상태 딕셔너리를 생성하여 반환합니다."""
    return {
        'status': 'no_position', 
        'entry_time': None, 
        'initial_entry_price': 0,
        'total_investment': 0, 
        'total_amount': 0, 
        'avg_price': 0,
        'buy_count': 0, 
        'highest_price_after_entry': 0, 
        'buy_points': [],
        # ⭐️ 쿨다운 상태 변수 추가
        'last_sell_timestamp': None 
    }

def _run_logic_for_ticker(ticker):
    """[단일 종목 거래 실행] 한 개의 종목에 대해 매수/매도/추가매수 로직을 실행합니다."""
    global trade_states

    client.cancel_open_orders(ticker)
    
    if ticker not in trade_states:
        log(f"[{ticker}] 추적 대상이 아닙니다. 건너뜕니다.")
        return
        
    current_trade_state = trade_states[ticker]

    # --- 1. 데이터 로드 및 RL 관측값 생성 ---
    rl_observation, df = rl_data_processor.get_processed_data(
        ticker,
        current_trade_state
        )
    
    if df is None or rl_observation is None:
        log(f"[{ticker}] 데이터 처리 중 오류가 발생하여 건너뜁니다.")
        return
    
    # 현재가 등 최신 정보
    latest_row = df.iloc[-1]
    current_price = latest_row['close']
    # ⭐️ 현재 캔들 시간 (datetime 객체)
    current_time = df.index[-1] 
    
    # 2. 포지션 상태에 따라 로직 분기
    current_positions = sum(1 for state in trade_states.values() if state['status'] != 'no_position')

    # 2-1. 포지션이 없는 경우: RL 모델로 매수 신호 확인
    if current_trade_state['status'] == 'no_position' and current_positions < config.MAX_POSITIONS:
        
        # ⭐️ 매도 쿨다운 체크 로직
        last_sell = current_trade_state.get('last_sell_timestamp')
        
        if last_sell is not None:
            time_since_sell = current_time - last_sell
            
            if time_since_sell < COOLDOWN_DURATION:
                remaining = COOLDOWN_DURATION - time_since_sell
                log(f"[{ticker}] 🚫 매수 금지: 쿨다운 중. 남은 시간: {remaining}")
                return 

        log(f"[{ticker}] RL 매수 신호 확인 중... (현재 보유: {current_positions}개 / 최대: {config.MAX_POSITIONS}개)")
        
        is_buy = rl_signal.generate_rl_buy_signal(rl_observation)
        
        if is_buy:
            log(">"*20 + f" [{ticker}] RL 모델 매수 신호 발생! " + "<"*20)
            
            current_balance = client.get_balance("KRW")
            if current_balance < 5000:
                log(f"[{ticker}] KRW 잔고 부족으로 매수 불가 (잔고: {current_balance:,.0f}원)")
                return
            
            first_investment = current_balance * config.TRADE_RATIO_1ST 
            log(f"[{ticker}] [매수 실행] 주문액: {first_investment:,.0f}원")
            
            buy_result = client.buy_market_order(ticker, first_investment)
            
            if buy_result and 'uuid' in buy_result:
                time.sleep(1) # 체결 대기
                coin_symbol = ticker.split('-')[1]
                filled_amount = client.get_balance(coin_symbol)
                
                if filled_amount == 0:
                    log(f"🚨 [{ticker}] 매수 주문은 성공했으나, 체결 수량이 0입니다.")
                    return

                log(f"✅ [{ticker}] 매수 주문 성공! 체결 수량: {filled_amount:.6f}")
                avg_price = first_investment / filled_amount
                db_handler.log_trade(ticker, 'BUY', 'INITIAL_RL', current_price, filled_amount)

                # 매수 성공 시 쿨다운 상태 초기화
                trade_states[ticker] = {
                    'status': 'buying', 'entry_time': current_time, 
                    'initial_entry_price': current_price,
                    'total_investment': first_investment, 
                    'total_amount': filled_amount, 
                    'avg_price': avg_price,
                    'buy_count': 1, 'highest_price_after_entry': current_price,
                    'buy_points': [{'time': current_time, 'price': current_price, 'order': 1}],
                    'last_sell_timestamp': None # ⭐️ 매수 성공 시 초기화
                }
                log(f"[{ticker}] 매수 완료. 현재 상태: {trade_states[ticker]}")
            else:
                log(f"🚨 [{ticker}] 매수 주문에 실패했습니다. API 결과: {buy_result}")

    elif current_trade_state['status'] == 'no_position' and current_positions >= config.MAX_POSITIONS:
        log(f"[{ticker}] RL 매수 신호 확인 건너뜀. 최대 보유 종목 수({config.MAX_POSITIONS}개)에 도달했습니다.")

    # 2-2. 포지션을 보유 중인 경우: strategy_trade로 관리
    elif current_trade_state['status'] in ['buying', 'holding']:
        
        # --- 매도 조건 확인 (기존 로직) ---
        is_sell, exit_price, reason = strategy_trade.check_exit_conditions(
            df, -1, current_trade_state, config.STOP_LOSS_PERCENT, config.PROFIT_TARGET_PERCENT_1ST,
            config.PROFIT_LOCK_START_PERCENT_1ST, config.PROFIT_LOCK_TRAILING_PERCENT_1ST,
            config.ATR_MULTIPLIER
        )

        if is_sell:
            log(">"*20 + f" [{ticker}] 매도 신호 발생! ({reason}) " + "<"*20)
            log(f"[{ticker}] [매도 실행] 보유 수량 전량 매도")
            sell_result = client.sell_market_order(ticker)

            if sell_result and 'uuid' in sell_result:
                log(f"✅ [{ticker}] 매도 주문 성공!")
                db_handler.log_trade(ticker, 'SELL', 'SELL', exit_price, current_trade_state['total_amount'], reason=reason)
                
                # ⭐️ 매도 시 상태 초기화 및 쿨다운 시간 기록
                new_state = _create_initial_state()
                new_state['last_sell_timestamp'] = current_time # ⭐️ 현재 시간 기록
                trade_states[ticker] = new_state
                
                log(f"[{ticker}] 매도 완료. 포지션을 정리합니다. 쿨다운 시작.")
            else:
                log(f"🚨 [{ticker}] 매도 주문에 실패했습니다. API 결과: {sell_result}")
            return # 매도했으므로 추가매수 로직은 건너뜀

        # --- 추가매수 조건 확인 (기존 로직) ---
        is_add, order = strategy_trade.should_add_buy(
            current_trade_state, 
            current_price
        )
        
        if is_add:
            log(">"*20 + f" [{ticker}] {order}차 추가매수 신호 발생! " + "<"*20)
            
            current_balance = client.get_balance("KRW")
            if current_balance < 5000:
                log(f"[{ticker}] KRW 잔고 부족으로 추가매수 불가 (잔고: {current_balance:,.0f}원)")
                return
            
            ratio = config.TRADE_RATIO_2ND if order == 2 else config.TRADE_RATIO_3RD
            investment = current_balance * ratio
            
            log(f"[{ticker}] [추가매수 실행] 주문액: {investment:,.0f}")
            buy_result = client.buy_market_order(ticker, investment)

            if buy_result and 'uuid' in buy_result:
                time.sleep(1) # 체결 대기
                log(f"✅ [{ticker}] 추가매수 주문 성공!")
                
                coin_symbol = ticker.split('-')[1]
                new_total_amount = client.get_balance(coin_symbol)
                added_amount = new_total_amount - current_trade_state['total_amount']

                if added_amount <= 0:
                    log(f"🚨 [{ticker}] 추가매수 주문은 성공했으나, 체결 수량이 0입니다.")
                    return

                db_handler.log_trade(ticker, 'BUY', f'ADD_BUY_{order}', current_price, added_amount)
                
                new_total_investment = current_trade_state['total_investment'] + investment
                
                current_trade_state.update({
                    'total_investment': new_total_investment, 
                    'total_amount': new_total_amount,
                    'avg_price': new_total_investment / new_total_amount, 
                    'buy_count': order,
                    'status': 'holding',
                    'buy_points': current_trade_state['buy_points'] + [{'time': current_time, 'price': current_price, 'order': order}]
                })
                log(f"[{ticker}] 추가매수 완료. 현재 상태: {current_trade_state}")
            else:
                log(f"🚨 [{ticker}] 추가매수 주문에 실패했습니다. API 결과: {buy_result}")

def run_trading_logic():
    """[거래 오케스트레이터] 선정된 모든 코인에 대해 거래 로직을 순차적으로 실행합니다."""
    
    all_tickers_to_check = target_tickers

    if not all_tickers_to_check:
        log("... 거래 대상 종목이 없어 대기합니다. (config 확인 필요)")
        return

    log(f"▶ {all_tickers_to_check[0]} 종목에 대한 거래 로직 실행 시작...")

    for ticker in all_tickers_to_check:
        try:
            _run_logic_for_ticker(ticker)
        except Exception as e:
            log(f"🚨 [{ticker}] 거래 로직 실행 중 치명적 오류 발생: {e}")
            import traceback
            log(traceback.format_exc())
            
    log("... 모든 종목 탐색 완료.")

def start_trading():
    """메인 실행 함수 (main.py에서 스레드로 실행됨)"""
    
    global trade_states, client, target_tickers
    
    db_handler.init_db()
    log(">"*20 + " RL 자동매매 트레이더를 시작합니다 (KRW-BTC 전용) " + "<"*20)

    # --- RL 모델 로드 ---
    rl_signal.load_rl_model()
    if rl_signal.RL_MODEL is None:
        log("🚨 RL 모델 로드에 실패하여 프로그램을 종료합니다.")
        return

    # UpbitClient 인스턴스 생성
    client = uc.UpbitClient(config.UPBIT_ACCESS_KEY, config.UPBIT_SECRET_KEY)
    if not client.upbit:
        log("🚨 업비트 클라이언트 생성 실패. 프로그램을 종료합니다.")
        return
    
    # --- 봇 시작 시 KRW-BTC 잔고가 있으면 전량 매도 ---
    try:
        btc_balance = client.get_balance("BTC") 
        
        MIN_SELL_AMOUNT = 0.00008 
        if btc_balance is not None and btc_balance > MIN_SELL_AMOUNT: 
            log(f"✅ [초기화] 기존에 보유 중인 BTC 발견: {btc_balance} BTC. 전량 매도를 시도합니다.")
            sell_result = client.sell_market_order("KRW-BTC")
            
            if sell_result and 'uuid' in sell_result:
                log(f"✅ [초기화] 전량 매도 주문 성공. {sell_result}")
                time.sleep(2) # API 처리를 위한 잠시 대기
            else:
                log(f"🚨 [초기화] 전량 매도 주문 실패. API 결과: {sell_result}")
        else:
            log(f"✅ [초기화] 기존 보유 BTC가 없거나({btc_balance} BTC) 매도 최소 수량 미만입니다.")
            
    except Exception as e:
        log(f"🚨 [초기화] 기존 잔고 매도 중 오류 발생: {e}")
        
    # --- KRW-BTC 전용으로 상태 초기화 (봇 시작 시 1회 실행) ---
    target_tickers = ["KRW-BTC"] # 거래 대상을 KRW-BTC로 고정
    trade_states = {} # 모든 상태를 비움
    trade_states["KRW-BTC"] = _create_initial_state() # KRW-BTC의 초기 상태만 생성
    log(f"✅ [초기화] {target_tickers[0]} 종목에 대한 거래 상태를 생성했습니다.")

    while True:
        try:
            run_trading_logic()

            # --- ⏰ 다음 30분봉 시간까지 대기 ---
            now = datetime.now(KST)
            minutes_to_wait = 30 - (now.minute % 30)
            seconds_to_wait = minutes_to_wait * 60 - now.second
            
            # 캔들 생성을 위한 3초 버퍼
            sleep_duration = seconds_to_wait + 3 
            log(f"다음 캔들(30분봉)까지 {sleep_duration-3}초 대기...")
            time.sleep(sleep_duration)

        except Exception as e:
            log(f"🔴 메인 루프에서 에러 발생: {e}")
            import traceback
            log(traceback.format_exc())
            time.sleep(60) # 에러 발생 시 1분 대기