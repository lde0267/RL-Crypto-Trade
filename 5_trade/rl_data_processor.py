# rl_data_processor.py
import pandas as pd
import numpy as np
import pyupbit
import pickle
import config  # RL_WINDOW_SIZE, STATS_PATH 등 설정을 위함
import ta      # ❗️ 지표 계산을 위해 TA 라이브러리를 import합니다.

# --- 설정 ---
try:
    with open(config.RL_STATS_PATH, 'rb') as f:
        STATS = pickle.load(f)
    # ❗️ [중요] Env 코드를 보면, 통계 파일은 6개의 시장 지표(obs_cols)에만 해당됩니다.
    OBS_MEANS = STATS['means']
    OBS_STDS = STATS['stds']
    OBS_COLS = OBS_MEANS.index.tolist() # 훈련 시 사용한 6개 지표
    print(f"✅ RL 통계 로드 성공. 관측 컬럼: {OBS_COLS}")
except Exception as e:
    print(f"🚨 치명적 오류: RL 통계 파일 '{config.RL_STATS_PATH}' 로드 실패: {e}")
    STATS = None

def calculate_all_indicators(df_full):
    """
    RL 모델 훈련(6개) + 매매 전략(ATR, is_downtrend)에 필요한 모든 지표를 계산합니다.
    """
    df = df_full.copy()
    
    # ❗️ [필수] Env의 self.obs_cols (6개)와 100% 동일하게 계산해야 합니다.
    # ❗️ 아래는 'ta' 라이브러리 사용 예시이며, 실제 훈련 시 사용한 로직으로 대체해야 합니다.
    
    # --- 1. RL 모델 훈련용 지표 (6개) ---
    # 예시: '30_to_60_Close_ratio'
    ma30 = ta.trend.sma_indicator(df['close'], window=30)
    ma60 = ta.trend.sma_indicator(df['close'], window=60)
    df['30_to_60_Close_ratio'] = ma30 / ma60

    # 예시: '60_OBV'
    df['60_OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    # (참고: Env에서는 60분봉 OBV를 썼을 수 있으므로, 이 계산이 정확하지 않을 수 있음)

    # 예시: 'day_of_week'
    df['day_of_week'] = df.index.dayofweek

    # 예시: '30_ATR'
    df['30_ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=30)

    # 예시: '30_Keltner_lband'
    keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=30)
    df['30_Keltner_lband'] = keltner.keltner_channel_lband()

    # 예시: '60_ADX'
    df['60_ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=60)
    # (참고: 60분봉 ADX를 30분봉 데이터로 근사 계산)

    
    # --- 2. strategy_trade.py용 지표 ---
    
    # strategy_trade는 'ATR' 컬럼을 사용합니다. RL의 '30_ATR'을 사용하도록 이름을 복사합니다.
    if '30_ATR' in df.columns:
        df['ATR'] = df['30_ATR']
    else:
        # ❗️ '30_ATR' 계산이 실패/변경될 경우를 대비한 기본값
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    # strategy_trade는 'is_downtrend'를 사용합니다.
    # ❗️ [필수] 훈련 시 사용한 하락장 정의와 동일하게 설정해야 합니다.
    # (예시: 60MA 미만일 때)
    if 'ma60' not in locals(): # ma60이 위에서 계산 안됐을 경우
         ma60 = ta.trend.sma_indicator(df['close'], window=60)
    df['is_downtrend'] = df['close'] < ma60
    
    
    # --- 3. 후처리 (NaN, inf 값) ---
    # ❗️ [필수] 훈련 환경의 NaN/inf 처리 방식과 100% 동일해야 합니다.
    df = df.fillna(0) # 예시: 0으로 채움
    df = df.replace([np.inf, -np.inf], 0) # inf 값 처리
    
    return df

def get_processed_data(ticker, current_trade_state):
    """
    Upbit 데이터를 로드/계산하고,
    RL 모델용 관측값(obs)과 매매 전략용 데이터프레임(df)을 반환합니다.
    
    [수정] current_trade_state를 인자로 받아 포트폴리오 상태를 obs에 포함시킵니다.
    """
    if STATS is None:
        print("🚨 RL 통계 파일이 로드되지 않아 데이터 처리를 중단합니다.")
        return None, None

    # 1. 데이터 로드 (지표 계산을 위해 넉넉하게 200개 로드)
    # ❗️ [수정] 훈련 시 120개 마진을 뒀으므로, 200개면 충분할 것입니다.
    df_full = pyupbit.get_ohlcv(ticker, interval="minute30", count=200)
    if df_full is None:
        print(f"[{ticker}] 데이터 로드 실패.")
        return None, None
    
    # ❗️ 현재 (미완성) 캔들을 버리고, '완성된 캔들'만 사용
    df_full = df_full.iloc[:-1] 
        
    # 2. 모든 지표 계산
    df = calculate_all_indicators(df_full)
    
    try:
        # --- 3. [시장 데이터] 정규화 (Env의 1번째 파트) ---
        
        # 3-1. 훈련에 사용된 6개 컬럼만 선택
        df_obs_features = df[OBS_COLS]
        
        # 3-2. 훈련 통계로 정규화
        df_normalized = (df_obs_features - OBS_MEANS) / OBS_STDS
        
        # 3-3. 훈련과 동일한 Window Size만큼 슬라이싱
        # ❗️ iloc[-config.RL_WINDOW_SIZE:] -> Env의 (start:end) 로직과 동일
        norm_obs_window_values = df_normalized.iloc[-config.RL_WINDOW_SIZE:].values

        # --- 4. [포트폴리오 상태] 생성 (Env의 2번째 파트) ---
        
        # ❗️ [필수] config.py에 훈련 시 사용한 initial_balance 값을 추가해야 합니다.
        # 예: INITIAL_BALANCE = 300000.0
        try:
            initial_balance = config.INITIAL_BALANCE 
        except AttributeError:
            print("🚨 [필수] config.py에 INITIAL_BALANCE = 300000.0 (훈련 시 초기자본) 값을 추가하세요!")
            initial_balance = 300000.0 # 임시 폴백
            
        current_price = df.iloc[-1]['close']
        
        is_holding = 1.0 if current_trade_state['status'] != 'no_position' else 0.0
        avg_buy_price = current_trade_state.get('avg_price', 0.0)
        unrealized_pnl = (current_price - avg_buy_price) / (avg_buy_price + 1e-9) if is_holding else 0.0
        
        # 🔴 [위험] Env의 'balance'는 시뮬레이션 값입니다.
        #          실제 'client.get_balance("KRW")'와 다릅니다.
        #          이 피처가 모델에 큰 영향을 줬다면, 실제 매매 시 성능이 다를 수 있습니다.
        #          여기서는 0.0으로 고정합니다.
        balance_change_ratio = 0.0 
        
        holdings_value = current_trade_state.get('total_amount', 0.0) * current_price
        holdings_value_ratio = holdings_value / initial_balance

        # 🔴 [위험] Env의 'step_idx'는 에피소드 내 시간입니다.
        #          실제 매매에는 이 개념이 없습니다. 
        #          모델이 "시간이 다 되면 매도"하도록 학습했다면, 성능이 다를 수 있습니다.
        #          여기서는 0.0 (항상 에피소드 시작)으로 고정합니다.
        time_step_ratio = 0.0

        portfolio_info = np.array([
            balance_change_ratio,   # (self.balance - self.initial_balance) / ...
            holdings_value_ratio,   # (self.coin_holdings * current_price) / ...
            is_holding,             # is_holding
            unrealized_pnl,         # unrealized_pnl
            time_step_ratio         # self.step_idx / self.episode_length
        ])
        
        # (window_size, 5) 형태로 복제
        portfolio_info_tiled = np.tile(portfolio_info, (config.RL_WINDOW_SIZE, 1))

        # --- 5. 최종 관측값(obs) 생성: (시장 데이터 + 포트폴리오 상태) ---
        obs_array = np.concatenate([norm_obs_window_values, portfolio_info_tiled], axis=1)
        
        # Env의 observation_space shape=(window_size, num_features) 였습니다.
        # Env는 (10, 11)의 2D 데이터를 기대합니다
        rl_observation = obs_array.astype(np.float32)
        
        return rl_observation, df
        
    except KeyError as e:
        print(f"🚨 [{ticker}] RL 데이터 처리 오류: 훈련에 사용된 컬럼({e})이 df에 없습니다.")
        print(f"    df 컬럼: {df.columns.tolist()}")
        return None, None
    except Exception as e:
        print(f"🚨 [{ticker}] RL 관측값 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ==============================================================================
# ❗️ [테스트 코드] (수정 없음)
# ==============================================================================
if __name__ == "__main__":
    
    import pprint
    import time

    print("="*60)
    print("🚀 [TEST] rl_data_processor.py 테스트를 시작합니다.")
    print("="*60)
    
    print("\n" + "!"*60)
    print("⚠️ [경고] 'calculate_all_indicators' 함수 내의 지표 계산 로직은")
    print("           단순 예시(Example)입니다.")
    print("           반드시 훈련 시 사용한 로직과 100% 동일하게 수정해야 합니다!")
    print("!"*60 + "\n")

    # --- 1. 통계 파일 로드 확인 ---
    if STATS is None:
        print("❌ [TEST] 통계 파일 로드에 실패했습니다. 테스트를 중단합니다.")
        print(f"   (경로: {config.RL_STATS_PATH})")
        exit()
    else:
        print(f"✅ [TEST] 통계 파일 로드 성공. (총 {len(OBS_COLS)}개 지표)")
        pprint.pprint(OBS_COLS)

    # --- 2. 테스트용 가상 'trade_state' 정의 ---
    mock_state_no_position = {
        'status': 'no_position', 'avg_price': 0.0, 'total_amount': 0.0
    }
    mock_state_holding = {
        'status': 'holding', 
        'avg_price': 90_000_000.0,
        'total_amount': 0.001       
    }
    TEST_TICKER = "KRW-BTC"

    # --- 3. 테스트 실행 함수 ---
    def run_test_scenario(ticker, state_name, trade_state):
        print("\n" + "-"*50)
        print(f"▶️  시나리오 테스트: '{state_name}'")
        print(f"   (TICKER: {ticker})")
        print(f"   (INPUT STATE: {trade_state})")
        print("-"*50)

        start_time = time.time()
        
        # --- 핵심 함수 호출 ---
        rl_observation, df = get_processed_data(ticker, trade_state)
        
        end_time = time.time()
        print(f"⏱️  데이터 처리 시간: {end_time - start_time:.4f} 초")

        if rl_observation is None or df is None:
            print("❌ [TEST] 실패: get_processed_data가 None을 반환했습니다.")
            return

        print("✅ [TEST] 성공: 데이터 및 관측값 생성 완료.")
        
        # 1. 계산된 DataFrame 확인 (최신 3줄)
        print("\n--- [확인 1] 계산된 지표 (DataFrame 최신 3줄) ---")
        check_cols = OBS_COLS + ['ATR', 'is_downtrend']
        check_cols = [col for col in check_cols if col in df.columns] 
        print(df[check_cols].tail(3))

        # 2. 최종 관측값(Observation) 형태 확인
        print("\n--- [확인 2] 최종 RL 관측값(Observation) 형태 ---")
        print(f"   - Type: {type(rl_observation)}")
        print(f"   - Shape: {rl_observation.shape}")
        
        # ❗️ [수정] Shape 기대값: 1D Flatten 벡터가 아닌 2D (Window, Features)
        expected_shape = (config.RL_WINDOW_SIZE, len(OBS_COLS) + 5)
        if rl_observation.shape == expected_shape:
            print(f"   - ✅ Shape 일치 (기대값: {expected_shape})")
        else:
            print(f"   - ❌ Shape 불일치! (기대값: {expected_shape})")
            print(f"   - ⚠️  (만약 1D Flatten을 의도했다면 테스트 코드를 수정하세요)")


        # 3. 최종 관측값(Observation) 내용 일부 확인
        print("\n--- [확인 3] 최종 RL 관측값(Observation) 일부 (첫 번째 행) ---")
        # (첫 번째 타임스텝의 6개 시장지표 + 5개 포트폴리오 상태)
        pprint.pprint(rl_observation[0])

    # --- 4. 테스트 실행 ---
    run_test_scenario(TEST_TICKER, "포지션 없음 (No Position)", mock_state_no_position)
    run_test_scenario(TEST_TICKER, "포지션 보유 중 (Holding)", mock_state_holding)
    
    print("\n" + "="*60)
    print("✅ [TEST] 모든 테스트가 완료되었습니다.")
    print("="*60)