import pandas as pd
import numpy as np
import pickle
import quantstats as qs
import matplotlib.pyplot as plt
import pyupbit
import ta 
from stable_baselines3 import PPO
from typing import Dict, Any, List, Tuple
import warnings

# 경고 무시 (QuantStats 등)
warnings.filterwarnings('ignore')

# ==============================================================================
# ❗️ [1. 사용자 설정]
# ==============================================================================
CONFIG = {
    "TICKER": "KRW-BTC",
    "INTERVAL": "minute30", 
    "DATA_FETCH_DAYS": 30, 
    "MODEL_PATH": "1_model/best_model_auto/best_model.zip", 
    "STATS_PATH": "1_model/obs_stats_btc_auto.pkl",
    "WINDOW_SIZE": 10,
    
    # ❗️ (중요) 훈련된 모델이 사용한 9개의 보조지표 리스트
    "OBS_COLS": [
        '60_BB_Width', '30_VPT', '30_ADI', 'day_of_week', '30_OBV', 
        '30_to_60_Close_ratio', '30_BB_High', '30_ATR', '60_ADX'
    ],
    
    # ❗️ (중요) 훈련 시 사용한 포트폴리오 정보 개수
    "PORTFOLIO_INFO_LEN": 5,
    
    # --- 시뮬레이션 설정 (test_env2.py와 동일하게) ---
    "INITIAL_BALANCE": 300_000.0,
    "TRADE_RATIO": 0.5, 
    "FEE": 0.0005, 
    "MIN_TRADE_KRW": 5000.0,
    
    # ❗️ 보조지표 계산을 위한 최소 데이터 마진
    "INDICATOR_WARMUP_MARGIN": 120,
    
    # ⭐️ [추가] 훈련 환경에서 정의된 강제 청산 스텝
    "MAX_HOLD_STEPS": 48 
}
# ==============================================================================


def fetch_data(ticker: str, interval: str, days: int) -> pd.DataFrame:
    """1. Upbit API를 통해 최근 데이터를 가져옵니다."""
    print(f"--- 1. {ticker} {interval} (최근 {days}일) 데이터 로드 중 ---")
    
    count_to_fetch = (24 * (60 // int(interval.replace("minute", "")))) * days
    
    try:
        df = pyupbit.get_ohlcv(ticker, interval=interval, count=count_to_fetch)
        if df is None or len(df) == 0:
            raise Exception("API 데이터 수신 실패 (None)")
            
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Value']
        df.index.name = 'datetime'
        
        print(f"데이터 로드 완료: {len(df)}개 캔들 ({df.index.min()} ~ {df.index.max()})")
        return df
    
    except Exception as e:
        print(f"오류: 데이터 로드 실패. {e}")
        print("인터넷 연결 및 Ticker/Interval 설정을 확인하세요.")
        return pd.DataFrame()

def calculate_indicators(df: pd.DataFrame, obs_cols: List[str]) -> pd.DataFrame:
    """보조지표 계산 로직 (기존과 동일)"""
    print("--- 2. 보조지표 계산 (훈련 모델 9개 지표 로직 적용) ---")
    
    # --- 1. 60분봉 리샘플링 (훈련 스크립트와 동일) ---
    print("60분봉 데이터를 리샘플링하여 병합합니다.")
    logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    
    # Upbit API 데이터 타임존 처리 (훈련 데이터 'Asia/Seoul' 기준)
    if df.index.tz is None:
        try:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Seoul')
        except Exception as e:
            print(f"경고: 타임존 변환 실패: {e}. 현지 시간대 가정.")
            pass
    else:
        df.index = df.index.tz_convert('Asia/Seoul')

    # ⭐️ [수정] QuantStats 호환성을 위해 Timezone 제거
    # df.index = df.index.tz_localize(None) 
    # -> QuantStats에서 Timezone을 요구하므로, 아래 generate_results에서 처리함

    df_60 = df.resample('60min', closed='left', label='left').agg(logic).dropna()
    df_60 = df_60.add_prefix('60_')
    df_merged = pd.merge_asof(df, df_60, left_index=True, right_index=True, direction='backward')
    
    df_final = df_merged.fillna(method='ffill') 

    # --- 2. 9개 보조지표 계산 (test_env2.py 기준) ---
    print("모델이 훈련된 9개 핵심 보조지표를 'ta' 라이브러리로 계산합니다...")

    # 1. '60_BB_Width' (window=60)
    bbands_60 = ta.volatility.BollingerBands(df_final['60_Close'], window=60)
    df_final['60_BB_Width'] = bbands_60.bollinger_wband()

    # 2. '30_VPT'
    df_final['30_VPT'] = ta.volume.volume_price_trend(df_final['Close'], df_final['Volume'])

    # 3. '30_ADI'
    df_final['30_ADI'] = ta.volume.acc_dist_index(df_final['High'], df_final['Low'], df_final['Close'], df_final['Volume'])

    # 4. 'day_of_week'
    df_final['day_of_week'] = df_final.index.dayofweek.astype(float)

    # 5. '30_OBV' (30분봉 원본 기준)
    df_final['30_OBV'] = ta.volume.on_balance_volume(df_final['Close'], df_final['Volume']) 

    # 6. '30_to_60_Close_ratio'
    df_final['30_to_60_Close_ratio'] = df_final['Close'] / (df_final['60_Close'] + 1e-6) 

    # 7. '30_BB_High' (window=30)
    bbands_30 = ta.volatility.BollingerBands(df_final['Close'], window=30)
    df_final['30_BB_High'] = bbands_30.bollinger_hband()

    # 8. '30_ATR' (window=30)
    df_final['30_ATR'] = ta.volatility.average_true_range(df_final['High'], df_final['Low'], df_final['Close'], window=30)

    # 9. '60_ADX' (window=60)
    df_final['60_ADX'] = ta.trend.adx(df_final['60_High'], df_final['60_Low'], df_final['60_Close'], window=60)
    
    print("9개 지표 계산 완료.")
    
    missing_cols = [col for col in obs_cols if col not in df_final.columns]
    
    if missing_cols:
        print(f"오류: 필수 보조지표가 누락되었습니다: {missing_cols}")
        raise ValueError("필수 보조지표 누락")
        
    df_cleaned = df_final.dropna()
    print(f"NaN 제거 후 최종 데이터 행 수: {len(df_cleaned)}")

    if len(df_cleaned) == 0:
        print("치명적 오류: NaN 조합으로 인해 모든 행이 제거되었습니다.")
        raise ValueError("데이터 부족 (NaN)")
    
    return df_cleaned

def load_model_and_stats(config: Dict[str, Any]) -> Tuple[Any, pd.Series, pd.Series]:
    """모델과 정규화 통계(.pkl)를 로드합니다. (기존과 동일)"""
    print("--- 3. 모델 및 정규화 통계 로드 중 ---")
    try:
        model = PPO.load(config['MODEL_PATH'])
    except Exception as e:
        print(f"오류: 모델 로드 실패 ({config['MODEL_PATH']}): {e}")
        raise
        
    try:
        with open(config['STATS_PATH'], 'rb') as f:
            stats = pickle.load(f)
        
        obs_means = pd.Series(stats['means'])
        obs_stds = pd.Series(stats['stds'])
        
        if len(obs_means) != len(config['OBS_COLS']):
            print("="*80)
            print(f"오류: 설정 오류 (OBS_COLS)")
            print("="*80)
            raise ValueError("보조지표 개수 불일치")
            
    except Exception as e:
        print(f"오류: 통계 파일 로드 실패 ({config['STATS_PATH']}): {e}")
        raise
        
    print("모델 및 통계 로드 완료.")
    return model, obs_means, obs_stds

def run_backtest(
    model: Any, 
    df: pd.DataFrame, 
    obs_means: pd.Series, 
    obs_stds: pd.Series, 
    config: Dict[str, Any]
) -> Tuple[List[Tuple], List[Tuple]]:
    """3. 롤링(Rolling) 방식으로 매 스텝을 진행하며 백테스트를 실행합니다."""
    print("--- 4. 롤링 백테스트 실행 중 ---")
    
    # --- 설정값 로드 ---
    window_size = config['WINDOW_SIZE']
    obs_cols = config['OBS_COLS']
    MAX_HOLD_STEPS = config['MAX_HOLD_STEPS'] # ⭐️ [추가] 강제 청산 스텝
    
    # --- 포트폴리오 상태 변수 ---
    balance = config['INITIAL_BALANCE']
    holdings = 0.0
    avg_buy_price = 0.0
    steps_since_buy = 0 # ⭐️ [추가] 보유 기간 카운터
    
    # --- 로그 기록용 리스트 ---
    portfolio_log = [] 
    trade_log = [] 
    
    # --- 실제 거래 시작 지점 설정 ---
    start_margin = config['INDICATOR_WARMUP_MARGIN'] + config['WINDOW_SIZE']
    if len(df) < start_margin:
        print("오류: 데이터가 너무 짧아 백테스트를 실행할 수 없습니다.")
        return [], []

    initial_log_time = df.index[start_margin - 1]
    portfolio_log.append((initial_log_time, balance))

    # --- 메인 롤링 루프 ---
    for i in range(start_margin, len(df)):
        
        current_price = df.iloc[i]['Close']
        current_time = df.index[i]
        
        is_holding = holdings > 0
        force_sell = False # ⭐️ [추가] 강제 매도 플래그

        # ⭐️⭐️⭐️ [1. 강제 청산 로직 반영] ⭐️⭐️⭐️
        if is_holding:
            steps_since_buy += 1 
            if steps_since_buy >= MAX_HOLD_STEPS:
                # 24시간 초과 시 강제 매도 실행
                force_sell = True
                
        # 2. 관측치(Obs) 생성 (test_env2.py의 _get_obs 로직)
        
        # 2-1. 보조지표 윈도우 (정규화)
        end_iloc = i + 1
        start_iloc = end_iloc - window_size
        window_df = df[obs_cols].iloc[start_iloc:end_iloc]
        norm_obs_window = (window_df - obs_means) / obs_stds
        
        # 2-2. 포트폴리오 정보 (정규화)
        is_holding_float = 1.0 if holdings > 0 else 0.0
        unrealized_pnl = (current_price - avg_buy_price) / (avg_buy_price + 1e-9) if is_holding else 0.0
        balance_pnl = (balance - config['INITIAL_BALANCE']) / (config['INITIAL_BALANCE'] * 0.5)
        asset_value = (holdings * current_price) / config['INITIAL_BALANCE']
        
        # ⭐️ [수정] 에피소드 진행률 대신 보유 시간 비율 사용
        holding_time_ratio = (steps_since_buy / MAX_HOLD_STEPS) if is_holding else 0.0 
        
        portfolio_info = np.array([
            balance_pnl, asset_value, is_holding_float, unrealized_pnl, holding_time_ratio # ⭐️ [수정]
        ])
        
        portfolio_info_tiled = np.tile(portfolio_info, (window_size, 1))
        
        # 2-3. Obs 결합
        obs = np.concatenate([norm_obs_window.values, portfolio_info_tiled], axis=1).astype(np.float32)
        
        if obs.shape[0] < window_size:
            padding = np.zeros((window_size - obs.shape[0], obs.shape[1]))
            obs = np.concatenate([padding, obs], axis=0)

        # 3. 모델 예측 (0:유지, 1:매수, 2:매도)
        action, _ = model.predict(obs, deterministic=True)
        
        # ⭐️ [추가] 강제 매도 시 모델의 예측을 무시
        if force_sell:
            action = 2 # 강제 매도

        # 4. 거래 로직 실행 (test_env2.py의 _buy/_sell 로직)
        
        # --- (매수) ---
        if action == 1 and not is_holding:
            cost_to_spend = balance * config['TRADE_RATIO']
            if cost_to_spend >= config['MIN_TRADE_KRW']:
                buy_qty = (cost_to_spend / current_price) / (1 + config['FEE'])
                cost = buy_qty * current_price * (1 + config['FEE'])
                
                balance -= cost
                holdings = buy_qty
                avg_buy_price = current_price
                steps_since_buy = 0 # ⭐️ [추가] 매수 시 카운터 초기화
                
                trade_log.append((current_time, 'BUY', current_price, buy_qty))

        # --- (매도) ---
        elif action == 2 and is_holding:
            sell_qty = holdings
            revenue = sell_qty * current_price * (1 - config['FEE'])
            
            balance += revenue
            holdings = 0.0
            avg_buy_price = 0.0
            steps_since_buy = 0 # ⭐️ [추가] 매도 시 카운터 초기화
            
            trade_log.append((current_time, 'SELL', current_price, sell_qty))
            
        # 5. 포트폴리오 가치 기록
        total_value = balance + (holdings * current_price)
        portfolio_log.append((current_time, total_value))

    # 에피소드 종료 시 강제 청산 (백테스트 종료 시)
    if holdings > 0:
        revenue = holdings * current_price * (1 - config['FEE'])
        balance += revenue
        holdings = 0.0
        # trade_log에 마지막 청산 기록은 선택 사항 (일반적으로 최종 자산 기록만 함)
        
    print(f"백테스트 완료. (총 {len(portfolio_log) - 1} 스텝 실행)")
    return portfolio_log, trade_log

def generate_results(
    portfolio_log: List[Tuple], 
    trade_log: List[Tuple], 
    df_backtest: pd.DataFrame,
    start_margin: int,
    config: Dict[str, Any] # ⭐️ [수정] config 인수를 추가
):
    """4. 결과를 정리하고 QuantStats 리포트 및 플롯을 생성합니다."""
    print("--- 5. 결과 생성 및 리포트 저장 중 ---")
    
    if not portfolio_log:
        print("경고: 백테스트 결과가 없습니다. 리포트를 생성할 수 없습니다.")
        return

    # 1. 수익률 시리즈 생성
    portfolio_df = pd.DataFrame(portfolio_log, columns=['date', 'value']).set_index('date')
    
    # ⭐️ [수정] Timezone을 제거하여 QuantStats 호환성을 높임
    portfolio_df.index = portfolio_df.index.tz_convert(None) 
    
    returns_series = portfolio_df['value'].pct_change().fillna(0)
    returns_series.name = 'RL_Model'

    # 2. 벤치마크 (Buy & Hold) 수익률 생성
    benchmark_price = df_backtest['Close'].iloc[start_margin:]
    
    # ⭐️ [수정] 벤치마크에서도 Timezone을 제거하여 비교 가능하게 함
    if benchmark_price.index.tz is not None:
        benchmark_price.index = benchmark_price.index.tz_convert(None)
        
    benchmark_returns = benchmark_price.pct_change().fillna(0)
    benchmark_returns.name = 'Buy_and_Hold'
    
    # 3. QuantStats 리포트
    REPORT_FILENAME = 'live_backtest_report.html'
    try:
        # ❗️ returns_series와 benchmark_returns의 Timezone이 모두 None이어야 합니다.
        qs.reports.html(
            returns_series, 
            benchmark=benchmark_returns,
            output=REPORT_FILENAME, 
            title=f'Live Backtest Report ({CONFIG["TICKER"]})'
        )
        print(f"\n✅ 성공: 상세 리포트가 '{REPORT_FILENAME}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"\n❌ QuantStats HTML 리포트 생성 실패: {e}")
        print("수익률이 0일 수 있습니다. 거래 로그를 확인하세요.")

    # 4. 거래 로그 출력
    print("\n--- 📊 Trade Log (Last 20 Trades) ---")
    buys = len([t for t in trade_log if t[1] == 'BUY'])
    sells = len([t for t in trade_log if t[1] == 'SELL'])
    print(f"Total Buys: {buys}")
    print(f"Total Sells: {sells}")
    print("-" * 40)
    for trade in trade_log[-20:]:
        # ⭐️ [수정] trade[0]의 Timezone을 제거하고 출력
        if trade[0].tz is not None:
             dt_local = trade[0].tz_convert(None)
        else:
             dt_local = trade[0]
        
        print(f"   {dt_local} | {trade[1]:<4} | @ {trade[2]:,.0f} | Qty: {trade[3]:.4f}")

    # 5. Matplotlib 플롯 생성
    PLOT_FILENAME = 'live_backtest_plot.png'
    try:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, 
            figsize=(18, 12), 
            sharex=True,
            gridspec_kw={'height_ratios': [2, 1]}
        )
        
        # --- Plot 1: 가격 + 매매 시점 ---
        price_data = df_backtest['Close'] # 전체 기간 가격
        
        # ⭐️ [수정] Timezone을 제거하여 플로팅 호환성을 높임
        if price_data.index.tz is not None:
            price_data.index = price_data.index.tz_convert(None)
        
        ax1.plot(price_data.index, price_data.values, label='Price', color='deepskyblue', alpha=0.7)
        
        if trade_log:
            buy_trades = [t for t in trade_log if t[1] == 'BUY']
            sell_trades = [t for t in trade_log if t[1] == 'SELL']
            
            # ⭐️ [수정] 거래 로그 날짜에서 Timezone 제거
            if buy_trades:
                buy_dates_tz = [t[0].tz_convert(None) if t[0].tz is not None else t[0] for t in buy_trades]
                buy_prices = [t[2] for t in buy_trades]
                ax1.scatter(buy_dates_tz, buy_prices, marker='^', color='green', s=120, label='Buy', edgecolors='black')
            if sell_trades:
                sell_dates_tz = [t[0].tz_convert(None) if t[0].tz is not None else t[0] for t in sell_trades]
                sell_prices = [t[2] for t in sell_trades]
                ax1.scatter(sell_dates_tz, sell_prices, marker='v', color='red', s=120, label='Sell', edgecolors='black')
        
        ax1.set_ylabel('Price', fontsize=12)
        ax1.set_title(f'Backtest: {CONFIG["TICKER"]} Price & Trades', fontsize=14)
        ax1.legend()
        ax1.grid(True, which='major', linestyle='--', alpha=0.5)

        # --- Plot 2: 자산 곡선 ---
        equity_curve = portfolio_df['value']
        benchmark_equity = (1 + benchmark_returns).cumprod() * config['INITIAL_BALANCE']
        
        ax2.plot(equity_curve.index, equity_curve.values, label='RL Model', color='blue')
        ax2.plot(benchmark_equity.index, benchmark_equity.values, label='Buy & Hold', color='grey', linestyle='--', alpha=0.8)
        
        ax2.set_ylabel('Portfolio Value', fontsize=12)
        ax2.set_title('Portfolio Equity Curve', fontsize=14)
        ax2.legend()
        ax2.grid(True, which='major', linestyle='--', alpha=0.5)

        fig.autofmt_xdate()
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(PLOT_FILENAME)
        print(f"✅ 성공: 그래프가 '{PLOT_FILENAME}' 파일로 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 그래프 저장 실패: {e}")

# ==============================================================================
# 5. [메인 실행 로직]
# ==============================================================================
if __name__ == "__main__":
    try:
        # 1. 데이터 가져오기
        df_raw = fetch_data(CONFIG['TICKER'], CONFIG['INTERVAL'], CONFIG['DATA_FETCH_DAYS'])
        if df_raw.empty:
            raise Exception("데이터 로드 실패")

        # 2. 보조지표 계산 
        df_with_indicators = calculate_indicators(df_raw, CONFIG['OBS_COLS'])
        
        # 3. 모델 및 통계 로드
        model, obs_means, obs_stds = load_model_and_stats(CONFIG)
        
        # 4. 백테스트 실행
        portfolio_log, trade_log = run_backtest(
            model, 
            df_with_indicators, 
            obs_means, 
            obs_stds, 
            CONFIG
        )
        
        # 5. 결과 생성
        # backtest2.py 파일의 가장 아래 if __name__ == "__main__": 블록 내에서 이 부분을 찾아서 수정하세요.

        # 5. 결과 생성
        generate_results(
            portfolio_log, 
            trade_log, 
            df_with_indicators, # 벤치마크 및 플로팅용
            CONFIG['INDICATOR_WARMUP_MARGIN'] + CONFIG['WINDOW_SIZE'],
            CONFIG # ⭐️ [수정] CONFIG 딕셔너리를 전달
        )
        
        print("\n--- 🚀 모든 작업 완료 ---")

    except NotImplementedError as e:
        print(f"\n❌ [중단] {e}")
    except Exception as e:
        print(f"\n❌ 백테스트 실행 중 치명적인 오류 발생: {e}")