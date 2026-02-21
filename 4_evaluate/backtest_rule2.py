import pandas as pd
import numpy as np
import pickle
import quantstats as qs
import matplotlib.pyplot as plt
from stable_baselines3 import PPO # ❗️ 훈련에 사용한 알고리즘

# ⭐️ 1. 'test_env.py' 파일에서 TradingEnv 클래스를 import 합니다.
try:
    from test_env_rule2 import TradingEnv 
except ImportError:
    print("="*80)
    print("오류: 'test_env.py' 파일을 찾을 수 없습니다.")
    print("이 스크립트와 동일한 폴더에 'test_env.py' 파일이 있는지 확인하세요.")
    print("="*80)
    exit()

# ==============================================================================
# ❗️ 1. [사용자 설정] 백테스트 기간 및 파일 경로
# ==============================================================================
START_DATE = "2025-08-7 00:00:00" # ❗️ 원하는 시작 날짜
END_DATE = "2025-09-7 22:59:59"   # ❗️ 원하는 종료 날짜

TEST_DATA_PATH = "0_data/btc_updated_6indi.csv"  # ❗️ 전체 테스트 데이터
STATS_PATH = "1_model/obs_stats_btc_rule.pkl"         # ❗️ 훈련 통계 (.pkl)
MODEL_PATH = "1_model/best_model_rule/best_model.zip" #final_model.zip"           # ❗️ 훈련된 모델
DATE_COLUMN = 'datetime'                         # ❗️ 날짜 컬럼명

PLOT_FILENAME = './98_result/rule2_2025_08to09.png'
REPORT_FILENAME = './98_result/rule2_2025_08to09.html'


# ❗️ (중요) 훈련 시 사용한 윈도우 사이즈
WINDOW_SIZE = 10 
# (test_env.py의 reset() 로직 참조: safe_start_margin = window_size + 120)
SAFE_START_MARGIN = WINDOW_SIZE + 120
# ==============================================================================

def load_data_and_filter(start_dt, end_dt):
    """전체 데이터를 로드하고, 지정된 기간으로 필터링합니다."""
    print(f"Loading data from '{TEST_DATA_PATH}'...")
    try:
        df_full = pd.read_csv(TEST_DATA_PATH)
    except FileNotFoundError:
        print(f"오류: 테스트 데이터 '{TEST_DATA_PATH}'를 찾을 수 없습니다.")
        return None
        
    # 1. 날짜 컬럼을 datetime으로 변환 (시간대 정보가 있다면 유지)
    df_full[DATE_COLUMN] = pd.to_datetime(df_full[DATE_COLUMN])
    
    # 2. 기간 필터링
    mask = (df_full[DATE_COLUMN] >= start_dt) & (df_full[DATE_COLUMN] <= end_dt)
    df_period = df_full[mask].reset_index(drop=True)
    
    if len(df_period) <= SAFE_START_MARGIN:
        print(f"오류: 지정된 기간의 데이터가 너무 짧습니다 (길이: {len(df_period)}).")
        print(f"안전 마진({SAFE_START_MARGIN})보다 긴 기간이 필요합니다.")
        return None
        
    print(f"Filtered data for period: {len(df_period)} steps")
    return df_period

def load_stats_and_model():
    """통계 파일과 모델 파일을 로드합니다."""
    print(f"Loading stats from '{STATS_PATH}'...")
    try:
        with open(STATS_PATH, 'rb') as f:
            stats = pickle.load(f)
        obs_means = stats['means']
        obs_stds = stats['stds']
    except Exception as e:
        print(f"오류: 통계 파일 '{STATS_PATH}' 로드 실패: {e}")
        return None, None, None

    print(f"Loading model from '{MODEL_PATH}'...")
    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"오류: 모델 '{MODEL_PATH}' 로드 실패: {e}")
        return None, None, None
        
    return model, obs_means, obs_stds

def run_backtest(model, env, df_period):
    """
    백테스트 루프를 실행하고, 포트폴리오 내역과 모든 거래를 기록합니다.
    """
    print("--- 백테스트 평가 시작 ---")
    
    obs, info = env.reset()
    terminated, truncated = False, False

    # 3-1. 지표 수집을 위한 리스트 초기화
    portfolio_log = [] # (date, portfolio_value)
    trade_log = []     # (date, 'BUY'/'SELL', price)

    # ⭐️ [수정] reset() 시점의 '직전' 날짜와 '초기 자본'을 기록합니다.
    initial_value = env.initial_balance
    initial_date_idx = SAFE_START_MARGIN - 1 # reset() 직전 인덱스
    
    if initial_date_idx < 0:
        initial_date = pd.to_datetime(df_period.loc[0, DATE_COLUMN])
    else:
        initial_date = df_period.loc[initial_date_idx, DATE_COLUMN]

    # ⭐️ 포트폴리오 로그의 첫 번째 시점: 거래 시작 직전의 초기 자본
    portfolio_log.append((initial_date, initial_value))
    
    while not terminated and not truncated:
        # ⭐️ 3-3. 거래 감지를 위해 step 이전의 보유 상태 저장
        holding_before = env.coin_holdings > 0
        
        # 3-4. deterministic=True (탐험 끄기)
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # ⭐️ 3-5. step 이후의 보유 상태 확인
        holding_after = env.coin_holdings > 0
        
        # 3-6. 'info' 딕셔너리에서 지표 수집
        current_value = info.get('portfolio_value')
        current_date = info.get('date')
        
        # (현재 스텝은 1 증가했으므로, -1을 하여 현재 스텝의 가격을 가져옴)
        current_price = df_period.loc[env.current_step - 1, 'Close']
        
        portfolio_log.append((current_date, current_value))
        
        # ⭐️ 3-7. 거래 로깅: 상태 변화 감지
        if not holding_before and holding_after:
            # (미보유 -> 보유) = 매수
            trade_log.append((current_date, 'BUY', current_price))
        elif holding_before and not holding_after:
            # (보유 -> 미보유) = 매도 (트레일링 스탑 포함)
            trade_log.append((current_date, 'SELL', current_price))
        
    print("--- 백테스트 평가 완료 ---")
    
    # 4-1. 수익률 시리즈(Returns Series) 생성
    portfolio_df = pd.DataFrame(portfolio_log, columns=['date', 'value'])
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    
    # ⭐️ [수정] 날짜 인덱스 설정 후 시간대(Timezone) 정보를 제거합니다.
    portfolio_df = portfolio_df.set_index('date').dropna()
    if portfolio_df.index.tz is not None:
        portfolio_df.index = portfolio_df.index.tz_localize(None)

    returns_series = portfolio_df['value'].pct_change().fillna(0)
    returns_series.name = 'RL_Model'
    
    # 4-2. 벤치마크 (Buy and Hold) 수익률 생성
    # ⭐️ [수정] 벤치마크 시작점을 명확히 정의합니다.
    start_step = SAFE_START_MARGIN 
    end_step = env.current_step - 1 

    benchmark_data = df_period.loc[start_step:end_step, [DATE_COLUMN, 'Close']]
    benchmark_series = pd.Series(
        benchmark_data['Close'].values, 
        index=pd.to_datetime(benchmark_data[DATE_COLUMN])
    )
    
    # ⭐️ [수정] 벤치마크 인덱스의 시간대(Timezone) 정보도 제거합니다.
    if benchmark_series.index.tz is not None:
        benchmark_series.index = benchmark_series.index.tz_localize(None)

    # 벤치마크 가격 시리즈를 자산 곡선으로 변환 (초기 가격을 초기 자산으로 가정)
    first_price = benchmark_series.iloc[0]
    benchmark_value_series = (benchmark_series / first_price) * env.initial_balance
    
    # RL 모델 로그의 첫 번째 날짜(initial_date)를 벤치마크에도 추가하여 인덱스를 맞춥니다.
    # portfolio_df 인덱스를 사용
    benchmark_log = pd.Series(
        [env.initial_balance] + benchmark_value_series.tolist(),
        index=portfolio_df.index
    )

    benchmark_returns = benchmark_log.pct_change().fillna(0)
    benchmark_returns.name = 'Buy_and_Hold'
    
    # 최종적으로 두 수익률 시리즈가 동일한 길이와 인덱스를 가지는지 확인합니다.
    if not returns_series.index.equals(benchmark_returns.index):
        # 인덱스가 다르면, 두 시리즈를 합친 후, returns_series의 인덱스로 재정렬합니다.
        combined_returns = pd.concat([returns_series, benchmark_returns], axis=1).fillna(0)
        returns_series = combined_returns['RL_Model']
        benchmark_returns = combined_returns['Buy_and_Hold']
    
    return returns_series, benchmark_returns, trade_log

def generate_metrics(returns_series, benchmark_returns, trade_log):
    """QuantStats를 사용해 상세 지표를 출력하고 HTML 리포트를 저장합니다."""
    
    print("\n--- 📈 1. Performance Metrics (RL Model vs. B&H) ---")
    # ⭐️ [수정] 수익률이 모두 0이 아닌지 확인하여 벤치마크 오류 방지
    if returns_series.sum() == 0 and benchmark_returns.sum() == 0:
        print("경고: RL 모델 및 벤치마크 수익률이 모두 0이므로, QuantStats를 실행하지 않습니다.")
        print("데이터 범위와 거래 로직을 확인하세요.")
        return
        
    qs.reports.metrics(returns_series, benchmark=benchmark_returns, display=True)
    
    print("\n--- 📊 2. Trade Log (Last 20 Trades) ---")
    buys = len([t for t in trade_log if t[1] == 'BUY'])
    sells = len([t for t in trade_log if t[1] == 'SELL'])
    print(f"Total Buys: {buys}")
    print(f"Total Sells: {sells}")
    print("-" * 30)
    # 마지막 20개 거래만 출력
    for trade in trade_log[-20:]:
        print(f"  {trade[0]} | {trade[1]:<4} | @ {trade[2]:,.0f} KRW")

    try:
        qs.reports.html(
            returns_series, 
            benchmark=benchmark_returns,
            output=REPORT_FILENAME, 
            title=f'RL Model Backtest ({START_DATE} to {END_DATE})'
        )
        print(f"\n✅ 성공: 상세 리포트가 '{REPORT_FILENAME}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"\n❌ QuantStats HTML 리포트 생성 실패: {e}")

def plot_results(df_period, returns_series, trade_log, initial_balance):
    """가격 차트, 매매 시점, 자산 곡선을 그래프로 저장합니다."""
    
    print("\n--- 🖼️ 3. Generating Plot... ---")
    
    # 1. 플롯을 위한 데이터 준비
    # 1-1. 가격 데이터 (전체 기간)
    price_data = df_period.set_index(DATE_COLUMN)['Close']
    # ⭐️ [수정] 시간대 제거
    if price_data.index.tz is not None:
        price_data.index = price_data.index.tz_localize(None)

    # 1-2. 자산 곡선 (수익률로부터 계산)
    equity_curve = (1 + returns_series).cumprod() * initial_balance
    # 1-3. 매매 시점
    buys = [t for t in trade_log if t[1] == 'BUY']
    sells = [t for t in trade_log if t[1] == 'SELL']
    
    # 벤치마크 자산 곡선 계산
    benchmark_equity = (1 + benchmark_returns).cumprod() * initial_balance

    # 2. 그래프 그리기 (2행 1열)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(18, 12), 
        sharex=True, # X축(날짜) 공유
        gridspec_kw={'height_ratios': [2, 1]} # 위 그래프를 더 크게
    )
    
    fig.suptitle(f"Backtest Result: {START_DATE} to {END_DATE}", fontsize=16)

    # --- Plot 1: 가격 차트 + 매매 시점 ---
    ax1.plot(price_data.index, price_data.values, label='BTC Price', color='deepskyblue', alpha=0.7)
    
    if buys:
        buy_dates, _, buy_prices = zip(*buys)
        # ⭐️ [수정] buy_dates에 시간대 정보가 있다면 제거
        buy_dates_tz_free = [d.tz_localize(None) if d.tz is not None else d for d in buy_dates]
        ax1.scatter(buy_dates_tz_free, buy_prices, marker='^', color='green', s=120, alpha=1.0, label='Buy', edgecolors='black')
    if sells:
        sell_dates, _, sell_prices = zip(*sells)
        # ⭐️ [수정] sell_dates에 시간대 정보가 있다면 제거
        sell_dates_tz_free = [d.tz_localize(None) if d.tz is not None else d for d in sell_dates]
        ax1.scatter(sell_dates_tz_free, sell_prices, marker='v', color='red', s=120, alpha=1.0, label='Sell', edgecolors='black')
    
    ax1.set_ylabel('Price (KRW)', fontsize=12)
    ax1.set_title('Price Chart with Buy/Sell Signals', fontsize=14)
    ax1.legend()
    ax1.grid(True, which='major', linestyle='--', alpha=0.5)

    # --- Plot 2: 자산 곡선 (Equity Curve) ---
    ax2.plot(equity_curve.index, equity_curve.values, label='Portfolio Value', color='blue')
    
    # 벤치마크(B&H) 자산 곡선
    ax2.plot(benchmark_equity.index, benchmark_equity.values, label='Buy & Hold', color='grey', linestyle='--', alpha=0.8)

    ax2.set_ylabel('Portfolio Value (KRW)', fontsize=12)
    ax2.set_title('Portfolio Equity Curve', fontsize=14)
    ax2.legend()
    ax2.grid(True, which='major', linestyle='--', alpha=0.5)

    # X축 날짜 포맷팅
    fig.autofmt_xdate()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # (sup title 공간 확보)
    
    try:
        plt.savefig(PLOT_FILENAME)
        print(f"✅ 성공: 그래프가 '{PLOT_FILENAME}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"❌ 그래프 저장 실패: {e}")
    # plt.show() # (선택) 그래프를 화면에 바로 띄우려면 주석 해제


# ==============================================================================
# 4. [메인 실행 로직]
# ==============================================================================
if __name__ == "__main__":
    
    # 1. 데이터 로드 및 필터링
    df_period = load_data_and_filter(START_DATE, END_DATE)
    if df_period is None:
        exit()
        
    # 2. 모델 및 통계 로드
    model, obs_means, obs_stds = load_stats_and_model()
    if model is None:
        exit()
        
    # 3. 백테스트 환경 초기화
    # ❗️ (중요) env가 사용하는 컬럼 리스트 (test_env.py 코드와 동일해야 함)
    env_obs_cols = obs_means.index
    
    # (중요) episode_length: (필터링된 길이) - (시작 마진)
    full_episode_length = len(df_period) - SAFE_START_MARGIN
    
    env = TradingEnv(
        df=df_period,
        obs_means=obs_means,
        obs_stds=obs_stds,
        window_size=WINDOW_SIZE,
        episode_length=full_episode_length, # 👈 필터링된 전체 길이로 설정
        # (기타 파라미터는 훈련 시와 동일하게 설정)
        trailing_stop_pct=0.007,
        trade_ratio=0.5,
        stop_loss_pct=0.01,            # 예: 1.5%
        take_profit_pct=0.01
    )
    
    # 4. 백테스트 실행 및 로그 수집
    returns_series, benchmark_returns, trade_log = run_backtest(model, env, df_period)
    
    # 5. 지표 계산 및 출력
    generate_metrics(returns_series, benchmark_returns, trade_log)
    
    # 6. 그래프 생성 및 저장
    plot_results(df_period, returns_series, trade_log, env.initial_balance)
    
    print("\n--- 🚀 모든 작업 완료 ---")