import pandas as pd
import numpy as np
import os
import time

# --- 1. 데이터 로드 함수 정의 (로컬 파일) ---
def load_local_ohlcv(file_path):
    # 파일 경로 설정 (이전 실행 환경 경로를 가정)
    full_file_path = "0_data/btc_updated.csv"
    
    if not os.path.exists(full_file_path):
        if os.path.exists(file_path):
             full_file_path = file_path
        else:
            print(f"'{full_file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
            return pd.DataFrame()

    print(f"'{full_file_path}' 파일을 불러옵니다.")
    df = pd.read_csv(full_file_path, index_col='datetime', parse_dates=True)
    
    df.columns = df.columns.str.lower()
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"경고: 필수 컬럼 ({missing_cols})이 데이터에 없습니다.")
        return pd.DataFrame()
    
    df = df[required_cols].replace([np.inf, -np.inf], np.nan).dropna()
    
    if df.empty:
        print("경고: 파일 로드 후 데이터가 비어 있거나 OHLCV 컬럼이 부족합니다.")
    else:
        print(f"✅ 데이터 로드 완료. 총 {len(df)}개의 30분봉 데이터 확보.")
        
    return df

# --- 2. 변동성 분석 및 N기간 탐색 함수 ---
def analyze_optimal_n(df, period_options):
    # (이 함수는 N기간별 통계적 분석을 수행하며 이전과 동일)
    results = {}
    print("\n--- 3. 최적의 N기간(보유 시간) 탐색 시작 ---")
    
    for N_PERIODS in period_options:
        future_high = df['high'].rolling(window=N_PERIODS).max().shift(-N_PERIODS + 1)
        future_low = df['low'].rolling(window=N_PERIODS).min().shift(-N_PERIODS + 1)
        
        potential_reward_pct = (future_high / df['close'] - 1) * 100
        potential_risk_pct = (1 - future_low / df['close']) * 100 
        
        data_clean = pd.DataFrame({'Reward': potential_reward_pct, 'Risk': potential_risk_pct}).dropna()

        if data_clean.empty:
            continue

        tp_median = data_clean['Reward'].median()
        sl_90th = data_clean['Risk'].quantile(0.90)
        rr_ratio = tp_median / sl_90th if sl_90th > 0 else np.nan

        results[N_PERIODS] = {
            'Holding_Time': f"{N_PERIODS * 0.5}h",
            'Median_TP': tp_median,
            '90th_SL': sl_90th,
            'RR_Ratio': rr_ratio
        }
        
        print(f"  - N={N_PERIODS} ({N_PERIODS * 0.5}h): TP_med={tp_median:.2f}%, SL_90th={sl_90th:.2f}%, R/R Ratio={rr_ratio:.2f}:1")

    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    if not results_df.empty:
        best_n_row = results_df.sort_values(by='RR_Ratio', ascending=False).iloc[0]
    else:
        best_n_row = pd.Series(dtype=object)

    return results_df, best_n_row

# --- 3. N기간별 유동 변동성 룰 출력 함수 (새로 추가) ---
def get_dynamic_volatility_rules(analysis_df):
    """
    분석된 N기간별 통계치를 기반으로 RL/트레이딩에 사용할 유동적 룰을 출력합니다.
    """
    print("\n" + "="*70)
    print("  📈 N기간별 유동적 익절/손절 룰 (변동성 기반)  ")
    print("="*70)

    # 출력 포맷 조정 및 유동적 룰 정의
    dynamic_rules = analysis_df.copy()
    dynamic_rules['Recommended_TP'] = dynamic_rules['Median_TP'].apply(lambda x: f"{x:.2f}% (Median)")
    dynamic_rules['Required_SL'] = dynamic_rules['90th_SL'].apply(lambda x: f"{x:.2f}% (90th Pct)")
    dynamic_rules['RR_Ratio'] = dynamic_rules['RR_Ratio'].apply(lambda x: f"{x:.2f}:1")

    # 필요한 컬럼만 선택하여 깔끔하게 출력
    print(dynamic_rules[['Holding_Time', 'Recommended_TP', 'Required_SL', 'RR_Ratio']].to_markdown(index=True, floatfmt=".2f"))
    
    print("\n💡 해석: 각 'Holding_Time'을 선택할 경우, 'Recommended_TP'를 익절 목표로 하고 'Required_SL'을 손절 폭으로 설정해야 통계적 잠재력에 부합합니다.")


# --- 메인 실행 로직 ---
if __name__ == "__main__":
    
    # 1. 데이터 로드 
    data_file_path = "0_data/btc_updated.csv"
    btc_df = load_local_ohlcv(data_file_path)

    if btc_df.empty:
        exit()

    # 2. 최적의 N기간 분석 (2h ~ 144h)
    period_options = [4, 8, 12, 24, 48, 96, 144, 192, 288]
    analysis_df, best_n = analyze_optimal_n(btc_df, period_options=period_options)
    
    # 3. N기간별 변동성 룰 출력
    if not analysis_df.empty:
        get_dynamic_volatility_rules(analysis_df)

        print("\n" + "="*70)
        print(f"🥇 **[최적 N기간]** R/R 비율이 가장 높은 기간:")
        print(f"  - 보유 시간: {best_n['Holding_Time']}, R/R Ratio: {best_n['RR_Ratio']:.2f}:1")
        print("="*70)