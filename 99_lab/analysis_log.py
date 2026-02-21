import pyupbit
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import time
import logging

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 설정 파라미터 ---
MARKET_TICKER = "KRW-BTC"  # 분석할 마켓 코드 (변경 가능)
INTERVAL = "minute30"      # 봉 단위 (30분봉)
COUNT = 200                # 불러올 봉의 개수 (약 4일치)

# --- 분석 파라미터 ---
# 'distance'는 두 극점 사이의 최소 봉 개수를 정의합니다.
# 2로 설정하면, 최소 3개 봉(극점-반전-새로운 극점)이 있어야 새로운 극점으로 인정합니다.
MIN_PEAK_DISTANCE = 3

def find_extrema_and_calculate_swings(df: pd.DataFrame, distance: int) -> pd.DataFrame:
    """
    DataFrame의 종가를 기반으로 지역 극점(Peaks/Troughs)을 찾고, 
    이 극점 간의 등락폭(Swing)을 계산합니다.
    """
    prices = df['close'].values
    
    # 1. 지역 국대점 (Local Maxima/Peaks) 찾기
    peaks_indices, _ = find_peaks(prices, distance=distance)
    
    # 2. 지역 국소점 (Local Minima/Troughs) 찾기 (가격에 -를 붙여 국소점을 국대점으로 변환하여 찾음)
    troughs_indices, _ = find_peaks(-prices, distance=distance)
    
    # 3. 모든 극점(Maxima & Minima)의 인덱스를 통합하고 시간 순으로 정렬합니다.
    extrema_indices = np.sort(np.unique(np.concatenate([peaks_indices, troughs_indices])))
    
    # 4. 극점 간의 스윙(파동) 등락폭 계산
    swing_results = []
    
    for i in range(1, len(extrema_indices)):
        current_idx = extrema_indices[i]
        prev_idx = extrema_indices[i-1]
        
        current_price = df.iloc[current_idx]['close']
        prev_price = df.iloc[prev_idx]['close']
        
        # 이전 극점 대비 현재 극점의 가격 변화 방향 확인
        price_change = current_price - prev_price
        change_pct = (price_change / prev_price) * 100
        
        # 연속된 극점의 타입이 달라야 유의미한 파동이 됩니다.
        prev_is_peak = prev_idx in peaks_indices
        current_is_peak = current_idx in peaks_indices
        
        if prev_is_peak == current_is_peak:
            # 연속된 국대점/국소점은 무시하거나, 가장 높은/낮은 값만 남겨야 합니다.
            # 여기서는 단순화를 위해 무시합니다.
            continue 
            
        swing_type = "상승 파동 (Trough -> Peak)" if change_pct > 0 else "하락 파동 (Peak -> Trough)"
        
        swing_results.append({
            '시작 시각': df.index[prev_idx].strftime('%Y-%m-%d %H:%M:%S'),
            '종료 시각': df.index[current_idx].strftime('%Y-%m-%d %H:%M:%S'),
            '시작 가격': prev_price,
            '종료 가격': current_price,
            '절대 등락폭 (KRW)': price_change,
            '등락폭 (%)': change_pct,
            '봉 개수': current_idx - prev_idx,
            '파동 타입': swing_type,
        })
        
    return pd.DataFrame(swing_results)

def analyze_upbit_swings():
    """업비트 API에서 데이터를 가져와 극점 기반 파동 분석을 실행합니다."""
    
    logging.info(f"▶️ {MARKET_TICKER} 마켓의 {INTERVAL} 데이터 {COUNT}개를 API로 조회 시도.")
    
    try:
        # 1. 데이터 조회 (API 호출)
        df = pyupbit.get_ohlcv(MARKET_TICKER, interval=INTERVAL, count=COUNT)
        
        if df is None or df.empty:
            logging.error("❌ 데이터 조회에 실패했습니다. 마켓 코드 또는 API 상태를 확인하세요.")
            return

        # 2. 극점 기반 파동 분석 실행
        swing_df = find_extrema_and_calculate_swings(df, MIN_PEAK_DISTANCE)
        
        if swing_df.empty:
            print("\n⚠️ 유의미한 극점 파동을 찾을 수 없습니다. (데이터 부족 또는 변동성 낮음)")
            return

        print(f"\n--- 📈 {MARKET_TICKER} {INTERVAL} 봉 극점 기반 파동(Swing) 분석 결과 (최근 10개) ---")

        # 결과 포맷팅 및 출력
        recent_swings = swing_df.tail(10)
        
        # 가격은 천단위 구분 (소수점 없이)
        pd.options.display.float_format = '{:,.0f}'.format
        print(recent_swings.drop(columns=['절대 등락폭 (KRW)', '등락폭 (%)']).to_string(index=False))

        # 등락폭은 소수점 4자리
        pd.options.display.float_format = '{:,.4f}'.format
        print("\n[등락폭 상세]")
        print(recent_swings[['절대 등락폭 (KRW)', '등락폭 (%)']].to_string(index=False))

        print("-" * 70)
        print(f"전체 파동의 평균 상승/하락폭 (절대값): {swing_df['등락폭 (%)'].abs().mean():.4f} %")

    except Exception as e:
        logging.error(f"❌ 분석 실행 중 오류 발생: {e}")

# --- 메인 실행 ---
if __name__ == "__main__":
    analyze_upbit_swings()