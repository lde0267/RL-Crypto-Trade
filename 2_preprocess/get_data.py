import requests
import pandas as pd
import time
import os  # (폴더 생성을 위해 추가)

# 1. 30분봉 데이터 불러오기 (Upbit API)
def get_ohlcv_30min(market="KRW-USDT", total_count=48*360*3): # 360일 * 3년
    """
    Upbit API를 통해 30분봉 OHLCV 데이터를 요청하고 DataFrame으로 반환합니다.
    API 제한(최대 200개)을 피하기 위해 분할 요청합니다.
    """
    url = "https://api.upbit.com/v1/candles/minutes/30"
    all_data = []
    to = None
    
    # API는 최대 200개씩 데이터를 반환합니다.
    rounds = (total_count // 200) + (1 if total_count % 200 != 0 else 0)
    
    print(f"총 {total_count}개 캔들 (30분봉) 수집 시작 (예상 요청 수: {rounds}회)")

    for i in range(rounds):
        # 마지막 요청일 경우 남은 개수만 요청
        count = 200
        if i == rounds - 1 and total_count % 200 != 0:
            count = total_count % 200
            
        params = {"market": market, "count": count}
        if to:
            params["to"] = to
            
        headers = {"Accept": "application/json"}
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()  # 200 OK가 아니면 에러 발생
            
            data = response.json()
            
            if not data:
                print("더 이상 데이터가 없습니다. 수집을 중단합니다.")
                break
                
            all_data.extend(data)
            
            # 다음 요청을 위해 마지막 캔들의 시간(UTC)을 저장
            to = data[-1]['candle_date_time_utc']
            
            print(f"[30분봉] 요청 {i+1}/{rounds}, 수집 {len(data)}개, 누적 {len(all_data)}개")
            
            # API 제한을 피하기 위해 잠시 대기
            time.sleep(0.2) 

        except requests.exceptions.RequestException as e:
            print(f"API 요청 중 에러 발생: {e}")
            break
            
    if not all_data:
        print("데이터를 전혀 수집하지 못했습니다.")
        return None

    # DataFrame으로 변환 및 컬럼명 정리
    df = pd.DataFrame(all_data)
    df['datetime'] = pd.to_datetime(df['candle_date_time_utc'])
    
    # 필요한 컬럼만 선택 (사용자가 요청한 원본 ohlcv 형태)
    df = df[['datetime', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']]
    df.columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # 시간순으로 정렬 (API가 최신순으로 반환하므로)
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

# ---------------------------
# 실행부
# ---------------------------

# 데이터를 저장할 폴더 이름
output_folder = "0_data"
# 저장할 파일 이름
output_filename = os.path.join(output_folder, "btc_ohlcv_30min.csv")

# data 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"'{output_folder}' 폴더를 생성했습니다.")

# 30분봉 데이터 가져오기 (비트코인, 약 8500개)
df_30 = get_ohlcv_30min(market="KRW-BTC", total_count=48*360*3) # 360일치

if df_30 is not None:
    # 결과 CSV 파일로 저장
    df_30.to_csv(output_filename, index=False)
    print(f"\n✅ 30분봉 원본 데이터 저장 완료! ({len(df_30)}개 행)")
    print(f"   -> {output_filename}")
else:
    print("\n❌ 데이터 저장 실패.")