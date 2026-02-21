import pickle
import pandas as pd # pd.Series, pd.DataFrame 확인용

STATS_PATH = "1_model/obs_stats_btc.pkl" # ❗️ 실제 통계 파일 경로

try:
    with open(STATS_PATH, 'rb') as f:
        data = pickle.load(f)

    print(f"--- '{STATS_PATH}' 파일 내용 확인 ---")
    
    print(f"\n[1] 로드된 데이터의 타입(Type):")
    print(type(data))

    print(f"\n[2] 로드된 데이터의 내용(Contents):")
    print(data)

    # 만약 데이터가 딕셔너리(dict)라면, 키(key)들을 출력
    if isinstance(data, dict):
        print(f"\n[3] 딕셔너리 키(Keys):")
        print(data.keys())

except Exception as e:
    print(f"파일 로드 오류: {e}")