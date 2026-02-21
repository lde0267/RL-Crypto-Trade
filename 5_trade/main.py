# main.py

import threading
import time
import trader
import notifier

if __name__ == "__main__":
    # 1. 트레이더 스레드 생성 및 시작
    # trader.py의 start_trading 함수를 실행할 스레드
    trading_thread = threading.Thread(target=trader.start_trading)
    trading_thread.daemon = True  # 메인 프로그램 종료 시 함께 종료
    
    # 2. 텔레그램 봇 스레드 생성 및 시작
    # notifier.py의 start_bot 함수를 실행할 스레드
    bot_thread = threading.Thread(target=notifier.start_bot)
    bot_thread.daemon = True # 메인 프로그램 종료 시 함께 종료

    # 스레드 실행
    trading_thread.start()
    bot_thread.start()
    
    print("✅ 트레이더와 텔레그램 봇이 성공적으로 시작되었습니다.")
    print("프로그램을 종료하려면 Ctrl+C를 누르세요.")

    # 메인 스레드는 자식 스레드들이 종료될 때까지 대기
    # (daemon=True 이므로 사실상 무한 대기)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n 프로그램을 종료합니다.")