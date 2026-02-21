import pyupbit
import time
import logging

# --- 로깅 설정 ---
# 로그를 더 명확하게 보기 위해 포맷을 지정합니다.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

class UpbitClient:
    def __init__(self, access_key, secret_key):
        """ UpbitClient 초기화 및 로그인 """
        self.upbit = None
        try:
            self.upbit = pyupbit.Upbit(access_key, secret_key)
            krw_balance = self.get_balance("KRW")
            logging.info(f"✅ Upbit 로그인 성공. KRW 잔고: {krw_balance:,.0f} 원")
        except Exception as e:
            logging.error(f"❌ Upbit 로그인 실패: {e}")

    def _is_ready(self):
        """ API 호출 전 로그인 상태 확인 """
        if self.upbit is None:
            logging.error("❌ 클라이언트가 로그인되지 않았습니다.")
            return False
        return True

    def get_tickers(self, fiat="KRW"):
        try:
            return pyupbit.get_tickers(fiat=fiat)
        except Exception as e:
            logging.error(f"❌ 티커 목록 조회 실패: {e}")
            return []

    def get_ohlcv(self, ticker, interval='minute15', count=200):
        if not self._is_ready(): return None
        try:
            time.sleep(0.1)  # API 요청 제한 방지
            return pyupbit.get_ohlcv(ticker, interval=interval, count=count)
        except Exception as e:
            logging.error(f"❌ OHLCV 데이터 조회 실패 ({ticker}): {e}")
            return None

    def get_balance(self, ticker="KRW"):
        """ 특정 자산의 보유 수량을 조회합니다. (예: "KRW", "BTC") """
        if not self._is_ready(): return 0
        try:
            # get_balance는 "KRW-BTC"가 아닌 "BTC"와 같은 심볼을 사용합니다.
            balance = self.upbit.get_balance(ticker)
            return balance if balance else 0
        except Exception as e:
            logging.error(f"❌ 잔고 조회 실패 ({ticker}): {e}")
            return 0

    def get_current_price(self, ticker):
        try:
            return pyupbit.get_current_price(ticker)
        except Exception as e:
            logging.error(f"❌ 현재가 조회 실패 ({ticker}): {e}")
            return None

    def buy_limit_order(self, ticker, price, volume):
        if not self._is_ready(): return None
        try:
            logging.info(f"▶️ 지정가 매수 주문 시도 - 티커: {ticker}, 가격: {price}, 수량: {volume}")
            return self.upbit.buy_limit_order(ticker, price, volume)
        except Exception as e:
            logging.error(f"❌ 지정가 매수 주문 실패 ({ticker}): {e}")
            return None

    # --- 👇 [추가된 함수] 시장가 매수 ---
    def buy_market_order(self, ticker, price):
        """
        시장가로 코인을 매수합니다.
        :param ticker: 마켓 티커 (예: "KRW-BTC")
        :param price: 매수할 총 금액 (KRW)
        :return: 주문 결과 딕셔너리 또는 None
        """
        if not self._is_ready(): return None
        try:
            krw_balance = self.get_balance("KRW")
            if krw_balance < price:
                logging.warning(f"⚠️ 주문 실패: 주문액({price:,.0f}원)이 KRW 잔고({krw_balance:,.0f}원)를 초과합니다.")
                return None
            
            logging.info(f"▶️ 시장가 매수 주문 시도 - 티커: {ticker}, 주문액: {price:,.0f}원")
            result = self.upbit.buy_market_order(ticker, price)
            logging.info(f"✅ 시장가 매수 주문 성공: {result}")
            return result
        except Exception as e:
            logging.error(f"❌ 시장가 매수 주문 실패 ({ticker}): {e}")
            return None

    # --- 👇 [수정된 함수] 시장가 매도 (전량 매도) ---
    def sell_market_order(self, ticker):
        """
        보유한 코인 전량을 시장가로 매도합니다.
        :param ticker: 매도할 마켓 티커 (예: "KRW-BTC")
        :return: 주문 결과 딕셔너리 또는 None
        """
        if not self._is_ready(): return None
        try:
            # "KRW-BTC"에서 코인 심볼("BTC")만 추출합니다.
            coin_symbol = ticker.split('-')[1]
            
            # 해당 코인의 보유 수량을 조회합니다.
            volume = self.get_balance(ticker=coin_symbol)

            # 보유 수량이 0보다 큰지 확인 (최소 주문 수량 등 정책은 pyupbit이 처리)
            if volume <= 0:
                logging.warning(f"⚠️ 주문 실패: 매도할 {coin_symbol} 코인이 없습니다. (보유 수량: {volume})")
                return None

            logging.info(f"▶️ 시장가 매도 주문 시도 - 티커: {ticker}, 매도 수량 (전량): {volume}")
            result = self.upbit.sell_market_order(ticker, volume)
            logging.info(f"✅ 시장가 매도 주문 성공: {result}")
            return result
        except Exception as e:
            logging.error(f"❌ 시장가 매도 주문 실패 ({ticker}): {e}")
            return None
            
    def get_order(self, uuid):
        if not self._is_ready(): return None
        try:
            return self.upbit.get_order(uuid)
        except Exception as e:
            logging.error(f"❌ 주문 상세 정보 조회 실패 (uuid: {uuid}): {e}")
            return None
        
    # upbit_client.py 내 UpbitClient 클래스 안에 추가

    def cancel_open_orders(self, ticker):
        """
        특정 마켓의 모든 미체결 주문을 취소합니다.
        :param ticker: 마켓 티커 (예: "KRW-BTC")
        :return: 취소 성공 여부 (True/False)
        """
        if not self._is_ready():
            return False
        
        try:
            # state='wait'는 미체결 주문만 가져옵니다.
            open_orders = self.upbit.get_order(ticker, state='wait')
            if not open_orders:
                # logging.info(f"[{ticker}] 취소할 미체결 주문이 없습니다.")
                return True

            for order in open_orders:
                uuid = order.get('uuid')
                if uuid:
                    self.upbit.cancel_order(uuid)
                    logging.info(f"✅ [{ticker}] 미체결 주문 취소 완료 (UUID: {uuid})")
            return True
        except Exception as e:
            logging.error(f"❌ [{ticker}] 미체결 주문 취소 중 오류 발생: {e}")
            return False


# --- 클래스 사용 예시 ---
if __name__ == '__main__':
    # ⚠️ 실제 키를 입력하세요. git 등에 올리지 않도록 주의하세요!
    ACCESS_KEY = "YOUR_ACCESS_KEY"
    SECRET_KEY = "YOUR_SECRET_KEY"

    # 1. 클라이언트 객체 생성 (생성과 동시에 로그인)
    client = UpbitClient(ACCESS_KEY, SECRET_KEY)

    # 2. 로그인이 성공했을 경우에만 아래 로직 실행
    if client.upbit:
        # --- 기능 테스트 ---
        
        # [테스트 1] 비트코인 10,000원어치 시장가 매수
        # client.buy_market_order("KRW-BTC", 10000)

        # [테스트 2] 보유한 비트코인 전량 시장가 매도
        # client.sell_market_order("KRW-BTC")

        # [테스트 3] 현재 내 자산 잔고 조회
        print("\n--- 💰 내 자산 조회 ---")
        my_krw = client.get_balance("KRW")
        my_btc = client.get_balance("BTC")
        print(f"원화 잔고: {my_krw:,.0f} KRW")
        print(f"비트코인 잔고: {my_btc} BTC")