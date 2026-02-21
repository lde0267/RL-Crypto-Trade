import threading
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from datetime import datetime
import requests
import asyncio 
import asyncio

# --- 설정 ---
MY_BOT_TOKEN = "Your Telegram Bot Token"
MY_CHAT_ID = "Your Telegram Chat ID"

# --- 1. 단순 메시지 전송 함수 (trader.py에서 호출용) ---
def send_message(message):
    """단순 텍스트 메시지를 텔레그램으로 전송하는 함수"""
    # 이 함수는 스레드 충돌을 방지하기 위해 간단한 requests를 사용합니다.
    url = f"https://api.telegram.org/bot{MY_BOT_TOKEN}/sendMessage"
    params = {'chat_id': MY_CHAT_ID, 'text': message}
    try:
        requests.get(url, params=params, timeout=5)
    except requests.exceptions.RequestException as e:
        print(f"🚨 텔레그램 메시지 전송 실패: {e}")


# --- 2. 텔레그램 봇 커맨드 핸들러 함수 ---
async def _start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/start 커맨드 핸들러: 봇 시작 인사"""
    user = update.effective_user
    await update.message.reply_html(
        rf"👋 안녕하세요, {user.mention_html()}님!",
        reply_markup=None,
    )
    await update.message.reply_text(
        "자동매매 봇 도우미입니다.\n"
        "오늘의 거래 현황이 궁금하시면 /status 를 입력하세요."
    )

async def _status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/status 커맨드 핸들러: 오늘의 거래 성과 보고"""
    await update.message.reply_text("📈 오늘의 거래 성과를 분석 중입니다. 잠시만 기다려주세요...")
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # --- [수정된 부분] ---
        # db_handler의 동기 함수를 별도 스레드에서 실행하여 봇의 작동을 막지 않게 합니다.
        report = await asyncio.to_thread(
            start_date=today_str, 
            end_date=today_str
        )
        # --------------------
        
        await update.message.reply_text(report)

    except Exception as e:
        # DB 분석 중 에러가 발생하면 사용자에게 알리고 로그를 남깁니다.
        print(f"🚨 /status 처리 중 DB 분석 오류: {e}")
        await update.message.reply_text("⚠️ 데이터를 분석하는 중 오류가 발생했습니다.")


# --- 3. 봇을 시작하는 메인 함수 ---
def start_bot():
    """텔레그램 봇을 시작하고 메시지 폴링을 시작하는 함수"""
    
    # --- 2. 아래 두 줄을 추가하세요 ---
    # 이 스레드를 위한 새로운 asyncio 이벤트 루프를 생성하고 설정합니다.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # ------------------------------------
    
    print("🤖 텔레그램 봇을 시작합니다...")
    
    application = Application.builder().token(MY_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", _start_command))
    application.add_handler(CommandHandler("status", _status_command))

    application.run_polling()