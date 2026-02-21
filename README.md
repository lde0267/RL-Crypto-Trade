# Reinforce Learning based bitcoin autotrade machine

## Overview
본 프로젝트는 변동성이 큰 암호화폐 시장에서 안정적인 수익 모델을 창출하기 위해 데이터 사이언스 기반의 피처 엔지니어링과 **심층 강화학습(Deep Reinforcement Learning)**을 결합한 자율 매매 시스템 개발을 목표로 합니다. 단순히 알고리즘을 적용하는 것에 그치지 않고, 데이터의 정제부터 모델 학습, 그리고 실전 거래 로직 구현에 이르기까지 총 3단계의 체계적인 파이프라인을 구축하였습니다.

피처 추출 => 강화학습 => 모델 생성 및 백테스트 => 실전매매

## Result
2025년 8월, 9월, 10월 대상으로 백테스트 진행하였습니다. 매수 신호는 모델이 생성하고 매도는 룰 기반으로 하였습니다. 매도 신호도 모델이 생성하는 학습방법도 코드에 포함되어 있으나 성과가 좋지 못해 정확한 매수 신호 생성에 집중하였습니다.

비트코인 buy & hold를 벤치마크로 그 수익률을 비교합니다. 매수와 매도 시점은 그래프에 표시되어있습니다.

1. Simple Trailing Stop
에이전트가 매수 결정을 내린 후, 고점 대비 일정 비율(0.1% ~ 0.7%) 하락 시 기계적으로 매도하는 방식
<img width="1800" height="1200" alt="rule_2025_08to09" src="https://github.com/user-attachments/assets/2d18b661-be77-4971-9c36-f9b87e57af5b" />
<img width="1800" height="1200" alt="rule_2025_09to10" src="https://github.com/user-attachments/assets/d28c81b1-a259-4a8b-a307-33065c6f13d5" />
<img width="1800" height="1200" alt="rule_2025_10to11" src="https://github.com/user-attachments/assets/c7044a72-4911-4a8f-ba0f-47f5c6ad1073" />

2. Optimized Stop-Loss & Take-Profit
<img width="1800" height="1200" alt="rule2_2025_08to09" src="https://github.com/user-attachments/assets/8e003a31-9d37-43e5-a14b-a4f955e7ffcf" />
<img width="1800" height="1200" alt="rule2_2025_09to10" src="https://github.com/user-attachments/assets/cc439ea7-2171-49a5-b2d7-3fa9dc1f9e01" />
<img width="1800" height="1200" alt="rule2_2025_10to11" src="https://github.com/user-attachments/assets/a2d46abd-6fc0-496d-b745-6b30a8912b43" />

## Method Pipline
### Phase 1: Advanced Data Preprocessing & Multi-Step Feature Selection
학습 데이터의 품질을 높이기 위해 단순히 과거 가격만을 사용하는 것이 아니라, 금융 시계열 분석에서 널리 사용되는 트리플 배리어(Triple Barrier Method) 기법을 도입하여 각 지표와 실제 수익 간의 관련성을 정밀하게 설정하였습니다.

Labeling with Triple Barrier Method: 각 데이터 포인트에 대해 미래 성과를 평가하는 세 가지 기준을 설정하여 레이블링을 수행했습니다.
  - Profit Target : 상단 배리어 도달 시 익절 성공으로 간주.
  - Stop Loss : 하단 배리어 도달 시 리스크 관리 실패로 간주.
  - Time Barrier (48 Periods / 24 Hours): 30분봉 기준 24시간 내에 상/하단 배리어에 도달하지 못할 경우 시간 제한에 의한 강제 종료.

Two-Stage Feature Selection Process:
1. Non-linear Importance Analysis
생성된 수많은 기술적 지표 중, 트리플 배리어 레이블과 가장 연관성이 높은 변수들을 선별하기 위해 Random Forest와 XGBoost 모델을 활용하였습니다. 이를 통해 데이터의 비선형적 관계를 파악하고 상위 20개의 핵심 지표를 1차적으로 선별했습니다.

3. Lasso-based Core Feature Extraction
선별된 20개 지표를 대상으로 Lasso(L1 Regularization) 회귀 모델을 적용했습니다. 이 과정에서 변수 간의 다중공선성을 억제하고, 앞서 설정한 익절 타겟(+1%) 달성과 가장 상관관계가 높은 최종 6개의 핵심 지표를 도출하여 모델의 예측 성능과 연산 효율성을 동시에 확보했습니다.

## Phase 2: Deep Reinforcement Learning with Recurrent PPO
선별된 핵심 지표들을 바탕으로 시계열 데이터의 패턴을 학습하고, 최적의 매매 타이밍을 결정하는 고도화된 강화학습 모델을 설계하였습니다. 단순히 현재의 상태만을 보고 판단하는 것이 아니라, 과거의 흐름을 기억하여 의사결정에 반영하는 아키텍처를 구축했습니다.

1. Algorithm & Architecture: Recurrent PPO (LSTM)
- Sequential Memory: 암호화폐 시장의 비정형적인 패턴을 파악하기 위해 LSTM(Long Short-Term Memory) 레이어가 통합된 Recurrent PPO(Proximal Policy Optimization) 알고리즘을 채택하였습니다. 이는 윈도우 사이즈(Window Size) 내의 시계열적 종속성을 효과적으로 학습하여 에이전트가 시장의 추세를 더 정밀하게 인지하도록 돕습니다.

- Policy & Value Network: 공유된 LSTM 추출기(Feature Extractor)를 기반으로 행동을 결정하는 **Actor(Policy)**와 현재 상태의 가치를 평가하는 Critic(Value) 네트워크가 상호작용하며 학습의 안정성을 높입니다.

2. Environment Design & Observation Space
- Multi-dimensional Observation: Lasso를 통해 선별된 6~9개의 핵심 기술 지표뿐만 아니라, 에이전트의 현재 상태를 나타내는 **포트폴리오 정보(Portfolio Info)**를 결합하였습니다.

- Portfolio Features: 현재 잔고 비율, 코인 보유 여부, 미실현 손익(Unrealized PnL), 그리고 **보유 시간 진행률(Holding Time Ratio)**을 포함하여 에이전트가 리스크 상황을 스스로 인지하도록 설계했습니다.

- Action Space: 시장 상황에 따라 매수(Buy), 매도(Sell), 관망(Hold) 중 최적의 행동을 선택하는 Discrete Action Space를 구성하였습니다.

3. Advanced Reward Shaping (Dense Reward)
- 단순히 매도 시점의 수익만을 보상으로 주는 희소 보상(Sparse Reward)의 문제를 해결하기 위해, 학습 효율을 극대화하는 Dense Reward Shaping 기법을 적용하였습니다.

- Step-wise Reward: 코인을 보유하고 있는 동안 가격 변동률(Price Change %)을 실시간 보상으로 제공하여, 에이전트가 미실현 손익의 변화에 민감하게 반응하도록 유도했습니다.

- Penalty Logic: 미보유 상태에서 가격이 상승할 경우 기회비용에 대한 페널티를 부여하고, 강제 청산(24시간 초과) 발생 시 강력한 페널티를 주어 효율적인 자산 회전율을 학습시켰습니다.

- Realized PnL Bonus: 최종 매도 시 실현된 손익에 대해 Reward Scaling과 Profit Bonus/Loss Penalty를 차등 부여하여 장기적인 기대 수익을 극대화했습니다.

## Phase 3: Real-time Trading System & Multi-layered Monitoring
학습된 모델을 실제 시장에 적용하기 위해 업비트(Upbit) API를 연동한 실시간 자동매매 시스템을 구축하였습니다. 단순한 매수/매도를 넘어 안정적인 운영을 위한 멀티스레딩 아키텍처와 리스크 관리 레이어를 포함하고 있습니다.

1. High-Performance Trading Pipeline
- Upbit API Integration: upbit_client.py를 통해 실시간 호가 및 잔고 데이터를 수신하며, 시장가 주문 및 미체결 주문 관리 기능을 모듈화하여 안정적인 인터페이스를 구축했습니다.

- Real-time Data Processing: 30분봉 캔들이 생성될 때마다 rl_data_processor.py가 즉각적으로 Lasso 기반 핵심 지표를 계산하고 정규화를 수행하여 모델에 입력 가능한 Observation 형태로 변환합니다.

- Signal Generation: rl_signal.py에서 메모리에 로드된 Recurrent PPO 모델이 실시간 데이터를 해석하여 최적의 액션(Buy/Hold)을 결정합니다.

2. Execution & Risk Management Strategy
- Scaling-in (분할 매수): 에이전트의 진입 신호 발생 시, strategy_trade.py에 정의된 전략에 따라 자산을 분할 집행하여 평균 단가를 관리합니다.

- Rule-based Exit Layer: RL 에이전트의 판단과 별개로, 자산 보호를 위해 하드코딩된 Stop Loss(-3.2%) 및 Trailing Stop 로직이 상시 가동되어 급격한 변동성에 대응합니다.

- Cooldown System: 불필요한 잦은 거래를 방지하기 위해 매도 후 일정 시간(30분봉 1틱 분량) 동안 거래를 제한하는 쿨다운 로직을 적용하여 심리적/비용적 효율성을 높였습니다.

3. System Architecture & Monitoring
- Multi-threaded Execution: main.py를 중심으로 매매 로직을 수행하는 Trader Thread와 사용자 소통을 담당하는 Telegram Bot Thread를 분리 운영하여 시스템의 가용성을 확보했습니다.

- Interactive Telegram Notifier: notifier.py를 통해 모든 체결 결과, 손익 리포트, 시스템 상태를 실시간으로 전송받습니다.

- Real-time Alert: 매수/매도 발생 시 즉시 알림 전송.

- Status Command: 사용자가 /status 명령어를 통해 언제 어디서든 현재 포트폴리오의 성과와 미실현 손익을 확인할 수 있는 인터랙티브 기능을 제공합니다.

## 사용 방법
본 프로젝트는 데이터 수집부터 실전 매매까지 단계별 파이프라인으로 구성되어 있습니다. 아래 순서에 따라 실행해 주세요.

Step 1. 데이터 수집 및 전처리 (Preprocessing)
2_preprocess 폴더 내의 스크립트를 실행하여 학습에 필요한 피처를 생성합니다.
실행: 전처리 스크립트 실행.
결과: Upbit API 기반의 원본 데이터와 Triple Barrier/Lasso 기법으로 추출된 지표 기반 정제 데이터가 0_data/ 폴더에 자동으로 저장됩니다.
(지표의 통계지표를 확인하고자 하면 99_lab폴더에 관련 파일이 있습니다.)

Step 2. 에이전트 학습 (Training)
3_train 폴더에서 분석 목적에 맞는 환경을 선택하여 강화학습을 진행합니다.
train_rule.py: 매수 타이밍 학습에 집중하며, 매도는 고정된 규칙(Rule)을 따르는 모델을 학습합니다.
train_auto.py: 매수와 매도 타이밍을 모두 에이전트가 결정하는 완전 자율 매매(Autonomous) 모델을 학습합니다.
결과: 학습이 완료된 최적의 가중치 파일은 1_model/ 폴더에 보관됩니다.

Step 3. 성과 검증 및 분석 (Evaluation)
4_evaluate 폴더의 스크립트를 활용해 과거 데이터에 대한 모델의 성과를 객관적으로 평가합니다.
실행: backtest.py 또는 backtest2.py 실행.
결과: 2025년 8~10월 실증 데이터를 바탕으로 한 Sharpe Ratio, MDD, 누적 수익률 지표와 시각화 리포트가 생성됩니다.

Step 4. 실전 자동매매 실행 (Live Trading)
5_trade 폴더를 통해 실제 암호화폐 시장에 모델을 투입합니다.
설정: config.py 파일에 본인의 Upbit API Key를 입력하고, 매수 비중 등 트레이딩 파라미터를 설정합니다.
실행: main.py를 실행하여 실시간 데이터 수집 - 신호 생성 - 주문 집행 시스템을 가동합니다.
모니터링: 텔레그램을 통해 실시간 체결 알림 및 포트폴리오 상태를 수신합니다.

## 개발자의 말
프로젝트는 정교한 피처 엔지니어링과 심층 강화학습을 통해 시장의 미세한 패턴을 학습하고 자동화된 수익 모델을 구축하려는 시도였습니다. 이 프레임워크는 설계할 때는 거시적인 흐름과 관계없이 발생하는 미시적인 파동에서 수익을 얻고자 하였습니다. 결국 하지만 수많은 백테스트와 실전 매매를 거치며 내린 결론은 결국 **'모든 지표는 거시경제(Macroeconomics)의 흐름 아래에 있다'**는 점입니다.

아무리 고도화된 LSTM 레이어가 시계열 패턴을 파악하고 Lasso 회귀가 핵심 지표를 골라내더라도, 글로벌 금리 결정, 유동성 변화, 혹은 지정학적 위기와 같은 거시적 변수가 시장을 뒤흔들 때 기술적 지표들은 그 힘을 잃었습니다. 에이전트가 학습한 '패턴'은 결국 거시경제라는 거대한 바다 위에서 치는 '파도'에 불과할지도 모른다는 사실을 깊이 체감했습니다.
