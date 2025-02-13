기존에 GPT2를 사용했기에 GPT2와 경량화 시킨 gemma-2-2b-it를 사용해봄

Quantization를 통해 경량화 한뒤 Lora를 함께 적용함

각 모델 차이가 명확하기 때문에 비교가 의미 없을 수도 있지만 일단 결과를 보고 판단하기로 함
## 비교
![Image](https://github.com/user-attachments/assets/0cfbd339-c754-4ac8-9302-b2664ed42f39)
![Image](https://github.com/user-attachments/assets/387e5227-4e90-47b8-932c-8a7e0eb418bb)
![Image](https://github.com/user-attachments/assets/2f4056ce-ad0a-4e8c-8a4f-c7746a34b189)

두 모델간 차이가 극명하게 갈림

### 공통 질문
```
How does the VIX (Volatility Index) influence trading decisions in the stock market?
VIX(변동성 지수)는 주식 시장의 거래 결정에 어떤 영향을 미칩니까?
```

*GPT2 학습전*
```
평균 답변 시간: 1.182 sec
답안:
How does the VIX impact the trading of stocks and other investments in the stock market?
To develop, understand, and assess VIX fundamentals, we asked for participants from several fields, including:
Market researchers, trade analysts, financial analysts, retail investors, investment planners, retail investors, forecasters, and investment analysts
Market participants from other U.S. states, Canada, and U.K.
Other large U.S. cities and counties
Market participants from around the globe from over 5,000 research, business advisory, market analysis, and technical, scientific, and academic institutions in the United States, Canada and other countries.
Participants are asked to complete a VIX (vix) questionnaire or to provide their current financial details in our market research section.
In addition to the VIX question, participants may report VIX investments to us, including the type of investment, market participation (such as a passive investment) or their share price, as well as their preferred share price (when they have made investments), on their VIX-based compensation. In addition, participants are asked

VIX는 주식 시장에서 주식 및 기타 투자 거래에 어떤 영향을 미칩니까?
VIX 기본 사항을 개발, 이해 및 평가하기 위해 다음을 포함한 여러 분야의 참여자를 요청했습니다.
시장 조사원, 거래 분석가, 재무 분석가, 소매 투자자, 투자 기획자, 소매 투자자, 예측자 및 투자 분석가
다른 미국 주, 캐나다 및 영국의 시장 참여자
다른 대규모 미국 도시 및 카운티
미국, 캐나다 및 기타 국가의 5,000개 이상의 연구, 비즈니스 자문, 시장 분석 및 기술, 과학 및 학술 기관의 전 세계 시장 참여자.
참여자는 VIX(vix) 설문지를 작성하거나 시장 조사 섹션에 현재 재무 세부 정보를 제공해야 합니다.
VIX 질문 외에도 참가자는 VIX 기반 보상에 대한 투자 유형, 시장 참여(수동적 투자 등) 또는 주가, 그리고 우선주 가격(투자한 경우)을 포함하여 VIX 투자를 보고할 수 있습니다. 또한 참가자는 다음과 같은 질문을 받습니다.
```

*GPT2 학습후*
```
평균 답변 시간: 0.841 sec
답안:
During early trading volumes, investors can focus on asset performance, historical performance, and overall market conditions that predict the future price of VIX stocks. The VIX is driven by market optimism and sentiment, which can influence the trading decisions of trading traders.
VIX Traders often assess the impact of economic conditions, market sentiment, and economic forecasts. Factors such as GDP growth, the participation of large multinational corporations, and changing investor expectations can influence stock prices.
Historical Trends and Sentiment Factors
Historical trends and sentiment factors are crucial for trading decisions as they predict market trends, market sentiment—and consumer opinions—based on financial information

초기 거래량 동안 투자자는 자산 성과, 과거 성과 및 VIX 주식의 미래 가격을 예측하는 전반적인 시장 상황에 집중할 수 있습니다. VIX는 시장의 낙관주의와 감정에 의해 주도되며, 이는 거래자의 거래 결정에 영향을 미칠 수 있습니다.
VIX 거래자는 종종 경제 상황, 시장 감정 및 경제 예측의 영향을 평가합니다. GDP 성장, 대형 다국적 기업의 참여 및 투자자 기대치의 변화와 같은 요인은 주가에 영향을 미칠 수 있습니다.
과거 추세 및 감정 요인
과거 추세 및 감정 요인은 재무 정보를 기반으로 시장 추세, 시장 감정 및 소비자 의견을 예측하기 때문에 거래 결정에 중요합니다.
```


*gemma-2-2b-it 학습전*
```
평균 답변 시간: 4.768 sec
답안:
The VIX, or the CBOE Volatility Index, is a measure of the market's expectation of future volatility. It's often referred to as the "fear index" because it tends to rise when investors are worried about the market's direction. 
Here's how the VIX influences trading decisions:
**1. Reflecting Market Sentiment:**
* **High VIX:** A high VIX indicates high market volatility and fear. This often leads to a "flight to safety" as investors seek out less risky assets like bonds or gold. This can cause stock prices to decline as investors become more cautious.
* **Low VIX:** A low VIX suggests low market volatility and calm. This can lead to increased risk-taking as investors feel more confident about the market's direction.
**2. Predicting Volatility:**
* **VIX as a predictor:** The VIX can be used as a predictor of future market volatility. When the VIX is high, it suggests that the market is likely to be more volatile in the near future. Conversely, a low VIX can indicate that the market

VIX 또는 CBOE 변동성 지수는 시장의 미래 변동성 기대를 측정하는 지표입니다. 투자자가 시장 방향에 대해 걱정할 때 상승하는 경향이 있기 때문에 종종 "공포 지수"라고 합니다.

VIX가 거래 결정에 영향을 미치는 방식은 다음과 같습니다.

**1. 시장 심리 반영:**
* **높은 VIX:** 높은 VIX는 높은 시장 변동성과 두려움을 나타냅니다. 이는 종종 투자자들이 채권이나 금과 같은 덜 위험한 자산을 찾으면서 "안전 자산으로의 도피"로 이어집니다. 이는 투자자들이 더 조심스러워짐에 따라 주가가 하락할 수 있습니다.
* **낮은 VIX:** 낮은 VIX는 낮은 시장 변동성과 평온함을 나타냅니다. 이는 투자자가 시장 방향에 대해 더 확신을 갖게 됨에 따라 위험 감수가 증가할 수 있습니다.
**2. 변동성 예측:**
* **예측 지표로서의 VIX:** VIX는 미래 시장 변동성의 예측 지표로 사용할 수 있습니다. VIX가 높으면 가까운 미래에 시장이 더 변동성이 커질 가능성이 있음을 시사합니다. 반대로 VIX가 낮으면 시장이
```


*gemma-2-2b-it 학습후*
```
평균 답변 시간: 2.695 sec
답안:
The VIX measures market expectations of volatility, and traders use it to gauge market sentiment and anticipate potential price movements.

VIX는 시장의 변동성 예측을 측정하며, 트레이더는 이를 사용하여 시장 심리를 파악하고 잠재적인 가격 움직임을 예상합니다.
```


## 정리
두 모델의 명확한 차이가 존재하지만

GPT2는 학습 후 데이터 품질이 늘어난게 보였으며

Gemma는 경량화되고 학습된 후 답변 시간이 빨라지고 답안이 최적화 되었다고 느껴짐

주어진 corpus는 답변이 짦게 최적화된 상태로 학습에 대한 반영율이 좋다고 생각이 듬
