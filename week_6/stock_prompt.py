
import pandas as pd
def normal_prompt_text(combined_stocks_data):
 
    """
    종목 데이터를 바탕으로 일반적인 투자 분석 프롬프트를 생성
    종목은 5개씩 묶어서 프롬프트를 생성
    간결한 설명을 유지해 토큰 낭비 방지

    """

    stock_symbols = list(combined_stocks_data.keys())
    chunk_size = 5  # 한 번에 처리할 종목 개수
    prompt_list = []  # 최종 프롬프트 리스트

    for i in range(0, len(stock_symbols), chunk_size):
        chunk_symbols = stock_symbols[i : i + chunk_size]

        prompt_text = """
제공된 데이터를 바탕으로 종합적인 투자 분석을 수행하고, 적절한 투자 전략을 제시해주세요.

## 1. 기술적 분석
- 이동평균선 (SMA 20, 50), MACD, ADX를 활용한 추세 분석
- RSI, 스토캐스틱을 통한 과매수/과매도 판단
- ATR, 볼린저 밴드를 이용한 변동성 평가
- 거래량 분석 (OBV, 거래량 이동평균)
- 주요 저항선 및 지지선 분석

## 2. 재무제표 분석
- 손익계산서: 매출, 영업이익, 순이익 변화 분석
- 대차대조표: 자산, 부채, 자본 구조 평가
- 현금흐름: 영업, 투자, 재무활동 흐름 분석

## 3. 핵심 재무 지표
- EPS, ROE, ROA, EBITDA, FCF 분석
- PER, PBR, PSR, EV/EBITDA 평가
- 시가총액 및 기업가치 비교

## 4. 성장성 및 배당 분석
- 매출 및 이익 성장률
- 배당수익률, 배당성향 평가
- 장기 성장 가능성 판단

아래 데이터를 참고하여 분석을 수행하세요.

각 종목에 대해
보유 종목은 매수/매도/홀딩 중 어떤 의사결정을 고려할 수 있을지, 
관심 종목은 매수/홀딩 중 어떤 의사결정을 고려할 수 있을지, 
그리고 그 이유(간단히)를 제시해 주세요.


주의: 이 분석은 단순 참고용이며 실제 투자 결정은 책임질 수 없음을 압니다.
"""

        for symbol in chunk_symbols:
            stock_info = combined_stocks_data[symbol]

            # 최신 종가 가져오기
            latest_close = stock_info['technical_analysis_df'].iloc[-1]['Close']
            
            # 숫자 단위 변환 함수 (가독성 개선)
            def format_large_number(value):
                """큰 숫자를 B(십억) / M(백만) 단위로 변환하고 소수점 2자리로 포맷"""
                if isinstance(value, (int, float)):
                    if abs(value) >= 1_000_000_000:
                        return f"{value / 1_000_000_000:.2f}B"
                    elif abs(value) >= 1_000_000:
                        return f"{value / 1_000_000:.2f}M"
                    else:
                        return f"{value:.2f}"
                return "N/A"
            
            # 재무 데이터 가공
            def format_value(value):
                """숫자 포맷 정리 (소수점 2자리, 단위 변환)"""
                if isinstance(value, (int, float)):
                    return format_large_number(value)
                elif isinstance(value, pd.Series) and not value.empty:
                    
                    return ', '.join([
                        f"({pd.to_datetime(date).strftime('%Y-%m-%d')}: {format_large_number(val)})"
                        if isinstance(val, (int, float)) else f"({pd.to_datetime(date).strftime('%Y-%m-%d')}: N/A)"
                        for date, val in value.tail(5).items()
                    ])
                return "N/A"

            prompt_text += f"""
### {symbol} 종목 분석
#### 1. 기술적 분석
- 종가: {format_value(latest_close)}
- SMA 20: {format_value(stock_info['technical_analysis_df'].iloc[-1]['SMA_20'])}
- SMA 50: {format_value(stock_info['technical_analysis_df'].iloc[-1]['SMA_50'])}
- RSI: {format_value(stock_info['technical_analysis_df'].iloc[-1]['RSI_14'])}
- MACD: {format_value(stock_info['technical_analysis_df'].iloc[-1]['MACD'])} / Signal: {format_value(stock_info['technical_analysis_df'].iloc[-1]['MACD_signal'])}
- 볼린저 밴드: 상단={format_value(stock_info['technical_analysis_df'].iloc[-1]['BB_upper'])}, 하단={format_value(stock_info['technical_analysis_df'].iloc[-1]['BB_lower'])}

#### 2. 재무제표 분석
- 매출: {format_value(stock_info['fundamental_dict']['income_stmt'].get('revenue', 'N/A'))}
- 영업이익: {format_value(stock_info['fundamental_dict']['income_stmt'].get('operating_income', 'N/A'))}
- 총자산: {format_value(stock_info['fundamental_dict']['balance_sheet'].get('total_assets', 'N/A'))}
- 총부채: {format_value(stock_info['fundamental_dict']['balance_sheet'].get('total_liabilities', 'N/A'))}
- 영업현금흐름: {format_value(stock_info['fundamental_dict']['cash_flow'].get('operating_cash_flow', 'N/A'))}
- 투자현금흐름: {format_value(stock_info['fundamental_dict']['cash_flow'].get('investing_cash_flow', 'N/A'))}

#### 3. 핵심 재무 지표
- EPS: {format_value(stock_info['fundamental_dict']['key_metrics'].get('EPS', 'N/A'))}
- ROE: {format_value(stock_info['fundamental_dict']['key_metrics'].get('ROE', 'N/A'))}
- PER: {format_value(stock_info['fundamental_dict']['stock_metrics'].get('PER', 'N/A'))}
- PBR: {format_value(stock_info['fundamental_dict']['stock_metrics'].get('PBR', 'N/A'))}
- EV/EBITDA: {format_value(stock_info['fundamental_dict']['stock_metrics'].get('EV/EBITDA', 'N/A'))}

#### 4. 성장성 및 배당
- 매출 성장률: {format_value(stock_info['fundamental_dict']['growth_and_dividend'].get('revenue_growth', 'N/A'))}
- EPS 성장률: {format_value(stock_info['fundamental_dict']['growth_and_dividend'].get('EPS_growth', 'N/A'))}
- 배당수익률: {format_value(stock_info['fundamental_dict']['growth_and_dividend'].get('dividend_yield', 'N/A'))}
- 배당성향: {format_value(stock_info['fundamental_dict']['growth_and_dividend'].get('payout_ratio', 'N/A'))}
"""

            # 보유 종목이라면 손익 계산 추가
            if stock_info['is_holding']:
                pnl = latest_close - stock_info['holding_info']['price']
                pnl_rate = (pnl / stock_info['holding_info']['price']) * 100
                prompt_text += f"""
#### 보유 종목 정보
- 보유 수량: {stock_info['holding_info']['quantity']}주
- 매수 단가: {format_value(stock_info['holding_info']['price'])}
- 현재 손익: {format_value(pnl)} ({format_value(pnl_rate)}%)
"""

        prompt_list.append(prompt_text)

    return prompt_list


def normal_system_text():
    return "당신은 유능한 금융 애널리스트입니다. 사용자에게 투자 조언을 제공해주세요."

def warrenBuffett_system_text():
    return """당신은 위대한 투자가 워렌 버핏입니다. 가치 투자 원칙에 맞추어 투자 조언을 제공해주세요
    저평가된 우량주 찾기
    장기 투자
    기업의 펀더멘털 분석 (수익성, 부채비율, ROE 등)
    """

def peterLynch_system_text():
    return """당신은 위대한 투자가 피터 린치입니다. 성장 투자 원칙에 맞추어 투자 조언을 제공해주세요
    고성장 기업 분석
    제품이나 서비스가 대중적으로 인정받는 기업
    PER이 지나치게 높지 않은지 점검
    """

def benjaminGraham_system_text():
    return """당신은 위대한 투자가 벤저민 그레이엄입니다. 보수적 가치 투자 원칙에 맞추어 투자 조언을 제공해주세요
    주가순자산비율(PBR)이 낮은 기업 분석
    재무제표 안정성 점검
    안전 마진 확보
    """

def rayDalio_system_text():
    return """당신은 위대한 투자가 레이 달리오입니다. 올웨더 포트폴리오 원칙에 맞추어 투자 조언을 제공해주세요
    주식, 채권, 금, 원자재 등 자산 배분 전략
    거시 경제 지표 반영
    경기 변동에 따른 리 밸런싱 고려
    """
