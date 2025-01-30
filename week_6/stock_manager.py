import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import json
from datetime import datetime
from streamlit_cookies_manager import EncryptedCookieManager
from openai import OpenAI # GPT-4 API
import stock_prompt
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
# 페이지 네비게이션 초기화
if "page" not in st.session_state:
    st.session_state["page"] = "select_stocks"  # 기본 페이지


# API 키 입력
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요:", 
                               value=st.session_state.api_key,
                               type="password")

if api_key:
    st.session_state.api_key = api_key

if not st.session_state.api_key:
    st.warning("사이드바에 OpenAI API 키를 입력해주세요.")
    st.stop()

# S&P 500 리스트 크롤링 함수
@st.cache_data
def load_sp500_symbols():
    """
    위키백과 페이지에서 S&P500 리스트를 가져온 뒤
    심볼(Symbol) 컬럼만 추출하여 리스트로 반환
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    # Symbol과 Security(회사명) 컬럼 추출
    return df[['Symbol', 'Security']].values.tolist()

# 선택된 종목에 대해 OHLCV 데이터를 가져오고 기술적 지표를 계산
@st.cache_data
def fetch_technical_analyze(symbol):
    """
    종목에 대해 OHLCV 데이터를 가져오고 기술적 지표를 계산
    :param symbol: 종목 심볼
    :return: 기술적 지표를 포함한 데이터프레임 딕셔너리
    """
    try:
        data = yf.download(symbol, period="6mo", interval="1d")
        data.dropna(inplace=True)
        # 멀티 인덱스인 경우 제거
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # 기술적 지표 계산
        data['SMA_20'] = ta.sma(data['Close'], length=20)
        data['SMA_50'] = ta.sma(data['Close'], length=50)
        macd = ta.macd(data["Close"], fast=12, slow=26, signal=9)

        data["MACD"] = macd["MACD_12_26_9"]
        data["MACD_signal"] = macd["MACDs_12_26_9"]
        data["MACD_hist"] = macd["MACDh_12_26_9"]
        
        data['ADX'] = ta.adx(data['High'], data['Low'], data['Close'])['ADX_14']
        data['RSI_14'] = ta.rsi(data['Close'], length=14)
        stoch = ta.stoch(data['High'], data['Low'], data['Close'])
        data['Stoch_%K'] = stoch['STOCHk_14_3_3']
        data['Stoch_%D'] = stoch['STOCHd_14_3_3']
        data['CCI'] = ta.cci(data['High'], data['Low'], data['Close'])
        bollinger = ta.bbands(data['Close'], length=20, std=2)
        data['BB_upper'] = bollinger['BBU_20_2.0']
        data['BB_middle'] = bollinger['BBM_20_2.0']
        data['BB_lower'] = bollinger['BBL_20_2.0']
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'])
        data['OBV'] = ta.obv(data['Close'], data['Volume'])
        data['Volume_MA_20'] = ta.sma(data['Volume'], length=20)

        ichimoku_base, ichimoku_lead = ta.ichimoku(data['High'], data['Low'], data['Close'])

        data['Ichimoku_base'] = ichimoku_base['IKS_26']
        data['Ichimoku_conversion'] = ichimoku_base['ITS_9']
        data['Ichimoku_lead1'] = ichimoku_lead['ISA_9']
        data['Ichimoku_lead2'] = ichimoku_lead['ISB_26']
        
        data['Momentum'] = ta.mom(data['Close'], length=10)
        
        data['Williams_%R'] = ta.willr(data['High'], data['Low'], data['Close'])


        # 결과 저장
        return data
    except Exception as e:
        st.error(f"{symbol} 데이터를 분석하는 중 오류 발생: {e}")
        return None


def fetch_fundamental_dict(symbol):
    """
    주식 심볼에 대한 기본적 분석 데이터를 수집
    :param symbol: 주식 심볼
    :return: 재무 데이터 딕셔너리
    """
    try:
        ticker = yf.Ticker(symbol)

        # 손익계산서 (Income Statement)
        income_stmt = ticker.financials
        if income_stmt is not None and not income_stmt.empty:
            income_stmt = income_stmt.T
            revenue = income_stmt.get('Total Revenue', None)
            operating_income = income_stmt.get('Operating Income', None)
            net_income = income_stmt.get('Net Income', None)
        else:
            revenue = operating_income = net_income = None

        income_stmt_metrics = {
            "revenue": revenue,
            "operating_income": operating_income,
            "net_income": net_income,
        }

        # 대차대조표 (Balance Sheet)
        balance_sheet = ticker.balance_sheet
        if balance_sheet is not None and not balance_sheet.empty:
            balance_sheet = balance_sheet.T
            total_assets = balance_sheet.get('Total Assets', None)
            total_liabilities = balance_sheet.get('Total Liabilities Net Minority Interest', None)
            shareholders_equity = balance_sheet.get('Total Equity Gross Minority Interest', None)
        else:
            total_assets = total_liabilities = shareholders_equity = None

        balance_sheet_metrics = {
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "shareholders_equity": shareholders_equity,
        }

        # 현금흐름표 (Cash Flow Statement)
        cash_flow = ticker.cashflow
        if cash_flow is not None and not cash_flow.empty:
            cash_flow = cash_flow.T
            operating_cf = cash_flow.get('Operating Cash Flow', None)
            investing_cf = cash_flow.get('Investing Cash Flow', None)
            financing_cf = cash_flow.get('Financing Cash Flow', None)
        else:
            operating_cf = investing_cf = financing_cf = None
        cash_flow_metrics = {
            "operating_cash_flow": operating_cf,
            "investing_cash_flow": investing_cf,
            "financing_cash_flow": financing_cf,
        }

        # 주요 재무 지표
        key_metrics = {
            "EPS": ticker.info.get("trailingEps"),
            "DPS": ticker.info.get("dividendRate"),
            "ROE": ticker.info.get("returnOnEquity"),
            "ROA": ticker.info.get("returnOnAssets"),
            "EBITDA": ticker.info.get("ebitda"),
            "FCF": ticker.info.get("freeCashflow"),
        }

        # 주가 관련 지표
        stock_metrics = {
            "market_cap": ticker.info.get("marketCap"),
            "PER": ticker.info.get("trailingPE"),
            "PBR": ticker.info.get("priceToBook"),
            "PSR": ticker.info.get("priceToSalesTrailing12Months"),
            "EV": ticker.info.get("enterpriseValue"),
            "EV/EBITDA": ticker.info.get("enterpriseToEbitda"),
        }

        # 성장 및 배당 데이터
        growth_and_dividend = {
            "revenue_growth": ticker.info.get("revenueGrowth"),
            "EPS_growth": ticker.info.get("earningsGrowth"),
            "dividend_yield": ticker.info.get("dividendYield"),
            "payout_ratio": ticker.info.get("payoutRatio"),
        }
        return {
            "income_stmt": income_stmt_metrics,
            "balance_sheet": balance_sheet_metrics,
            "cash_flow": cash_flow_metrics,
            "key_metrics": key_metrics,
            "stock_metrics": stock_metrics,
            "growth_and_dividend": growth_and_dividend,
        }

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None
def generate_quant_strategies(technical_analysis_df, fundamental_dict):
    """
    기술적/기본적 분석 데이터를 활용하여
    검증된 투자가들의 공식 기준을 생성해 반환
    :param technical_analysis_df: 기술적 분석 데이터 프레임
    :param fundamental_dict: 기본적 분석 데이터 딕셔너리
    :return: 퀀트 투자 전략에 활용할 공식 딕셔너리
    """
    
    income_stmt = fundamental_dict["income_stmt"]
    balance_sheet = fundamental_dict["balance_sheet"]
    cash_flow = fundamental_dict["cash_flow"]
    key_metrics = fundamental_dict["key_metrics"]
    stock_metrics =  fundamental_dict["stock_metrics"]
    growth_dividend = fundamental_dict["growth_and_dividend"]


    # total_current_assets = total_assets 개념이 같은건가? 일단 데이터상 제공되는대로 진행
    total_current_assets = balance_sheet.get("total_assets", None)
    total_liabilities = balance_sheet.get("total_liabilities", None)
    market_cap = stock_metrics.get("market_cap", None)


    # NCAV (Net Current Asset Value) 계산
    # 저평가된 주식을 찾기 위한 전통적인 방법 중 하나
    # 시장이 순유동자산보다 싸게 거래 중인지 판단
    # 
    # NCAV 계산
    if total_current_assets is not None and total_liabilities is not None:
        ncav = total_current_assets - total_liabilities
    else:
        ncav = None

    # NCAV/시가총액 비율
    if ncav is not None and market_cap is not None and market_cap > 0:
        ncav_ratio = ncav / market_cap
    else:
        ncav_ratio = None


    # Magic Formula (Joel Greenblatt)
    # 저평가 우량주를 찾기 위한 전략
    # 공식 쓸꺼면 특정 섹터를 제거하고 최소 시가총액 이상의 종목만 고려해야함
    # 좋은 공식으로 판단되지만, 일단 간단하게만 구현
    # 어차피 나중에 퀀트쪽은 더 개선이 필요함 TODO
    ebit = key_metrics.get("EBITDA", None)  # 근사적으로 EBIT를 EBITDA로 사용
    ev = stock_metrics.get("EV", None)
    net_working_capital = (balance_sheet.get("total_assets", 0) - balance_sheet.get("total_liabilities", 0))
    fixed_assets = balance_sheet.get("shareholders_equity", 0)
    
    # Earnings Yield 계산 (EBIT / EV)
    if ebit and ev and ev > 0:
        earnings_yield = ebit / ev
    else:
        earnings_yield = None
    
    # Return on Capital 계산 (EBIT / (Net Working Capital + Fixed Assets))
    if ebit and (net_working_capital + fixed_assets) > 0:
        return_on_capital = ebit / (net_working_capital + fixed_assets)
    else:
        return_on_capital = None

    # Fama-French 3 공식 TODO
    # 시장의 위험 프리미엄을 고려한 수익률 계산
    # 데이터 부족해서 일단 방식만 명시
    # SP500 지수 데이터, 미국 국채 수익률, SMB, HML 등을 사용해야 함

    # Piotroski F-Score 재무 안전성 평가 TODO
    # 0 ~ 9점까지, 9점은 가장 안전한 종목
    # 스코어 계산을 위해 전년도 재무 평가표와 신규 주식 발행 여부, 매출총이익률증가, 자산회전율을 추가해야함
    # 현재 단계에서는 계산 불가

    # CANSLIM 전략 TODO
    # C: Current Earnings(최근 분기 이익)
    # A: Annual Earnings(연간 이익)
    # N: New Products or Services(신제품, 서비스): 없는데 나중에 감정분석 추가되면 가능
    # S: Supply and Demand(공급과 수요)
    # L: Leader or Laggard(시장 점유율): 없음, 나중에 마법 공식 추가하면 전체 주식 비교도 들어가야하는데 이때 진행
    # I: Institutional Sponsorship(기관 투자): 없음
    # M: Market Direction(시장 방향): Fama-French 3 공식과 연관 있음

    # PEG Ratio
    # PER 대비 EPS 성장률을 고려한 지표
    # 주식이 고평가되었는지 저평가되었는지 판단
    per = fundamental_dict["stock_metrics"].get("PER")  # P/E Ratio
    eps_growth = fundamental_dict["growth_and_dividend"].get("EPS_growth")  # EPS 성장률

    if per and eps_growth and eps_growth > 0:
        peg_ratio = per / (eps_growth * 100)  # EPS 성장률을 %로 사용하므로 *100
        # PEG Ratio가 1보다 작으면 저평가, 1보다 크면 고평가로 판단
    else:
        peg_ratio = None
        neff_score = None
    
    return {
        "NCAV": ncav,
        "NCAV_Ratio": ncav_ratio,
        "Magic_Formula_Earnings_Yield": earnings_yield,
        "Magic_Formula_Return_on_Capital": return_on_capital,
        "PEG_Ratio": peg_ratio,
    }


def generate_quant_features(technical_analysis_df, fundamental_dict):
    """
    기술적/기본적 분석 데이터를 활용하여
    (1) 이벤트/플래그, (2) 추가 파생 지표, (3) 간단한 스코어 등을 생성해 반환합니다.
    
    :param technical_analysis_df: 기술적 분석 데이터 프레임
    :param fundamental_dict: 기본적 분석 데이터 딕셔너리
    :return: 퀀트 투자 전략에 활용할 피처 딕셔너리
    """
    quant_features = {}

    # 기술적 분석: 최근 (가장 마지막 row) 기준으로 시그널 뽑기
    # ====================================================
    latest = technical_analysis_df.iloc[-1]  # 가장 최근 날짜 행
    prev = technical_analysis_df.iloc[-2] if len(technical_analysis_df) > 1 else latest  # 바로 전날 (데이터가 충분히 있다고 가정)

    # RSI 방향성(전일 대비 상승/하락)
    rsi_flag = 1 if latest["RSI_14"] > prev["RSI_14"] else 0

    # MACD > MACD_signal 교차 여부(골든크로스?)
    macd_crossover = 1 if (
        (latest["MACD"] > latest["MACD_signal"]) and
        (prev["MACD"] <= prev["MACD_signal"])
    ) else 0

    # SMA20 vs SMA50 (단기 > 중기)
    sma_trend = 1 if latest["SMA_20"] > latest["SMA_50"] else 0

    # ATR, ADX, Stoch, Bollinger 밴드 이벤트 등은....TODO


    # 기본적 분석: PER, PBR, EV/EBITDA 등으로 Value Score를 간단 계산
    # ====================================================

    income_stmt = fundamental_dict["income_stmt"]
    balance_sheet = fundamental_dict["balance_sheet"]
    cash_flow = fundamental_dict["cash_flow"]
    key_metrics = fundamental_dict["key_metrics"]
    stock_mertrics =  fundamental_dict["stock_metrics"]
    growth_dividend = fundamental_dict["growth_and_dividend"]

    
    per = stock_mertrics.get("PER", None)
    pbr = stock_mertrics.get("PBR", None)
    ev_ebitda = stock_mertrics.get("EV/EBITDA", None)
    
    
    roe = key_metrics.get("ROE", None)

    # Value Score = (PER + PBR + EV/EBITDA)를 단순 합
    # 실제로는 Z-score로 정규화하거나, 결측치 처리 등을 해야한다는데...TODO
    if per and pbr and ev_ebitda and all(x > 0 for x in [per, pbr, ev_ebitda]):
        value_score = per + pbr + ev_ebitda
    else:
        value_score = None

    # ROE가 높을수록, 혹은 FCF가 클수록 점수가 높도록
    # 간단히 "ROE" 자체를 점수화할 수도 있고, 여러 요소 합산 가능
    # 여기서는 ROE가 있으면 ROE 그대로, 없으면 0으로 처리 TODO
    quality_score = roe if roe else 0


    # 퀀트 스코어: 기술적 + 가치 + 퀄리티 통합 TODO
    # ====================================================
    # 단순히 "기술적 시그널 합 + (1 / value_score) + quality_score" 등으로 정리했는데
    # 실제 퀀트 스코어 계산 기준은 다 다르고, 복잡하기에 일단 PASS

    technical_sum = rsi_flag + macd_crossover + sma_trend  # 단순 예시
    final_score = technical_sum + (1 / value_score) + (quality_score if quality_score else 0)

    quant_features= {
        "RSI_Up_Flag": rsi_flag,
        "MACD_Crossover": macd_crossover,
        "SMA_Trend_Flag": sma_trend,
        "Value_Score": value_score,
        "Quality_Score": quality_score,
        "Final_Quant_Score": final_score,
    }

    return quant_features

cookie_manager = EncryptedCookieManager(
    prefix="stock_manager_cookies",
    password="temporary-password"  
)

# 쿠키 매니저가 준비되지 않았다면 종료
if not cookie_manager.ready():
    st.stop()


# 종목 선택 페이지
def select_stocks():
    st.title("주식 관리 프로그램 (S&P 500)")

    
    # 쿠키에서 관심/보유 종목 정보를 불러옴. 없으면 기본값 "[]" (빈 리스트)
    if "favorite_stocks" not in cookie_manager:
        cookie_manager["favorite_stocks"] = "[]"
    if "holding_stocks" not in cookie_manager:
        cookie_manager["holding_stocks"] = "[]"

    favorite_stocks = json.loads(cookie_manager["favorite_stocks"])
    holding_stocks = json.loads(cookie_manager["holding_stocks"])

    sp500_data = load_sp500_symbols()
    sp500_dict = {symbol: name for symbol, name in sp500_data}

    # 관심 종목 선택
    st.subheader("관심 종목 선택")
    favorite_stocks = st.multiselect(
        "관심 종목을 선택하세요",
        options=[symbol for symbol, _ in sp500_data],
        default=favorite_stocks,
        format_func=lambda x: f"{x} - {sp500_dict.get(x, '')}"
    )

    # 보유 종목 추가
    st.subheader("보유 종목 관리")
    # 새로운 보유 종목 추가
    with st.expander("새로운 보유 종목 추가"):
        new_symbol = st.selectbox(
            "종목 선택",
            options=[symbol for symbol, _ in sp500_data],
            format_func=lambda x: f"{x} - {sp500_dict.get(x, '')}"
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            purchase_quantity = st.number_input("수량", min_value=1, step=1)
        with col2:
            purchase_price = st.number_input("매수가 (USD)", min_value=0.01, step=0.01)
        with col3:
            purchase_date = st.date_input("매수일", datetime.now())
        
        if st.button("보유 종목 추가"):
            new_holding = {
                "symbol": new_symbol,
                "quantity": purchase_quantity,
                "price": purchase_price,
                "date": purchase_date.strftime("%Y-%m-%d")
            }
            # 기존 보유 종목 리스트에 새로운 종목 추가
            holding_stocks.append(new_holding)
            # 즉시 쿠키에 저장
            cookie_manager["holding_stocks"] = json.dumps(holding_stocks)
            cookie_manager.save()
            st.success(f"{new_symbol} 종목이 보유 종목에 추가되었습니다!")
            st.rerun()

    # 보유 종목 목록 표시
    if holding_stocks:
        st.subheader("보유 종목 목록")
        for idx, holding in enumerate(holding_stocks):
            col1, col2, col3, col4, col5 = st.columns([3, 1, 2, 2, 1])
            with col1:
                st.write(f"{holding['symbol']} - {sp500_dict.get(holding['symbol'], '')}")
            with col2:
                st.write(f"수량: {holding['quantity']}주")
            with col3:
                st.write(f"매수가: ${holding['price']:.2f}")
            with col4:
                st.write(f"매수일: {holding['date']}")
            with col5:
                if st.button("삭제", key=f"del_{idx}"):
                    holding_stocks.pop(idx)
                    # 즉시 쿠키에 저장
                    cookie_manager["holding_stocks"] = json.dumps(holding_stocks)
                    cookie_manager.save()
                    st.rerun()

    # 종목 분석 진행 버튼
    if st.button("종목 분석 진행"):
        cookie_manager["favorite_stocks"] = json.dumps(favorite_stocks)
        cookie_manager["holding_stocks"] = json.dumps(holding_stocks)
        cookie_manager.save()
        st.session_state["favorite_stocks"] = favorite_stocks
        st.session_state["holding_stocks"] = holding_stocks
        st.session_state["page"] = "data_analysis"
        # 분석을 위해 데이터를 초기화
        st.session_state["combined_stocks_data"] = {}
        st.rerun()

# 데이터 분석 페이지
def data_analysis():
    """
    전체 종목에 대해 기술적 분석과 퀀트 투자 전략을 계산
    """
    st.title("데이터 분석")

    # 선택한 종목 목록
    favorite_stocks = st.session_state.get("favorite_stocks", [])
    holding_stocks = st.session_state.get("holding_stocks", [])


    if "combined_stocks_data" not in st.session_state:
        st.session_state["combined_stocks_data"] = {}

    if not st.session_state["combined_stocks_data"]:

        combined_stocks_data = {}
        all_selected = set(favorite_stocks + [h['symbol'] for h in holding_stocks])

        for symbol in all_selected:
            is_favorite = symbol in favorite_stocks
            is_holding = symbol in [h['symbol'] for h in holding_stocks]
            holding_info = next((h for h in holding_stocks if h['symbol'] == symbol), None)
            technical_analysis_df = fetch_technical_analyze(symbol)
            fundamental_dict = fetch_fundamental_dict(symbol)

            # technical_analysis_df,fundamental_dict 두 데이터가 모두 존재하는 경우에만 분석 진행
            if technical_analysis_df is None or fundamental_dict is None:
                continue
            
            # quant_features_dict = generate_quant_features(technical_analysis_df,fundamental_dict)
            # quant_strategies_dict = generate_quant_strategies(technical_analysis_df,fundamental_dict)
            # 아직 효용성이 검증되지 않아서 비활성화
            quant_features_dict = None
            quant_strategies_dict = None

            combined_stocks_data[symbol] ={
                "symbol": symbol,
                "is_favorite": is_favorite,
                "holding_info": holding_info,
                "is_holding": is_holding,
                "technical_analysis_df": technical_analysis_df,
                "fundamental_dict": fundamental_dict,
                "quant_features_dict": quant_features_dict,
                "quant_strategies_dict": quant_strategies_dict,
            }
        
        # 분석된 데이터 저장 (세션 유지)
        st.session_state["combined_stocks_data"] = combined_stocks_data

    # 선택한 종목 (기본값 설정)
    all_selected = list(st.session_state["combined_stocks_data"].keys())

    # 선택 종목이 없으면 초기화
    # 세션 관리하려니까 방식이 맘에 안드네
    if "selected_symbol" not in st.session_state:
        st.session_state["selected_symbol"] = all_selected[0] if all_selected else None

    
    # 종목 선택 UI
    selected_symbol = st.selectbox(
        "분석할 종목 선택",
        all_selected,
        index=all_selected.index(st.session_state["selected_symbol"]) if st.session_state["selected_symbol"] in all_selected else 0
    )

    # 선택된 종목을 세션에 저장 (사용자가 선택할 때마다 갱신)
    st.session_state["selected_symbol"] = selected_symbol

    # 선택한 종목의 데이터 불러오기
    selected_data = st.session_state["combined_stocks_data"].get(selected_symbol, {})

    if selected_data:
        st.write(f"### 📈 {selected_symbol} 분석 결과")
        st.write("기술적 지표 요약:")
        st.dataframe(selected_data["technical_analysis_df"].tail())  # 최근 데이터 표시
        st.write("기본적 지표 요약:")
    
        st.write("#### 손익계산서")
        st.json(selected_data["fundamental_dict"]["income_stmt"])

        st.write("#### 대차대조표")
        st.json(selected_data["fundamental_dict"]["balance_sheet"])

        st.write("#### 현금흐름표")
        st.json(selected_data["fundamental_dict"]["cash_flow"])

        st.write("#### 주요 재무 지표")
        st.json(selected_data["fundamental_dict"]["key_metrics"])

        st.write("#### 주가 관련 지표")
        st.json(selected_data["fundamental_dict"]["stock_metrics"])

        st.write("#### 성장 및 배당 데이터")
        st.json(selected_data["fundamental_dict"]["growth_and_dividend"])

        # 퀀트 투자 전략 표시는 생략, 아직 기술이 확립되지 않음

    else:
        st.warning(f"📉 {selected_symbol}에 대한 분석 데이터가 없습니다.")


    # 투자가 선택 버튼
    if st.button("분석 투자가 선택"):
        st.session_state["selected_symbol"] = selected_symbol
        st.session_state["page"] = "select_investor"
        st.rerun()

# 투자가 선택 페이지
def select_investor():
    st.title("투자가 선택")

    investors = {
        "일반 AI": "일반적인 투자 의사결정",
        "워렌 버핏": "장기 가치 투자",
        "피터 린치": "성장주 투자",
        "벤저민 그레이엄": "저평가 주식 투자",
    }

    selected_investor = st.radio("선호하는 투자가를 선택하세요", list(investors.keys()))
    
    if selected_investor:
        st.write(f"🔍 {selected_investor} 스타일이 적용됩니다: {investors[selected_investor]}")

    # AI 분석 진행 버튼
    if st.button("🤖 AI 분석 진행"):
        st.session_state["selected_investor"] = selected_investor
        st.session_state["page"] = "ai_analysis"
        st.session_state["full_ai_response"] = ""
        st.rerun()

# AI 분석 페이지
def ai_analysis():
    st.title("AI 분석 결과")

    selected_investor = st.session_state.get("selected_investor", "")
    combined_stocks_data = st.session_state.get("combined_stocks_data", {})

    # 상태 변수 초기화
    if "full_ai_response" not in st.session_state:
        st.session_state["full_ai_response"] = ""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if not st.session_state["combined_stocks_data"]:
        st.warning("분석할 데이터가 없습니다. 먼저 데이터 분석을 진행하세요.")
        if st.button("데이터 분석으로 이동"):
            st.session_state["page"] = "data_analysis"
            st.rerun()
        return
    elif not selected_investor:
        st.warning("투자가를 선택하지 않았습니다. 먼저 투자가를 선택하세요.")
        if st.button("투자가 선택으로 이동"):
            st.session_state["page"] = "select_investor"
            st.rerun()
        return
    elif not st.session_state["full_ai_response"]:
        # 분석 진행
        client = OpenAI(api_key=st.session_state.api_key)
        # 선택한 투자가에 따라서 AI 모델에 전달할 시스템/프롬프트 텍스트 생성
        if(selected_investor == "워렌 버핏"):
            system_text = stock_prompt.warrenBuffett_system_text()
        elif(selected_investor == "피터 린치"):
            system_text = stock_prompt.peterLynch_system_text()
        elif(selected_investor == "벤저민 그레이엄"):
            system_text = stock_prompt.benjaminGraham_system_text()
        elif(selected_investor == "레이 달리오"): # 이건 리스크 관리 위주로 진행해야해서 TODO
            system_text = stock_prompt.rayDalio_system_text()
        else:

            system_text = stock_prompt.normal_system_text()
        
        prompt_text_list = stock_prompt.normal_prompt_text(combined_stocks_data)

        st.write(f"적용된 투자 스타일: {selected_investor}")
        # OpenAI API 요청 (동기 방식)
        ai_responses = []
        # 분석을 종복별로 따로 요청을 하는 방식으로 변경 고려중 TODO
        for idx, prompt_text in enumerate(prompt_text_list):
            try:
                print(f"AI 분석 진행 중... ({idx + 1}/{len(prompt_text_list)})")
                with st.spinner(f"AI 분석 진행 중... ({idx + 1}/{len(prompt_text_list)})"):

                    # langchain을 사용한 방식으로 이후 ai_chat과 연동이 되도록 변경 TODO
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_text},
                            {"role": "user", "content": prompt_text}
                        ]
                    )
                    ai_responses.append(response.choices[0].message.content if response.choices else "응답 없음")
            except Exception as e:
                ai_responses.append(f"AI 요청 중 오류 발생: {e}")

        # AI 분석 결과를 session_state에 즉시 저장**
        # 데이터 양 문제인지 추가질문시 데이터 인식 못하고 연산 두번 돌림
        st.session_state["full_ai_response"] = "\n\n".join(ai_responses)
    st.write("AI 분석 결과")
    st.markdown(st.session_state["full_ai_response"])
    # 추가 질문 버튼
    if st.button("추가 질문"):
        st.session_state["page"] = "ai_chat"
        st.session_state["chat_history"] = []
        st.rerun()

# AI 채팅 페이지
def ai_chat():
    st.title("AI와 대화")

    st.write("AI와 대화하면서 추가 질문을 해보세요.")

    # AI 분석 결과 가져오기
    full_ai_response = st.session_state.get("full_ai_response", "")
    
    if not full_ai_response:
        st.warning("AI 분석 결과가 없습니다. 먼저 AI 분석을 진행하세요.")
        if st.button("AI 분석으로 이동"):
            st.session_state["page"] = "ai_analysis"
            st.rerun()
        return


    try:
        model = ChatOpenAI(
            api_key=st.session_state.api_key,
            model="gpt-4o"
        )

        # 세션 상태 초기화
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # 기존 AI 분석 결과 표시
        if st.session_state.full_ai_response:
            with st.expander("기존 AI 분석 결과 보기"):
                st.markdown(st.session_state.full_ai_response)

        # 이전 대화 내역 표시
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 사용자 입력
        if prompt := st.chat_input("주식 분석 관련 질문을 입력하세요!"):
            # 사용자 메시지 저장
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # OpenAI에 전달할 메시지 형식 맞춤
            messages = [
                {"role": "system", "content": "다음은 주식 분석 결과입니다. 이 내용을 기반으로 질문에 답변해주세요."},
                {"role": "assistant", "content": st.session_state.full_ai_response},
                {"role": "user", "content": prompt}
            ]

            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("분석 중..."):
                    response = model.invoke(messages)  
                    ai_response = response.content if response else "응답 없음"
                    st.markdown(ai_response)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

    except Exception as e:
        st.error(f"오류 발생: {str(e)}")

    # 초기 화면으로 돌아가기
    if st.button("초기 화면으로"):
        st.session_state["page"] = "select_stocks"
        st.rerun()

# 페이지 렌더링
if st.session_state["page"] == "select_stocks":
    select_stocks()
elif st.session_state["page"] == "data_analysis":
    data_analysis()
elif st.session_state["page"] == "select_investor":
    select_investor()
elif st.session_state["page"] == "ai_analysis":
    ai_analysis()
elif st.session_state["page"] == "ai_chat":
    ai_chat()