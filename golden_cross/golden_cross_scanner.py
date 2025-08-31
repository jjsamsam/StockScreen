import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

class MarketTickerCollector:
    def __init__(self):
        """시장별 티커 수집 및 골든크로스 스크리닝 시스템"""
        self.data_dir = "market_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def get_nasdaq_tickers(self) -> pd.DataFrame:
        """나스닥 상장 종목 리스트 가져오기"""
        print("📊 나스닥 티커 수집 중...")
        
        try:
            # NASDAQ에서 공식 리스트 다운로드
            nasdaq_url = "https://www.nasdaq.com/api/v1/screener?download=true&limit=10000"
            
            # 대안: 미리 정의된 주요 나스닥 종목들 (더 안정적)
            major_nasdaq = [
                'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'NVDA',
                'PYPL', 'ADBE', 'NFLX', 'CRM', 'INTC', 'CSCO', 'CMCSA', 'PEP',
                'COST', 'TMUS', 'AVGO', 'TXN', 'QCOM', 'CHTR', 'AMGN', 'SBUX',
                'INTU', 'GILD', 'MDLZ', 'ISRG', 'BKNG', 'REGN', 'ADP', 'VRTX',
                'FISV', 'CSX', 'ATVI', 'BIIB', 'ILMN', 'MRVL', 'KHC', 'ADI',
                'LRCX', 'EXC', 'XEL', 'ORLY', 'WLTW', 'KLAC', 'CTAS', 'PAYX',
                'EA', 'CTSH', 'FAST', 'VRSK', 'CDNS', 'MXIM', 'SWKS', 'ALXN',
                'SIRI', 'BMRN', 'XLNX', 'MCHP', 'CERN', 'INCY', 'ULTA', 'TTWO'
            ]
            
            # yfinance를 통해 실제 거래되는 종목인지 확인
            valid_tickers = []
            for ticker in major_nasdaq:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    if 'longName' in info and info.get('exchange') in ['NMS', 'NGM', 'NCM']:
                        valid_tickers.append({
                            'Symbol': ticker,
                            'Name': info.get('longName', ''),
                            'Market': 'NASDAQ',
                            'Country': 'US',
                            'Exchange': info.get('exchange', '')
                        })
                except:
                    continue
            
            df = pd.DataFrame(valid_tickers)
            print(f"✅ 나스닥 {len(df)}개 종목 수집 완료")
            return df
            
        except Exception as e:
            print(f"❌ 나스닥 데이터 수집 실패: {e}")
            return pd.DataFrame()
    
    def get_swedish_tickers(self) -> pd.DataFrame:
        """스웨덴 주식시장(Nasdaq Stockholm) 종목 리스트"""
        print("📊 스웨덴 티커 수집 중...")
        
        # 스웨덴 주요 종목들 (OMX Stockholm 30 + 기타 주요 종목)
        swedish_stocks = [
            # OMX Stockholm 30 구성종목
            'ASSA-B.ST', 'ATCO-A.ST', 'ATCO-B.ST', 'ALFA.ST', 'BOL.ST',
            'ELUX-B.ST', 'ERIC.ST', 'ESSITY-B.ST', 'EVO.ST', 'GETI-B.ST',
            'HM-B.ST', 'HEXA-B.ST', 'ICA.ST', 'INVE-B.ST', 'KINV-B.ST',
            'NDA-SE.ST', 'SAND.ST', 'SCA-B.ST', 'SEB-A.ST', 'SECU-B.ST',
            'SKF-B.ST', 'SWED-A.ST', 'SHB-A.ST', 'TEL2-B.ST', 'VOLV-A.ST',
            'VOLV-B.ST', 'WALL-B.ST', 'SWMA.ST', 'SINCH.ST', 'ABB.ST',
            
            # 기타 주요 종목들
            'ASTRA.ST', 'CAST.ST', 'DUST.ST', 'EMBR-B.ST', 'EPIR-B.ST',
            'FABG.ST', 'FING-B.ST', 'HUSQ-B.ST', 'LATO-B.ST', 'LIFCO-B.ST',
            'LUND-B.ST', 'NCC-B.ST', 'PEAB-B.ST', 'RROS.ST', 'SAGA-B.ST',
            'SAAB-B.ST', 'SKA-B.ST', 'SSAB-A.ST', 'TELIA.ST', 'THULE.ST'
        ]
        
        valid_tickers = []
        for ticker in swedish_stocks:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                if 'longName' in info:
                    valid_tickers.append({
                        'Symbol': ticker,
                        'Name': info.get('longName', ''),
                        'Market': 'Stockholm',
                        'Country': 'SE',
                        'Exchange': 'STO'
                    })
            except:
                continue
        
        df = pd.DataFrame(valid_tickers)
        print(f"✅ 스웨덴 {len(df)}개 종목 수집 완료")
        return df
    
    def get_kospi_tickers(self) -> pd.DataFrame:
        """코스피 종목 리스트 가져오기"""
        print("📊 코스피 티커 수집 중...")
        
        # 코스피 주요 종목들 (시가총액 기준 상위 종목들)
        kospi_stocks = [
            # 대형주
            '005930.KS', '000660.KS', '035420.KS', '005490.KS', '051910.KS',
            '068270.KS', '035720.KS', '012330.KS', '028260.KS', '066570.KS',
            '003670.KS', '096770.KS', '034730.KS', '015760.KS', '323410.KS',
            '017670.KS', '030200.KS', '036570.KS', '033780.KS', '018260.KS',
            
            # 중견주
            '011200.KS', '086790.KS', '161390.KS', '024110.KS', '009830.KS',
            '047050.KS', '271560.KS', '006400.KS', '032830.KS', '010130.KS',
            '003550.KS', '029780.KS', '000270.KS', '051900.KS', '267250.KS',
            '042700.KS', '097950.KS', '007070.KS', '267260.KS', '251270.KS',
            
            # 기타 주요 종목
            '000810.KS', '105560.KS', '055550.KS', '016360.KS', '090430.KS',
            '009150.KS', '010950.KS', '018880.KS', '004020.KS', '139480.KS'
        ]
        
        valid_tickers = []
        for ticker in kospi_stocks:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                if 'longName' in info:
                    valid_tickers.append({
                        'Symbol': ticker,
                        'Name': info.get('longName', ''),
                        'Market': 'KOSPI',
                        'Country': 'KR',
                        'Exchange': 'KRX'
                    })
            except:
                continue
        
        df = pd.DataFrame(valid_tickers)
        print(f"✅ 코스피 {len(df)}개 종목 수집 완료")
        return df
    
    def get_kosdaq_tickers(self) -> pd.DataFrame:
        """코스닥 종목 리스트 가져오기"""
        print("📊 코스닥 티커 수집 중...")
        
        # 코스닥 주요 종목들
        kosdaq_stocks = [
            # 대형주
            '091990.KQ', '066970.KQ', '196170.KQ', '240810.KQ', '035900.KQ',
            '293490.KQ', '067310.KQ', '214390.KQ', '112040.KQ', '263750.KQ',
            '277810.KQ', '068760.KQ', '058470.KQ', '095660.KQ', '053800.KQ',
            
            # 중견주
            '225570.KQ', '357780.KQ', '131970.KQ', '086520.KQ', '078600.KQ',
            '206640.KQ', '121600.KQ', '064550.KQ', '039030.KQ', '041510.KQ',
            '088350.KQ', '060280.KQ', '108860.KQ', '900140.KQ', '348210.KQ',
            
            # 바이오/IT 주요 종목
            '326030.KQ', '195940.KQ', '137400.KQ', '336260.KQ', '101490.KQ',
            '048410.KQ', '950210.KQ', '950130.KQ', '290650.KQ', '140860.KQ'
        ]
        
        valid_tickers = []
        for ticker in kosdaq_stocks:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                if 'longName' in info:
                    valid_tickers.append({
                        'Symbol': ticker,
                        'Name': info.get('longName', ''),
                        'Market': 'KOSDAQ',
                        'Country': 'KR',
                        'Exchange': 'KRX'
                    })
            except:
                continue
        
        df = pd.DataFrame(valid_tickers)
        print(f"✅ 코스닥 {len(df)}개 종목 수집 완료")
        return df
    
    def collect_all_tickers(self) -> Dict[str, pd.DataFrame]:
        """모든 시장의 티커를 수집하고 CSV로 저장"""
        print("🚀 전체 시장 티커 수집 시작!")
        print("=" * 60)
        
        tickers_data = {}
        
        # 각 시장별 데이터 수집
        markets = {
            'nasdaq': self.get_nasdaq_tickers,
            'swedish': self.get_swedish_tickers,
            'kospi': self.get_kospi_tickers,
            'kosdaq': self.get_kosdaq_tickers
        }
        
        for market_name, collect_func in markets.items():
            try:
                df = collect_func()
                if not df.empty:
                    # CSV 파일로 저장
                    csv_path = os.path.join(self.data_dir, f"{market_name}_tickers.csv")
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    tickers_data[market_name] = df
                    
                    print(f"💾 {market_name.upper()} 티커 저장: {csv_path}")
                    print(f"   총 {len(df)}개 종목")
                else:
                    print(f"❌ {market_name.upper()} 데이터 없음")
                    
                time.sleep(1)  # API 호출 제한 방지
                
            except Exception as e:
                print(f"❌ {market_name} 수집 실패: {e}")
        
        # 전체 통합 파일 생성
        if tickers_data:
            all_tickers = pd.concat(tickers_data.values(), ignore_index=True)
            all_csv_path = os.path.join(self.data_dir, "all_markets_tickers.csv")
            all_tickers.to_csv(all_csv_path, index=False, encoding='utf-8')
            
            print("\n" + "=" * 60)
            print("✅ 전체 수집 완료!")
            print(f"📁 데이터 저장 위치: {self.data_dir}")
            print(f"📊 총 수집 종목 수: {len(all_tickers)}")
            print(f"📋 통합 파일: {all_csv_path}")
            
            # 시장별 요약
            market_summary = all_tickers.groupby('Market').size()
            print("\n📈 시장별 종목 수:")
            for market, count in market_summary.items():
                print(f"   {market}: {count}개")
        
        return tickers_data

class GoldenCrossScreener:
    def __init__(self, data_dir: str = "market_data"):
        """CSV 파일 기반 골든크로스 스크리너"""
        self.data_dir = data_dir
        
    def load_market_tickers(self, market: str = None) -> pd.DataFrame:
        """저장된 CSV 파일에서 티커 로드"""
        if market:
            csv_path = os.path.join(self.data_dir, f"{market}_tickers.csv")
            if os.path.exists(csv_path):
                return pd.read_csv(csv_path)
            else:
                print(f"❌ {csv_path} 파일이 없습니다.")
                return pd.DataFrame()
        else:
            # 모든 시장 로드
            all_path = os.path.join(self.data_dir, "all_markets_tickers.csv")
            if os.path.exists(all_path):
                return pd.read_csv(all_path)
            else:
                print(f"❌ {all_path} 파일이 없습니다.")
                return pd.DataFrame()
    
    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """이동평균선 및 기울기 계산"""
        data['MA60'] = data['Close'].rolling(window=60).mean()
        data['MA120'] = data['Close'].rolling(window=120).mean()
        
        # 기울기 계산 (최근 5일간)
        data['MA60_slope'] = data['MA60'].rolling(window=5).apply(
            lambda x: np.polyfit(range(5), x, 1)[0] if len(x) == 5 else np.nan, raw=True
        )
        data['MA120_slope'] = data['MA120'].rolling(window=5).apply(
            lambda x: np.polyfit(range(5), x, 1)[0] if len(x) == 5 else np.nan, raw=True
        )
        
        return data
    
    def check_golden_cross_conditions(self, symbol: str, market: str) -> Dict:
        """골든크로스 조건 확인"""
        try:
            # 데이터 다운로드
            stock = yf.Ticker(symbol)
            data = stock.history(period="1y")
            
            if len(data) < 150:
                return None
                
            # 이동평균 계산
            data = self.calculate_moving_averages(data)
            
            # 최근 120일 데이터
            recent_data = data.iloc[-120:].copy()
            
            # 조건 1: 지난 3달(90일)간 60일선이 120일선 아래
            three_months_period = recent_data.iloc[-90:-10]
            was_below = (three_months_period['MA60'] < three_months_period['MA120']).all()
            
            # 조건 2: 1주일 전(5-10일 전) 골든크로스 발생
            cross_period = recent_data.iloc[-10:-5]
            cross_detected = False
            cross_date = None
            
            for i in range(len(cross_period)-1):
                before = cross_period.iloc[i]
                after = cross_period.iloc[i+1]
                
                if (before['MA60'] <= before['MA120'] and 
                    after['MA60'] > after['MA120']):
                    cross_detected = True
                    cross_date = after.name
                    break
            
            # 조건 3: 현재 두 이평선 모두 우상향
            current_ma60_slope = recent_data['MA60_slope'].iloc[-1]
            current_ma120_slope = recent_data['MA120_slope'].iloc[-1]
            
            ma60_uptrend = current_ma60_slope > 0
            ma120_uptrend = current_ma120_slope > 0
            currently_above = recent_data['MA60'].iloc[-1] > recent_data['MA120'].iloc[-1]
            
            # 결과 반환
            result = {
                'symbol': symbol,
                'market': market,
                'was_below_3months': was_below,
                'golden_cross_detected': cross_detected,
                'cross_date': cross_date,
                'ma60_uptrend': ma60_uptrend,
                'ma120_uptrend': ma120_uptrend,
                'currently_above': currently_above,
                'current_price': recent_data['Close'].iloc[-1],
                'ma60_current': recent_data['MA60'].iloc[-1],
                'ma120_current': recent_data['MA120'].iloc[-1],
                'ma60_slope': current_ma60_slope,
                'ma120_slope': current_ma120_slope,
                'meets_all_conditions': (was_below and cross_detected and 
                                       ma60_uptrend and ma120_uptrend and currently_above)
            }
            
            return result
            
        except Exception as e:
            print(f"❌ {symbol} 분석 실패: {e}")
            return None
    
    def screen_market(self, market: str = None, max_workers: int = 10) -> pd.DataFrame:
        """특정 시장 또는 전체 시장 스크리닝"""
        # 티커 로드
        tickers_df = self.load_market_tickers(market)
        if tickers_df.empty:
            return pd.DataFrame()
        
        # 시장 필터링 (특정 시장 지정 시)
        if market:
            market_names = {
                'nasdaq': 'NASDAQ',
                'swedish': 'Stockholm', 
                'kospi': 'KOSPI',
                'kosdaq': 'KOSDAQ'
            }
            if market in market_names:
                tickers_df = tickers_df[tickers_df['Market'] == market_names[market]]
        
        print(f"🔍 {len(tickers_df)}개 종목 스크리닝 시작...")
        print("=" * 60)
        
        results = []
        completed = 0
        
        # 병렬 처리로 성능 향상
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 작업 제출
            future_to_ticker = {
                executor.submit(self.check_golden_cross_conditions, row['Symbol'], row['Market']): 
                (row['Symbol'], row['Market']) 
                for _, row in tickers_df.iterrows()
            }
            
            # 결과 수집
            for future in as_completed(future_to_ticker):
                symbol, market = future_to_ticker[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        
                        # 조건 만족 종목 실시간 출력
                        if result['meets_all_conditions']:
                            print(f"✅ [{result['market']}] {result['symbol']}")
                            print(f"   골든크로스: {result['cross_date'].strftime('%Y-%m-%d') if result['cross_date'] else 'N/A'}")
                            print(f"   현재가: {result['current_price']:.2f}")
                            print(f"   60일선: {result['ma60_current']:.2f} ↗")
                            print(f"   120일선: {result['ma120_current']:.2f} ↗")
                            print("-" * 40)
                        
                        # 진행상황 표시
                        if completed % 10 == 0:
                            print(f"📊 진행상황: {completed}/{len(tickers_df)} ({completed/len(tickers_df)*100:.1f}%)")
                            
                except Exception as e:
                    print(f"❌ {symbol} 처리 실패: {e}")
        
        results_df = pd.DataFrame(results)
        
        # 결과 저장
        if not results_df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.data_dir, f"golden_cross_results_{timestamp}.csv")
            results_df.to_csv(results_path, index=False, encoding='utf-8')
            print(f"\n💾 결과 저장: {results_path}")
        
        return results_df
    
    def display_results_by_market(self, results_df: pd.DataFrame):
        """시장별 결과 요약 표시"""
        if results_df.empty:
            print("❌ 분석 결과가 없습니다.")
            return
        
        print("\n" + "="*80)
        print("🎯 골든크로스 스크리닝 결과 (시장별 요약)")
        print("="*80)
        
        # 전체 요약
        total_analyzed = len(results_df)
        total_qualified = len(results_df[results_df['meets_all_conditions'] == True])
        
        print(f"📊 총 분석 종목: {total_analyzed}개")
        print(f"✅ 조건 만족 종목: {total_qualified}개 ({total_qualified/total_analyzed*100:.1f}%)")
        
        # 시장별 상세 결과
        markets = results_df['market'].unique()
        
        for market in markets:
            market_data = results_df[results_df['market'] == market]
            qualified = market_data[market_data['meets_all_conditions'] == True]
            
            print(f"\n🏢 {market} 시장:")
            print(f"   분석: {len(market_data)}개 / 조건만족: {len(qualified)}개")
            
            if len(qualified) > 0:
                print("   ✅ 골든크로스 완료 종목:")
                for _, stock in qualified.iterrows():
                    cross_date = stock['cross_date'].strftime('%m/%d') if stock['cross_date'] else 'N/A'
                    print(f"      • {stock['symbol']}: {cross_date} 교차, ${stock['current_price']:.2f}")
    
    def plot_qualified_stocks(self, results_df: pd.DataFrame, limit: int = 5):
        """조건 만족 종목들의 차트 그리기"""
        qualified = results_df[results_df['meets_all_conditions'] == True]
        
        if qualified.empty:
            print("❌ 차트를 그릴 조건 만족 종목이 없습니다.")
            return
        
        # 상위 N개 종목만 차트 생성
        stocks_to_plot = qualified.head(limit)
        
        fig, axes = plt.subplots(len(stocks_to_plot), 1, figsize=(15, 6*len(stocks_to_plot)))
        if len(stocks_to_plot) == 1:
            axes = [axes]
        
        for idx, (_, stock) in enumerate(stocks_to_plot.iterrows()):
            try:
                # 데이터 다운로드
                data = yf.Ticker(stock['symbol']).history(period="6mo")
                data = self.calculate_moving_averages(data)
                
                ax = axes[idx]
                
                # 차트 그리기
                ax.plot(data.index, data['Close'], label='Close', linewidth=2, color='black')
                ax.plot(data.index, data['MA60'], label='MA60', linewidth=2, color='blue')
                ax.plot(data.index, data['MA120'], label='MA120', linewidth=2, color='red')
                
                # 골든크로스 지점 표시
                if stock['cross_date']:
                    cross_idx = data.index.get_loc(stock['cross_date'], method='nearest')
                    ax.axvline(x=data.index[cross_idx], color='gold', linestyle='--', alpha=0.8, label='Golden Cross')
                
                ax.set_title(f"[{stock['market']}] {stock['symbol']} - Golden Cross Pattern", 
                           fontsize=14, fontweight='bold')
                ax.set_ylabel('Price')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"❌ {stock['symbol']} 차트 생성 실패: {e}")
        
        plt.tight_layout()
        plt.show()

# 사용 예시
def main():
    print("🚀 시장별 티커 수집 및 골든크로스 스크리닝 시스템")
    print("="*80)
    
    # 1단계: 티커 수집
    collector = MarketTickerCollector()
    
    print("\n1️⃣ 단계: 시장별 티커 수집")
    tickers_data = collector.collect_all_tickers()
    
    if not tickers_data:
        print("❌ 티커 데이터 수집에 실패했습니다.")
        return
    
    # 2단계: 골든크로스 스크리닝
    print("\n2️⃣ 단계: 골든크로스 패턴 스크리닝")
    screener = GoldenCrossScreener()
    
    # 전체 시장 스크리닝 (원하는 특정 시장만 하려면 market='nasdaq' 등으로 지정)
    results = screener.screen_market()  # 또는 market='nasdaq', 'swedish', 'kospi', 'kosdaq'
    
    # 3단계: 결과 분석
    print("\n3️⃣ 단계: 결과 분석")
    screener.display_results_by_market(results)
    
    # 4단계: 차트 생성
    if len(results[results['meets_all_conditions'] == True]) > 0:
        print("\n4️⃣ 단계: 조건 만족 종목 차트")
        screener.plot_qualified_stocks(results, limit=3)

if __name__ == "__main__":
    main()