"""
backtesting_system.py
백테스팅 시스템 - 매수/매도 전략 검증 (추천도 기반 백테스팅 추가)
"""

import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 최적화 모듈
from cache_manager import get_stock_data
from matplotlib_optimizer import ChartManager


class RecommendationBacktestingEngine:
    """추천도 기반 백테스팅 엔진 - 특정 시점에서 최고 추천도 종목 선택"""
    
    def __init__(self, technical_analyzer):
        self.technical_analyzer = technical_analyzer
        self.results = []
    
    def run_recommendation_backtest(self, symbols, months_back=6, min_recommendation_score=75):
        """
        추천도 기반 백테스팅 실행
        
        과정:
        1. N개월 전 시점에서 모든 종목 스크리닝
        2. 매수/매도 조건 만족하는 종목들의 추천도 계산
        3. 추천도가 가장 높은 종목 선택
        4. 그 종목에 투자했다면 현재까지의 수익률 계산
        
        매개변수:
        - symbols: 분석할 종목 리스트
        - months_back: 몇 개월 전부터 백테스팅할지 (6 또는 12)
        - min_recommendation_score: 최소 추천도 (기본 75점)
        """
        
        target_date = datetime.now() - timedelta(days=30 * months_back)
        
        print(f"🎯 추천도 기반 백테스팅 시작")
        print(f"📅 분석 기준일: {target_date.strftime('%Y-%m-%d')}")
        print(f"📊 분석 종목 수: {len(symbols)}개")
        print(f"⭐ 최소 추천도: {min_recommendation_score}점")
        print("-" * 60)
        
        candidates = []
        
        # 1단계: 각 종목별로 분석 기준일의 추천도 계산
        for i, symbol in enumerate(symbols):
            try:
                print(f"분석 중 ({i+1}/{len(symbols)}): {symbol}")
                
                # 과거 데이터 가져오기 (분석일 기준 충분한 과거 데이터 필요) - 캐싱 사용
                data_start = target_date - timedelta(days=200)  # 지표 계산용 여유
                data_end = target_date + timedelta(days=30)     # 분석일 이후 여유

                # 기간 계산 후 캐싱 사용
                days_diff = (data_end - data_start).days + 10
                period_str = f"{days_diff}d"

                data = get_stock_data(symbol, period=period_str)
                
                if len(data) < 120:
                    print(f"   ⚠️ 데이터 부족: {len(data)}일")
                    continue
                
                # 분석 기준일에 가장 가까운 데이터 찾기
                target_idx = data.index.get_indexer([target_date], method='nearest')[0]
                
                if target_idx < 60:  # 충분한 과거 데이터 필요
                    print(f"   ⚠️ 과거 데이터 부족")
                    continue
                
                # 기술적 지표 계산 (분석일까지의 데이터만 사용)
                analysis_data = data.iloc[:target_idx+1]
                analysis_data = self.technical_analyzer.calculate_all_indicators(analysis_data)
                
                # 매수/매도 조건 체크 및 추천도 계산
                recommendation_score = self.calculate_recommendation_score(analysis_data)
                
                if recommendation_score >= min_recommendation_score:
                    entry_price = analysis_data.iloc[-1]['Close']
                    entry_date = analysis_data.index[-1]
                    
                    candidate = {
                        'symbol': symbol,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'recommendation_score': recommendation_score
                    }
                    
                    candidates.append(candidate)
                    print(f"   ✅ 매수 후보 - 추천도: {recommendation_score}점, 가격: {entry_price:,.0f}")
                else:
                    print(f"   ❌ 조건 미달 - 추천도: {recommendation_score}점")
                    
            except Exception as e:
                print(f"   ❌ 분석 실패: {e}")
                continue
        
        print(f"\n🎯 매수 후보 종목: {len(candidates)}개 발견")
        
        if not candidates:
            return {
                'status': 'no_candidates',
                'message': '조건을 만족하는 종목이 없습니다.',
                'total_analyzed': len(symbols),
                'min_score_required': min_recommendation_score
            }
        
        # 2단계: 추천도가 가장 높은 종목 선택
        best_candidate = max(candidates, key=lambda x: x['recommendation_score'])
        
        print(f"\n🏆 선택된 종목: {best_candidate['symbol']}")
        print(f"   📅 매수일: {best_candidate['entry_date'].strftime('%Y-%m-%d')}")
        print(f"   💰 매수가: {best_candidate['entry_price']:,.0f}")
        print(f"   ⭐ 추천도: {best_candidate['recommendation_score']}점")
        
        # 3단계: 현재까지의 투자 성과 계산
        performance = self.calculate_investment_performance(best_candidate)
        
        if performance:
            # 결과 종합
            result = {
                'status': 'success',
                'analysis_date': target_date,
                'months_back': months_back,
                'total_analyzed': len(symbols),
                'total_candidates': len(candidates),
                'selected_stock': best_candidate,
                'performance': performance,
                'other_candidates': sorted(candidates, key=lambda x: x['recommendation_score'], reverse=True)[:5]  # 상위 5개
            }
            
            self.print_performance_summary(result)
            return result
        else:
            return {
                'status': 'price_error',
                'message': '현재가 조회에 실패했습니다.',
                'selected_stock': best_candidate
            }
    
    def calculate_recommendation_score(self, data):
        """
        추천도 계산 (기존 스크리너의 로직과 유사)
        
        매수 조건들을 종합하여 0-100점 사이의 점수로 환산
        """
        if len(data) < 60:
            return 0
        
        score = 0
        current = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else current
        
        try:
            # 기본 조건들 (총 100점)
            
            # 1. 이동평균 정렬 상태 (25점)
            if 'MA60' in data.columns and 'MA120' in data.columns:
                ma60_current = current['MA60']
                ma120_current = current['MA120']
                ma60_prev = prev['MA60']
                ma120_prev = prev['MA120']
                
                # 60일선이 120일선 위에 있고 상승 추세
                if ma60_current > ma120_current:
                    score += 15
                    # 최근에 돌파했다면 추가 점수
                    if ma60_prev <= ma120_prev:
                        score += 10
                
            # 2. RSI 조건 (20점)
            if 'RSI' in data.columns:
                rsi_current = current['RSI']
                rsi_prev = prev['RSI']
                
                # 과매도에서 반등
                if rsi_prev <= 30 and rsi_current > 30:
                    score += 20
                # 적정 구간
                elif 30 <= rsi_current <= 70:
                    score += 10
            
            # 3. 볼린저밴드 조건 (15점)
            if 'BB_Lower' in data.columns:
                close_price = current['Close']
                bb_lower = current['BB_Lower']
                
                # 하단 밴드 근처 (저점 매수 기회)
                if close_price <= bb_lower * 1.05:  # 5% 여유
                    score += 15
            
            # 4. MACD 조건 (20점)
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                macd_current = current['MACD']
                macd_signal_current = current['MACD_Signal']
                macd_prev = prev['MACD']
                macd_signal_prev = prev['MACD_Signal']
                
                # 골든크로스
                if macd_current > macd_signal_current:
                    score += 10
                    # 최근 골든크로스
                    if macd_prev <= macd_signal_prev:
                        score += 10
            
            # 5. 거래량 조건 (10점)
            if 'Volume' in data.columns and len(data) >= 20:
                current_volume = current['Volume']
                avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                
                if current_volume > avg_volume * 1.5:  # 평균 대비 50% 이상 증가
                    score += 10
            
            # 6. 가격 모멘텀 (10점)
            if len(data) >= 5:
                current_close = current['Close']
                week_ago_close = data['Close'].iloc[-5]
                
                if current_close > week_ago_close:
                    score += 10
        
        except Exception as e:
            print(f"추천도 계산 오류: {e}")
            return 0
        
        return min(score, 100)  # 최대 100점으로 제한
    
    def calculate_investment_performance(self, candidate):
        """투자 성과 계산"""
        try:
            symbol = candidate['symbol']
            entry_price = candidate['entry_price']
            entry_date = candidate['entry_date']
            
            # 현재가 조회 - 캐싱 사용
            current_data = get_stock_data(symbol, period="2d")
            
            if len(current_data) == 0:
                return None
            
            current_price = current_data['Close'].iloc[-1]
            current_date = datetime.now()
            
            # 수익률 계산
            return_rate = (current_price - entry_price) / entry_price * 100
            holding_period = (current_date - entry_date).days
            
            # 연환산 수익률 (복리 적용)
            if holding_period > 0:
                annual_return = ((current_price / entry_price) ** (365 / holding_period) - 1) * 100
            else:
                annual_return = 0
            
            return {
                'entry_price': entry_price,
                'current_price': current_price,
                'return_rate': return_rate,
                'annual_return': annual_return,
                'holding_period': holding_period,
                'profit_loss_amount': current_price - entry_price
            }
            
        except Exception as e:
            print(f"성과 계산 오류: {e}")
            return None
    
    def print_performance_summary(self, result):
        """성과 요약 출력"""
        perf = result['performance']
        stock = result['selected_stock']
        
        print(f"\n" + "="*60)
        print(f"📈 투자 성과 요약")
        print(f"="*60)
        print(f"🏢 종목: {stock['symbol']}")
        print(f"📅 매수일: {stock['entry_date'].strftime('%Y-%m-%d')}")
        print(f"💰 매수가: {perf['entry_price']:,.0f}원")
        print(f"💰 현재가: {perf['current_price']:,.0f}원")
        print(f"📊 수익률: {perf['return_rate']:+.2f}%")
        print(f"📊 연환산 수익률: {perf['annual_return']:+.2f}%")
        print(f"⏱️ 보유기간: {perf['holding_period']}일")
        print(f"🎯 당시 추천도: {stock['recommendation_score']}점")
        print(f"-"*40)
        print(f"💵 투자금액별 손익:")
        print(f"   100만원 → {((perf['current_price'] / perf['entry_price']) * 1000000):,.0f}원 (손익: {((perf['current_price'] / perf['entry_price'] - 1) * 1000000):+,.0f}원)")
        print(f"   1000주 → {(perf['current_price'] * 1000):,.0f}원 (손익: {(perf['profit_loss_amount'] * 1000):+,.0f}원)")


class BacktestingEngine:
    """기존 백테스팅 엔진 (유지)"""
    
    def __init__(self, technical_analyzer):
        self.technical_analyzer = technical_analyzer
        self.results = []
    
    def run_backtest(self, symbols, buy_conditions, sell_conditions, 
                    start_date, end_date, initial_capital=100000):
        """기존 백테스팅 실행 (기존 코드 유지)"""
        
        print(f"🔄 백테스팅 시작: {start_date} ~ {end_date}")
        print(f"💰 초기 자본: ${initial_capital:,}")
        
        portfolio = Portfolio(initial_capital)
        trade_log = []
        
        # 각 종목별로 백테스팅 수행
        for symbol in symbols:
            try:
                print(f"\n📊 {symbol} 분석 중...")
                
                # 과거 데이터 다운로드 (백테스팅 기간 + 여유분) - 캐싱 사용
                data_start = start_date - timedelta(days=180)  # 지표 계산용 여유

                # 기간 계산
                days_diff = (end_date - data_start).days + 10
                period_str = f"{days_diff}d"

                data = get_stock_data(symbol, period=period_str)
                
                if len(data) < 120:
                    print(f"⚠️ {symbol}: 데이터 부족")
                    continue
                
                # 기술적 지표 계산
                data = self.technical_analyzer.calculate_all_indicators(data)
                
                # 백테스팅 기간만 추출
                backtest_data = data[start_date:end_date]
                
                if len(backtest_data) < 30:
                    print(f"⚠️ {symbol}: 백테스팅 기간 데이터 부족")
                    continue
                
                # 일별 신호 체크 및 거래 실행
                trades = self.simulate_trading(
                    symbol, backtest_data, buy_conditions, sell_conditions, portfolio
                )
                
                trade_log.extend(trades)
                
            except Exception as e:
                print(f"❌ {symbol} 오류: {e}")
                continue
        
        # 백테스팅 결과 분석
        results = self.analyze_results(portfolio, trade_log, initial_capital)
        
        return results, trade_log
    
    def simulate_trading(self, symbol, data, buy_conditions, sell_conditions, portfolio):
        """개별 종목 거래 시뮬레이션 (기존 코드 유지)"""
        trades = []
        position = None  # 현재 포지션 (None: 보유 없음, dict: 매수 정보)
        
        for date, row in data.iterrows():
            try:
                # 현재 보유 중이 아니면 매수 신호 체크
                if position is None:
                    if self.check_buy_signal(data.loc[:date], buy_conditions):
                        # 매수 실행
                        shares = int(portfolio.cash * 0.1 / row['Close'])  # 10% 비중
                        if shares > 0:
                            cost = shares * row['Close']
                            if portfolio.cash >= cost:
                                portfolio.buy(symbol, shares, row['Close'], date)
                                position = {
                                    'symbol': symbol,
                                    'shares': shares,
                                    'buy_price': row['Close'],
                                    'buy_date': date
                                }
                                
                                print(f"📈 매수: {symbol} {shares}주 @ ${row['Close']:.2f}")
                
                # 현재 보유 중이면 매도 신호 체크
                elif position is not None:
                    sell_signal = False
                    sell_reason = ""
                    
                    # 매도 조건 체크
                    if self.check_sell_signal(data.loc[:date], sell_conditions, position):
                        sell_signal = True
                        sell_reason = "조건 매도"
                    
                    # 손절/익절 체크 (예시: -7% 손절, +20% 익절)
                    profit_rate = (row['Close'] - position['buy_price']) / position['buy_price']
                    if profit_rate <= -0.07:
                        sell_signal = True
                        sell_reason = "손절 (-7%)"
                    elif profit_rate >= 0.20:
                        sell_signal = True
                        sell_reason = "익절 (+20%)"
                    
                    # 매도 실행
                    if sell_signal:
                        portfolio.sell(symbol, position['shares'], row['Close'], date)
                        
                        # 거래 기록
                        trade = {
                            'symbol': symbol,
                            'buy_date': position['buy_date'],
                            'sell_date': date,
                            'buy_price': position['buy_price'],
                            'sell_price': row['Close'],
                            'shares': position['shares'],
                            'profit': (row['Close'] - position['buy_price']) * position['shares'],
                            'profit_rate': profit_rate,
                            'holding_days': (date - position['buy_date']).days,
                            'reason': sell_reason
                        }
                        trades.append(trade)
                        
                        print(f"📉 매도: {symbol} {position['shares']}주 @ ${row['Close']:.2f} ({profit_rate*100:.1f}%)")
                        
                        position = None
            
            except Exception as e:
                print(f"❌ {date} {symbol} 거래 오류: {e}")
                continue
        
        return trades
    
    def check_buy_signal(self, data, buy_conditions):
        """매수 신호 체크 (기존 코드 유지)"""
        if len(data) < 2:
            return False
            
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        signals = 0
        
        # 60일선이 120일선 돌파
        if buy_conditions.get('ma_cross', False):
            if (prev['MA60'] <= prev['MA120'] and 
                current['MA60'] > current['MA120']):
                signals += 1
        
        # RSI 과매도에서 반등
        if buy_conditions.get('rsi_oversold', False):
            if prev['RSI'] <= 30 and current['RSI'] > 30:
                signals += 1
        
        # 볼린저밴드 하단 터치
        if buy_conditions.get('bb_touch', False):
            if current['Close'] <= current['BB_Lower'] * 1.02:  # 2% 여유
                signals += 1
        
        # MACD 골든크로스
        if buy_conditions.get('macd_cross', False):
            if (prev['MACD'] <= prev['MACD_Signal'] and 
                current['MACD'] > current['MACD_Signal']):
                signals += 1
        
        # 최소 1개 이상의 신호가 있어야 매수
        return signals >= 1
    
    def check_sell_signal(self, data, sell_conditions, position):
        """매도 신호 체크 (기존 코드 유지)"""
        if len(data) < 2:
            return False
            
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        # 데드크로스
        if sell_conditions.get('dead_cross', False):
            if (prev['MA60'] >= prev['MA120'] and 
                current['MA60'] < current['MA120']):
                return True
        
        # RSI 과매수
        if sell_conditions.get('rsi_overbought', False):
            if current['RSI'] >= 70:
                return True
        
        # 볼린저밴드 상단
        if sell_conditions.get('bb_upper', False):
            if current['Close'] >= current['BB_Upper'] * 0.98:
                return True
        
        return False
    
    def analyze_results(self, portfolio, trade_log, initial_capital):
        """백테스팅 결과 분석 (기존 코드 유지)"""
        if not trade_log:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'message': '거래가 없었습니다.'
            }
        
        df_trades = pd.DataFrame(trade_log)
        
        # 기본 통계
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['profit'] > 0])
        win_rate = winning_trades / total_trades * 100
        
        total_profit = df_trades['profit'].sum()
        total_return = total_profit / initial_capital * 100
        
        avg_profit = df_trades['profit'].mean()
        avg_holding_days = df_trades['holding_days'].mean()
        
        # 최고/최악 거래
        best_trade = df_trades.loc[df_trades['profit'].idxmax()]
        worst_trade = df_trades.loc[df_trades['profit'].idxmin()]
        
        results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_return': total_return,
            'avg_profit': avg_profit,
            'avg_holding_days': avg_holding_days,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'final_portfolio_value': portfolio.total_value()
        }
        
        return results


class Portfolio:
    """포트폴리오 관리 (기존 코드 유지)"""
    
    def __init__(self, initial_capital):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.holdings = {}  # {symbol: {'shares': int, 'avg_price': float}}
        self.transaction_log = []
    
    def buy(self, symbol, shares, price, date):
        """매수"""
        cost = shares * price
        if self.cash >= cost:
            self.cash -= cost
            
            if symbol in self.holdings:
                # 기존 보유량과 평균 단가 계산
                old_shares = self.holdings[symbol]['shares']
                old_avg_price = self.holdings[symbol]['avg_price']
                
                new_shares = old_shares + shares
                new_avg_price = ((old_shares * old_avg_price) + (shares * price)) / new_shares
                
                self.holdings[symbol] = {
                    'shares': new_shares,
                    'avg_price': new_avg_price
                }
            else:
                self.holdings[symbol] = {
                    'shares': shares,
                    'avg_price': price
                }
            
            self.transaction_log.append({
                'date': date,
                'action': 'BUY',
                'symbol': symbol,
                'shares': shares,
                'price': price,
                'amount': cost
            })
    
    def sell(self, symbol, shares, price, date):
        """매도"""
        if symbol in self.holdings and self.holdings[symbol]['shares'] >= shares:
            revenue = shares * price
            self.cash += revenue
            
            self.holdings[symbol]['shares'] -= shares
            if self.holdings[symbol]['shares'] == 0:
                del self.holdings[symbol]
            
            self.transaction_log.append({
                'date': date,
                'action': 'SELL',
                'symbol': symbol,
                'shares': shares,
                'price': price,
                'amount': revenue
            })
    
    def total_value(self, current_prices=None):
        """포트폴리오 총 가치"""
        if current_prices is None:
            # 현재가를 모르면 현금만 반환
            return self.cash
        
        total = self.cash
        for symbol, holding in self.holdings.items():
            if symbol in current_prices:
                total += holding['shares'] * current_prices[symbol]
        
        return total


class BacktestingDialog(QDialog):
    """백테스팅 다이얼로그 (기존 + 추천도 백테스팅 추가)"""
    
    def __init__(self, stock_screener, parent=None):
        super().__init__(parent)
        self.stock_screener = stock_screener
        self.setWindowTitle('📈 백테스팅 - 전략 성과 검증')
        self.setGeometry(200, 200, 1000, 800)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 상단 설명
        info_label = QLabel(
            "💡 과거 데이터로 매수/매도 전략의 효과를 검증할 수 있습니다.\n"
            "🎯 새로운 기능: 특정 시점에서 추천도가 가장 높은 종목에 투자했다면 현재 수익은?"
        )
        info_label.setStyleSheet("color: #666; padding: 15px; background-color: #f9f9f9; border-radius: 8px; font-size: 13px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 탭 위젯 생성
        tab_widget = QTabWidget()
        
        # 탭 1: 추천도 기반 백테스팅
        recommendation_tab = self.create_recommendation_tab()
        tab_widget.addTab(recommendation_tab, "🎯 추천도 기반 백테스팅")
        
        # 탭 2: 기존 백테스팅
        traditional_tab = self.create_traditional_tab()
        tab_widget.addTab(traditional_tab, "📊 전통적 백테스팅")
        
        layout.addWidget(tab_widget)
        
        # 결과 표시 영역 (공통)
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(250)
        self.results_text.setPlaceholderText("백테스팅 결과가 여기에 표시됩니다...")
        layout.addWidget(self.results_text)
        
        # 닫기 버튼
        close_btn = QPushButton("❌ 닫기")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
    
    def create_recommendation_tab(self):
        """추천도 기반 백테스팅 탭"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 설명
        desc_label = QLabel(
            "🎯 특정 시점에서 스크리닝 조건을 만족하는 종목 중 추천도가 가장 높은 종목을 선택하여\n"
            "그 종목에 투자했다면 현재까지 얼마의 수익률을 거뒀는지 계산합니다."
        )
        desc_label.setStyleSheet("color: #444; padding: 10px; background-color: #e8f5e8; border-radius: 5px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # 설정 그룹
        settings_group = QGroupBox("분석 설정")
        settings_layout = QGridLayout()
        
        # 분석 기간
        settings_layout.addWidget(QLabel("분석 기준일:"), 0, 0)
        self.rec_period_combo = QComboBox()
        self.rec_period_combo.addItems([
            "3개월 전",
            "6개월 전",
            "9개월 전",
            "1년 전",
            "2년 전"
        ])
        self.rec_period_combo.setCurrentText("6개월 전")
        settings_layout.addWidget(self.rec_period_combo, 0, 1)
        
        # 최소 추천도
        settings_layout.addWidget(QLabel("최소 추천도:"), 1, 0)
        self.min_score_spin = QSpinBox()
        self.min_score_spin.setRange(50, 100)
        self.min_score_spin.setValue(75)
        self.min_score_spin.setSuffix("점")
        settings_layout.addWidget(self.min_score_spin, 1, 1)
        
        # 분석 대상
        settings_layout.addWidget(QLabel("분석 종목:"), 2, 0)
        self.rec_stocks_combo = QComboBox()
        self.rec_stocks_combo.addItems([
            "현재 로드된 전체 종목",
            "한국 종목만",
            "미국 종목만",
            "스웨덴 종목만"
        ])
        settings_layout.addWidget(self.rec_stocks_combo, 2, 1)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # 실행 버튼
        self.rec_run_btn = QPushButton("🚀 추천도 백테스팅 실행")
        self.rec_run_btn.clicked.connect(self.run_recommendation_backtest)
        self.rec_run_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 12px; }")
        layout.addWidget(self.rec_run_btn)
        
        widget.setLayout(layout)
        return widget
    
    def create_traditional_tab(self):
        """기존 백테스팅 탭"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 설정 패널
        settings_group = QGroupBox("백테스팅 설정")
        settings_layout = QGridLayout()
        
        # 기간 설정
        settings_layout.addWidget(QLabel("백테스팅 기간:"), 0, 0)
        
        period_layout = QHBoxLayout()
        self.period_combo = QComboBox()
        self.period_combo.addItems([
            "3개월 (최근 3개월간)",
            "6개월 (최근 6개월간)", 
            "1년 (최근 1년간)",
            "사용자 정의"
        ])
        self.period_combo.setCurrentText("6개월 (최근 6개월간)")
        period_layout.addWidget(self.period_combo)
        
        # 사용자 정의 날짜 (초기에는 비활성화)
        self.start_date = QDateEdit()
        self.start_date.setDate(datetime.now().date() - timedelta(days=180))
        self.start_date.setEnabled(False)
        period_layout.addWidget(QLabel("시작:"))
        period_layout.addWidget(self.start_date)
        
        self.end_date = QDateEdit()
        self.end_date.setDate(datetime.now().date())
        self.end_date.setEnabled(False)
        period_layout.addWidget(QLabel("종료:"))
        period_layout.addWidget(self.end_date)
        
        settings_layout.addLayout(period_layout, 0, 1, 1, 2)
        
        # 초기 자본
        settings_layout.addWidget(QLabel("초기 자본:"), 1, 0)
        self.capital_spin = QSpinBox()
        self.capital_spin.setRange(10000, 10000000)
        self.capital_spin.setValue(100000)
        self.capital_spin.setSuffix(" 원")
        settings_layout.addWidget(self.capital_spin, 1, 1)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # 조건 설정
        conditions_group = QGroupBox("테스트할 조건 선택")
        conditions_layout = QHBoxLayout()
        
        # 매수 조건
        buy_group = QGroupBox("매수 조건")
        buy_layout = QVBoxLayout()
        
        self.buy_ma_cross = QCheckBox("60일선이 120일선 돌파")
        self.buy_rsi_oversold = QCheckBox("RSI 과매도 반등 (30 돌파)")
        self.buy_bb_touch = QCheckBox("볼린저밴드 하단 터치")
        self.buy_macd_cross = QCheckBox("MACD 골든크로스")
        
        buy_layout.addWidget(self.buy_ma_cross)
        buy_layout.addWidget(self.buy_rsi_oversold)
        buy_layout.addWidget(self.buy_bb_touch)
        buy_layout.addWidget(self.buy_macd_cross)
        
        buy_group.setLayout(buy_layout)
        conditions_layout.addWidget(buy_group)
        
        # 매도 조건
        sell_group = QGroupBox("매도 조건")
        sell_layout = QVBoxLayout()
        
        self.sell_dead_cross = QCheckBox("데드크로스 (MA60 < MA120)")
        self.sell_rsi_overbought = QCheckBox("RSI 과매수 (>= 70)")
        self.sell_bb_upper = QCheckBox("볼린저밴드 상단")
        self.sell_stop_loss = QCheckBox("손절/익절 (-7% / +20%)")
        
        sell_layout.addWidget(self.sell_dead_cross)
        sell_layout.addWidget(self.sell_rsi_overbought)
        sell_layout.addWidget(self.sell_bb_upper)
        sell_layout.addWidget(self.sell_stop_loss)
        
        sell_group.setLayout(sell_layout)
        conditions_layout.addWidget(sell_group)
        
        conditions_group.setLayout(conditions_layout)
        layout.addWidget(conditions_group)
        
        # 실행 버튼
        self.run_btn = QPushButton("🚀 전통적 백테스팅 실행")
        self.run_btn.clicked.connect(self.run_traditional_backtest)
        self.run_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 12px; }")
        layout.addWidget(self.run_btn)
        
        widget.setLayout(layout)
        
        # 이벤트 연결
        self.period_combo.currentTextChanged.connect(self.on_period_changed)
        
        return widget
    
    def on_period_changed(self, text):
        """기간 선택 변경 시"""
        is_custom = "사용자 정의" in text
        self.start_date.setEnabled(is_custom)
        self.end_date.setEnabled(is_custom)
    
    def run_recommendation_backtest(self):
        """추천도 기반 백테스팅 실행"""
        try:
            self.rec_run_btn.setEnabled(False)
            self.rec_run_btn.setText("🔄 분석 중...")
            QApplication.processEvents()
            
            # 기간 설정
            period_text = self.rec_period_combo.currentText()
            if "3개월" in period_text:
                months_back = 3
            elif "6개월" in period_text:
                months_back = 6
            elif "9개월" in period_text:
                months_back = 9
            elif "1년" in period_text:
                months_back = 12
            elif "2년" in period_text:
                months_back = 24
            else:
                months_back = 6
            
            # 최소 추천도
            min_score = self.min_score_spin.value()
            
            # 분석 대상 종목 선택
            stock_selection = self.rec_stocks_combo.currentText()
            symbols = self.get_symbols_for_analysis(stock_selection)
            
            if not symbols:
                QMessageBox.warning(self, "경고", "분석할 종목이 없습니다. 먼저 샘플 생성을 해주세요.")
                return
            
            # 추천도 백테스팅 실행
            engine = RecommendationBacktestingEngine(self.stock_screener.technical_analyzer)
            result = engine.run_recommendation_backtest(symbols, months_back, min_score)
            
            # 결과 표시
            self.display_recommendation_results(result)
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"추천도 백테스팅 중 오류 발생: {str(e)}")
        
        finally:
            self.rec_run_btn.setEnabled(True)
            self.rec_run_btn.setText("🚀 추천도 백테스팅 실행")
    
    def run_traditional_backtest(self):
        """기존 백테스팅 실행"""
        try:
            self.run_btn.setEnabled(False)
            self.run_btn.setText("🔄 실행 중...")
            QApplication.processEvents()
            
            # 기간 설정
            if "3개월" in self.period_combo.currentText():
                start_date = datetime.now() - timedelta(days=90)
            elif "6개월" in self.period_combo.currentText():
                start_date = datetime.now() - timedelta(days=180)
            elif "1년" in self.period_combo.currentText():
                start_date = datetime.now() - timedelta(days=365)
            else:
                start_date = self.start_date.date().toPython()
            
            end_date = datetime.now()
            
            # 조건 설정
            buy_conditions = {
                'ma_cross': self.buy_ma_cross.isChecked(),
                'rsi_oversold': self.buy_rsi_oversold.isChecked(),
                'bb_touch': self.buy_bb_touch.isChecked(),
                'macd_cross': self.buy_macd_cross.isChecked()
            }
            
            sell_conditions = {
                'dead_cross': self.sell_dead_cross.isChecked(),
                'rsi_overbought': self.sell_rsi_overbought.isChecked(),
                'bb_upper': self.sell_bb_upper.isChecked()
            }
            
            # 종목 리스트 (현재 로드된 종목들 사용)
            symbols = []
            for market_stocks in self.stock_screener.stock_lists.values():
                symbols.extend([stock['ticker'] for stock in market_stocks[:10]])  # 각 시장에서 10개씩
            
            if not symbols:
                QMessageBox.warning(self, "경고", "종목 리스트가 없습니다. 먼저 샘플 생성을 해주세요.")
                return
            
            # 백테스팅 실행
            engine = BacktestingEngine(self.stock_screener.technical_analyzer)
            results, trade_log = engine.run_backtest(
                symbols, buy_conditions, sell_conditions,
                start_date, end_date, self.capital_spin.value()
            )
            
            # 결과 표시
            self.display_traditional_results(results, trade_log)
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"백테스팅 중 오류 발생: {str(e)}")
        
        finally:
            self.run_btn.setEnabled(True)
            self.run_btn.setText("🚀 전통적 백테스팅 실행")
    
    def get_symbols_for_analysis(self, selection):
        """분석 대상 종목 리스트 가져오기"""
        symbols = []
        
        if "전체" in selection:
            for market_stocks in self.stock_screener.stock_lists.values():
                symbols.extend([stock['ticker'] for stock in market_stocks])
        elif "한국" in selection:
            if 'korea' in self.stock_screener.stock_lists:
                symbols = [stock['ticker'] for stock in self.stock_screener.stock_lists['korea']]
        elif "미국" in selection:
            if 'usa' in self.stock_screener.stock_lists:
                symbols = [stock['ticker'] for stock in self.stock_screener.stock_lists['usa']]
        elif "스웨덴" in selection:
            if 'sweden' in self.stock_screener.stock_lists:
                symbols = [stock['ticker'] for stock in self.stock_screener.stock_lists['sweden']]
        
        return symbols
    
    def display_recommendation_results(self, result):
        """추천도 백테스팅 결과 표시"""
        if result['status'] == 'no_candidates':
            self.results_text.setText(
                f"❌ 분석 결과\n\n"
                f"분석 종목 수: {result['total_analyzed']}개\n"
                f"최소 추천도: {result['min_score_required']}점\n\n"
                f"조건을 만족하는 종목이 없었습니다.\n"
                f"추천도 기준을 낮추거나 분석 기간을 변경해보세요."
            )
            return
        
        if result['status'] == 'price_error':
            self.results_text.setText(
                f"❌ 오류 발생\n\n"
                f"선택된 종목: {result['selected_stock']['symbol']}\n"
                f"{result['message']}"
            )
            return
        
        if result['status'] == 'success':
            stock = result['selected_stock']
            perf = result['performance']
            
            # 성과 평가
            if perf['return_rate'] >= 20:
                performance_emoji = "🏆"
                performance_text = "훌륭한 성과!"
            elif perf['return_rate'] >= 10:
                performance_emoji = "👍"
                performance_text = "좋은 성과!"
            elif perf['return_rate'] >= 0:
                performance_emoji = "😊"
                performance_text = "수익 달성!"
            else:
                performance_emoji = "😔"
                performance_text = "손실 발생"
            
            result_text = f"""
🎯 추천도 기반 백테스팅 결과

📊 분석 정보:
- 분석 기준일: {result['analysis_date'].strftime('%Y-%m-%d')} ({result['months_back']}개월 전)
- 분석 종목 수: {result['total_analyzed']}개
- 매수 후보: {result['total_candidates']}개 종목

🏆 선택된 종목:
- 종목: {stock['symbol']}
- 매수일: {stock['entry_date'].strftime('%Y-%m-%d')}
- 당시 추천도: {stock['recommendation_score']}점
- 매수가: {perf['entry_price']:,.0f}원

📈 투자 성과:
- 현재가: {perf['current_price']:,.0f}원
- 수익률: {perf['return_rate']:+.2f}%
- 연환산 수익률: {perf['annual_return']:+.2f}%
- 보유기간: {perf['holding_period']}일

💰 투자 시뮬레이션:
- 100만원 투자 → {((perf['current_price'] / perf['entry_price']) * 1000000):,.0f}원
  (손익: {((perf['current_price'] / perf['entry_price'] - 1) * 1000000):+,.0f}원)
- 1000주 투자 → {(perf['current_price'] * 1000):,.0f}원
  (손익: {(perf['profit_loss_amount'] * 1000):+,.0f}원)

{performance_emoji} 성과 평가: {performance_text}
            """.strip()
            
            self.results_text.setText(result_text)
            
            # 다른 후보들 정보
            if len(result['other_candidates']) > 1:
                other_info = "\n\n📋 다른 매수 후보들 (상위 5개):\n"
                for i, candidate in enumerate(result['other_candidates'][:5]):
                    if candidate['symbol'] != stock['symbol']:
                        other_info += f"{i+1}. {candidate['symbol']} (추천도: {candidate['recommendation_score']}점)\n"
                
                self.results_text.append(other_info)
            
            # 엑셀 저장 옵션
            reply = QMessageBox.question(
                self, "결과 저장", 
                f"백테스팅이 완료되었습니다!\n"
                f"선택 종목: {stock['symbol']}\n"
                f"수익률: {perf['return_rate']:+.2f}%\n\n"
                f"결과를 엑셀 파일로 저장하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.save_recommendation_results(result)
    
    def display_traditional_results(self, results, trade_log):
        """기존 백테스팅 결과 표시"""
        if results['total_trades'] == 0:
            self.results_text.setText(
                "❌ 백테스팅 기간 중 매수 조건에 맞는 거래가 없었습니다.\n"
                "조건을 완화하거나 기간을 늘려보세요."
            )
            return
        
        # 결과 포맷팅
        result_text = f"""
📈 전통적 백테스팅 결과 요약

💰 수익 성과:
- 초기 자본: {self.capital_spin.value():,}원
- 총 수익: {results['total_profit']:,.0f}원
- 수익률: {results['total_return']:.2f}%

📊 거래 통계:
- 총 거래: {results['total_trades']}건
- 성공 거래: {results['winning_trades']}건
- 실패 거래: {results['losing_trades']}건  
- 승률: {results['win_rate']:.1f}%

📅 평균 보유기간: {results['avg_holding_days']:.1f}일
💵 평균 거래당 수익: {results['avg_profit']:,.0f}원

🏆 최고 거래: {results['best_trade']['symbol']} ({results['best_trade']['profit_rate']*100:.1f}%, {results['best_trade']['profit']:,.0f}원)
📉 최악 거래: {results['worst_trade']['symbol']} ({results['worst_trade']['profit_rate']*100:.1f}%, {results['worst_trade']['profit']:,.0f}원)
        """
        
        self.results_text.setText(result_text.strip())
        
        # 상세 거래 로그 표시 옵션
        reply = QMessageBox.question(
            self, "상세 결과", 
            f"백테스팅이 완료되었습니다!\n"
            f"총 {results['total_trades']}건 거래, 수익률 {results['total_return']:.2f}%\n\n"
            f"상세 거래 내역을 엑셀 파일로 저장하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.save_trade_log(trade_log)
    
    def save_recommendation_results(self, result):
        """추천도 백테스팅 결과 저장"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"recommendation_backtest_{result['months_back']}m_{timestamp}.xlsx"
            
            stock = result['selected_stock']
            perf = result['performance']
            
            # 결과 데이터 준비
            summary_data = {
                '항목': [
                    '분석 기준일', '분석 기간', '분석 종목 수', '매수 후보 수',
                    '선택 종목', '당시 추천도', '매수일', '매수가',
                    '현재가', '수익률(%)', '연환산 수익률(%)', '보유기간(일)',
                    '100만원 투자 결과', '100만원 투자 손익', '1000주 투자 결과', '1000주 투자 손익'
                ],
                '값': [
                    result['analysis_date'].strftime('%Y-%m-%d'),
                    f"{result['months_back']}개월",
                    result['total_analyzed'],
                    result['total_candidates'],
                    stock['symbol'],
                    f"{stock['recommendation_score']}점",
                    stock['entry_date'].strftime('%Y-%m-%d'),
                    f"{perf['entry_price']:,.0f}원",
                    f"{perf['current_price']:,.0f}원",
                    f"{perf['return_rate']:+.2f}%",
                    f"{perf['annual_return']:+.2f}%",
                    f"{perf['holding_period']}일",
                    f"{((perf['current_price'] / perf['entry_price']) * 1000000):,.0f}원",
                    f"{((perf['current_price'] / perf['entry_price'] - 1) * 1000000):+,.0f}원",
                    f"{(perf['current_price'] * 1000):,.0f}원",
                    f"{(perf['profit_loss_amount'] * 1000):+,.0f}원"
                ]
            }
            
            # 다른 후보들 데이터
            candidates_data = []
            for i, candidate in enumerate(result['other_candidates'][:10]):  # 상위 10개
                candidates_data.append({
                    '순위': i + 1,
                    '종목': candidate['symbol'],
                    '추천도': f"{candidate['recommendation_score']}점",
                    '매수가': f"{candidate['entry_price']:,.0f}원",
                    '매수일': candidate['entry_date'].strftime('%Y-%m-%d')
                })
            
            # 엑셀 저장
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='백테스팅 결과', index=False)
                if candidates_data:
                    pd.DataFrame(candidates_data).to_excel(writer, sheet_name='매수 후보 목록', index=False)
            
            QMessageBox.information(
                self, "저장 완료", 
                f"백테스팅 결과가 {filename} 파일로 저장되었습니다."
            )
            
        except Exception as e:
            QMessageBox.warning(self, "저장 실패", f"파일 저장 중 오류: {str(e)}")
    
    def save_trade_log(self, trade_log):
        """거래 로그를 엑셀로 저장 (기존 코드)"""
        try:
            if not trade_log:
                return
                
            filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            df = pd.DataFrame(trade_log)
            df['buy_date'] = df['buy_date'].dt.strftime('%Y-%m-%d')
            df['sell_date'] = df['sell_date'].dt.strftime('%Y-%m-%d')
            df['profit_rate'] = df['profit_rate'] * 100  # 백분율로 변환
            
            # 컬럼명 한글화
            df.columns = [
                '종목', '매수일', '매도일', '매수가', '매도가', 
                '수량', '수익금', '수익률(%)', '보유일수', '매도사유'
            ]
            
            df.to_excel(filename, index=False)
            
            QMessageBox.information(
                self, "저장 완료", 
                f"거래 내역이 {filename} 파일로 저장되었습니다."
            )
            
        except Exception as e:
            QMessageBox.warning(self, "저장 실패", f"파일 저장 중 오류: {str(e)}")