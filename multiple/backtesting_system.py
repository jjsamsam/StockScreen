"""
backtesting.py
백테스팅 시스템 - 매수/매도 전략 검증
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


class BacktestingEngine:
    """백테스팅 엔진 - 전략 성과 검증"""
    
    def __init__(self, technical_analyzer):
        self.technical_analyzer = technical_analyzer
        self.results = []
    
    def run_backtest(self, symbols, buy_conditions, sell_conditions, 
                    start_date, end_date, initial_capital=100000):
        """
        백테스팅 실행
        
        예시 사용법:
        - start_date: 6개월 전 (2024-03-01)
        - end_date: 현재 (2024-09-01)  
        - symbols: ['AAPL', 'MSFT', '005930.KS']
        - buy_conditions: 선택된 매수 조건들
        - sell_conditions: 선택된 매도 조건들
        """
        
        print(f"🔄 백테스팅 시작: {start_date} ~ {end_date}")
        print(f"💰 초기 자본: ${initial_capital:,}")
        
        portfolio = Portfolio(initial_capital)
        trade_log = []
        
        # 각 종목별로 백테스팅 수행
        for symbol in symbols:
            try:
                print(f"\n📊 {symbol} 분석 중...")
                
                # 과거 데이터 다운로드 (백테스팅 기간 + 여유분)
                data_start = start_date - timedelta(days=180)  # 지표 계산용 여유
                
                stock = yf.Ticker(symbol)
                data = stock.history(start=data_start, end=end_date)
                
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
        """개별 종목 거래 시뮬레이션"""
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
        """매수 신호 체크"""
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
        """매도 신호 체크"""
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
        """백테스팅 결과 분석"""
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
    """포트폴리오 관리"""
    
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
    """백테스팅 다이얼로그"""
    
    def __init__(self, stock_screener, parent=None):
        super().__init__(parent)
        self.stock_screener = stock_screener
        self.setWindowTitle('📈 백테스팅 - 전략 성과 검증')
        self.setGeometry(200, 200, 900, 700)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 상단 설명
        info_label = QLabel(
            "💡 과거 데이터로 매수/매도 전략의 효과를 검증할 수 있습니다.\n"
            "예시: 6개월 전 매수 조건에 맞는 종목을 매수했다면 현재 수익은?"
        )
        info_label.setStyleSheet("color: #666; padding: 15px; background-color: #f9f9f9; border-radius: 8px; font-size: 13px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
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
        button_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("🚀 백테스팅 실행")
        self.run_btn.clicked.connect(self.run_backtest)
        self.run_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 12px; }")
        button_layout.addWidget(self.run_btn)
        
        self.close_btn = QPushButton("❌ 닫기")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # 결과 표시 영역
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setPlaceholderText("백테스팅 결과가 여기에 표시됩니다...")
        layout.addWidget(self.results_text)
        
        self.setLayout(layout)
        
        # 이벤트 연결
        self.period_combo.currentTextChanged.connect(self.on_period_changed)
    
    def on_period_changed(self, text):
        """기간 선택 변경 시"""
        is_custom = "사용자 정의" in text
        self.start_date.setEnabled(is_custom)
        self.end_date.setEnabled(is_custom)
    
    def run_backtest(self):
        """백테스팅 실행"""
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
            self.display_results(results, trade_log)
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"백테스팅 중 오류 발생: {str(e)}")
        
        finally:
            self.run_btn.setEnabled(True)
            self.run_btn.setText("🚀 백테스팅 실행")
    
    def display_results(self, results, trade_log):
        """결과 표시"""
        if results['total_trades'] == 0:
            self.results_text.setText(
                "❌ 백테스팅 기간 중 매수 조건에 맞는 거래가 없었습니다.\n"
                "조건을 완화하거나 기간을 늘려보세요."
            )
            return
        
        # 결과 포맷팅
        result_text = f"""
📈 백테스팅 결과 요약

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
    
    def save_trade_log(self, trade_log):
        """거래 로그를 엑셀로 저장"""
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


# screener.py에 추가할 메소드
def open_backtesting_dialog(self):
    """백테스팅 다이얼로그 열기"""
    dialog = BacktestingDialog(self)
    dialog.exec_()