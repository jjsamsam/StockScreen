"""
expectancy_calculator.py
기대값(Expectancy) 계산 시스템

기대값(Expectancy)은 거래 전략의 장기적 수익성을 측정하는 지표입니다.

핵심 공식:
E = (P_win × Avg_win) - (P_loss × Avg_loss)

추가 지표:
- Profit Factor = Gross Profit / Gross Loss
- Kelly Criterion = W - [(1-W) / R]
- Risk-Reward Ratio
- Expected Return per Trade
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from logger_config import get_logger

logger = get_logger(__name__)


class ExpectancyCalculator:
    """
    기대값 계산기

    입력:
    - p_final: 예측 확률 (상승 확률)
    - positions: 포지션 데이터 (진입가, 청산가, 수량 등)
    - outcomes: 실제 결과 (수익/손실)

    출력:
    - Expectancy, Profit Factor, Kelly %, Risk-Reward Ratio 등
    """

    def __init__(self):
        self.trade_history = []

    def calculate_expectancy(self,
                             trades: pd.DataFrame,
                             profit_col: str = 'profit',
                             result_col: str = 'result') -> Dict[str, float]:
        """
        기대값 계산 (전체 거래 기반)

        Args:
            trades: 거래 데이터프레임
                - profit: 수익금 (양수/음수)
                - result: 'win' 또는 'loss'

        Returns:
            기대값 지표 딕셔너리
        """
        if len(trades) == 0:
            logger.warning("No trades to calculate expectancy")
            return self._get_default_expectancy()

        # 승리/패배 거래 분리
        winning_trades = trades[trades[result_col] == 'win']
        losing_trades = trades[trades[result_col] == 'loss']

        total_trades = len(trades)
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)

        # 승률
        win_rate = num_wins / total_trades if total_trades > 0 else 0.0

        # 평균 수익/손실
        avg_win = winning_trades[profit_col].mean() if num_wins > 0 else 0.0
        avg_loss = abs(losing_trades[profit_col].mean()) if num_losses > 0 else 0.0

        # 기대값
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # 리스크-리워드 비율
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        # Profit Factor
        gross_profit = winning_trades[profit_col].sum() if num_wins > 0 else 0.0
        gross_loss = abs(losing_trades[profit_col].sum()) if num_losses > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Kelly Criterion
        kelly_pct = self._calculate_kelly(win_rate, risk_reward_ratio)

        # 거래당 기대 수익률 (%)
        avg_position_size = trades.get('position_value', trades[profit_col]).mean()
        expectancy_pct = (expectancy / avg_position_size * 100) if avg_position_size != 0 else 0.0

        # System Quality Number (SQN)
        sqn = self._calculate_sqn(trades[profit_col].values)

        results = {
            'expectancy': expectancy,
            'expectancy_pct': expectancy_pct,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'risk_reward_ratio': risk_reward_ratio,
            'profit_factor': profit_factor,
            'kelly_pct': kelly_pct,
            'total_trades': total_trades,
            'num_wins': num_wins,
            'num_losses': num_losses,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'sqn': sqn
        }

        return results

    def calculate_expectancy_by_probability(self,
                                            predictions: np.ndarray,
                                            actual_outcomes: np.ndarray,
                                            profit_per_trade: np.ndarray,
                                            threshold: float = 0.5) -> Dict[str, float]:
        """
        예측 확률 기반 기대값 계산

        Args:
            predictions: 예측 확률 배열 (0~1)
            actual_outcomes: 실제 결과 (1: 상승, 0: 하락)
            profit_per_trade: 거래당 수익 (양수/음수)
            threshold: 진입 임계값

        Returns:
            기대값 지표
        """
        # 임계값 이상만 거래
        trade_mask = predictions >= threshold

        if not np.any(trade_mask):
            logger.warning(f"No trades above threshold {threshold}")
            return self._get_default_expectancy()

        filtered_outcomes = actual_outcomes[trade_mask]
        filtered_profits = profit_per_trade[trade_mask]

        # 승/패 분리
        wins = filtered_profits[filtered_outcomes == 1]
        losses = filtered_profits[filtered_outcomes == 0]

        # 기대값 계산
        trades_df = pd.DataFrame({
            'profit': filtered_profits,
            'result': ['win' if outcome == 1 else 'loss' for outcome in filtered_outcomes]
        })

        return self.calculate_expectancy(trades_df)

    def calculate_position_expectancy(self,
                                      p_final: float,
                                      entry_price: float,
                                      stop_loss: float,
                                      take_profit: float,
                                      position_size: float = 1.0) -> Dict[str, float]:
        """
        개별 포지션의 기대값 계산

        Args:
            p_final: 상승 확률
            entry_price: 진입가
            stop_loss: 손절가
            take_profit: 목표가
            position_size: 포지션 크기

        Returns:
            포지션 기대값 지표
        """
        # 수익/손실 계산
        potential_profit = (take_profit - entry_price) * position_size
        potential_loss = abs((stop_loss - entry_price) * position_size)

        # 기대값
        expectancy = (p_final * potential_profit) - ((1 - p_final) * potential_loss)

        # 리스크-리워드 비율
        risk_reward_ratio = potential_profit / potential_loss if potential_loss > 0 else 0.0

        # Kelly 비율
        kelly_pct = self._calculate_kelly(p_final, risk_reward_ratio)

        # 기대 수익률
        expectancy_pct = (expectancy / (entry_price * position_size)) * 100 if entry_price > 0 else 0.0

        return {
            'expectancy': expectancy,
            'expectancy_pct': expectancy_pct,
            'potential_profit': potential_profit,
            'potential_loss': potential_loss,
            'risk_reward_ratio': risk_reward_ratio,
            'kelly_pct': kelly_pct,
            'recommended_position_size': position_size * kelly_pct * 0.25  # Kelly의 25%만 사용
        }

    def _calculate_kelly(self, win_rate: float, risk_reward_ratio: float) -> float:
        """
        Kelly Criterion 계산

        K = W - [(1-W) / R]
        W: 승률, R: 리스크-리워드 비율
        """
        if risk_reward_ratio <= 0:
            return 0.0

        kelly = win_rate - ((1 - win_rate) / risk_reward_ratio)
        kelly = max(0.0, min(1.0, kelly))  # 0~1 사이로 클리핑

        return kelly

    def _calculate_sqn(self, profits: np.ndarray) -> float:
        """
        System Quality Number (SQN) 계산

        SQN = sqrt(N) * (평균 수익 / 수익 표준편차)

        해석:
        - SQN < 1.6: Poor
        - 1.6 <= SQN < 2.0: Average
        - 2.0 <= SQN < 2.5: Good
        - 2.5 <= SQN < 3.0: Excellent
        - SQN >= 3.0: Superb
        """
        if len(profits) == 0:
            return 0.0

        n = len(profits)
        mean_profit = np.mean(profits)
        std_profit = np.std(profits)

        if std_profit == 0:
            return 0.0

        sqn = np.sqrt(n) * (mean_profit / std_profit)

        return float(sqn)

    def _get_default_expectancy(self) -> Dict[str, float]:
        """기본 기대값 (거래 없음)"""
        return {
            'expectancy': 0.0,
            'expectancy_pct': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'risk_reward_ratio': 0.0,
            'profit_factor': 0.0,
            'kelly_pct': 0.0,
            'total_trades': 0,
            'num_wins': 0,
            'num_losses': 0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'sqn': 0.0
        }

    def calculate_breakeven_win_rate(self, risk_reward_ratio: float) -> float:
        """
        손익분기 승률 계산

        공식: Breakeven WR = 1 / (1 + R)
        """
        if risk_reward_ratio <= 0:
            return 1.0

        breakeven_wr = 1.0 / (1.0 + risk_reward_ratio)
        return breakeven_wr

    def optimize_threshold(self,
                           predictions: np.ndarray,
                           actual_outcomes: np.ndarray,
                           profit_per_trade: np.ndarray,
                           thresholds: Optional[List[float]] = None) -> Tuple[float, Dict[str, float]]:
        """
        최적 임계값 찾기 (기대값 최대화)

        Args:
            predictions: 예측 확률
            actual_outcomes: 실제 결과
            profit_per_trade: 거래당 수익
            thresholds: 테스트할 임계값 리스트

        Returns:
            (최적 임계값, 기대값 지표)
        """
        if thresholds is None:
            thresholds = np.arange(0.5, 0.9, 0.05)

        best_threshold = 0.5
        best_expectancy = float('-inf')
        best_results = None

        for threshold in thresholds:
            try:
                results = self.calculate_expectancy_by_probability(
                    predictions, actual_outcomes, profit_per_trade, threshold
                )

                if results['expectancy'] > best_expectancy:
                    best_expectancy = results['expectancy']
                    best_threshold = threshold
                    best_results = results

            except Exception as e:
                logger.debug(f"Threshold {threshold} failed: {e}")
                continue

        logger.info(f"Optimal threshold: {best_threshold:.2f} (Expectancy: {best_expectancy:.2f})")

        return best_threshold, best_results

    def generate_expectancy_report(self, results: Dict[str, float]) -> str:
        """
        기대값 리포트 생성
        """
        report = f"""
╔══════════════════════════════════════════════════════════╗
║               Expectancy Analysis Report                ║
╚══════════════════════════════════════════════════════════╝

📊 Core Metrics:
  • Expectancy (per trade):     {results['expectancy']:>12,.2f}
  • Expectancy (%):              {results['expectancy_pct']:>12.2f}%
  • System Quality Number (SQN): {results['sqn']:>12.2f}

💰 Profitability:
  • Win Rate:                    {results['win_rate']:>12.1%}
  • Avg Win:                     {results['avg_win']:>12,.2f}
  • Avg Loss:                    {results['avg_loss']:>12,.2f}
  • Risk-Reward Ratio:           {results['risk_reward_ratio']:>12.2f}
  • Profit Factor:               {results['profit_factor']:>12.2f}

📈 Trade Statistics:
  • Total Trades:                {results['total_trades']:>12,}
  • Winning Trades:              {results['num_wins']:>12,}
  • Losing Trades:               {results['num_losses']:>12,}
  • Gross Profit:                {results['gross_profit']:>12,.2f}
  • Gross Loss:                  {results['gross_loss']:>12,.2f}

🎯 Position Sizing:
  • Kelly Criterion:             {results['kelly_pct']:>12.1%}
  • Recommended (25% Kelly):     {results['kelly_pct'] * 0.25:>12.1%}

📋 Quality Assessment:
"""
        # SQN 평가
        sqn = results['sqn']
        if sqn >= 3.0:
            report += "  • SQN Rating: ⭐⭐⭐⭐⭐ Superb\n"
        elif sqn >= 2.5:
            report += "  • SQN Rating: ⭐⭐⭐⭐ Excellent\n"
        elif sqn >= 2.0:
            report += "  • SQN Rating: ⭐⭐⭐ Good\n"
        elif sqn >= 1.6:
            report += "  • SQN Rating: ⭐⭐ Average\n"
        else:
            report += "  • SQN Rating: ⭐ Poor\n"

        # Profit Factor 평가
        pf = results['profit_factor']
        if pf >= 2.0:
            report += "  • Profit Factor: Excellent (≥2.0)\n"
        elif pf >= 1.5:
            report += "  • Profit Factor: Good (≥1.5)\n"
        elif pf >= 1.0:
            report += "  • Profit Factor: Profitable (≥1.0)\n"
        else:
            report += "  • Profit Factor: Losing (<1.0)\n"

        # 손익분기 승률
        breakeven_wr = self.calculate_breakeven_win_rate(results['risk_reward_ratio'])
        report += f"  • Breakeven Win Rate:        {breakeven_wr:>12.1%}\n"
        report += f"  • Margin Above Breakeven:    {(results['win_rate'] - breakeven_wr):>12.1%}\n"

        report += "\n" + "═" * 58 + "\n"

        return report

    def add_trade(self, profit: float, result: str, metadata: Optional[Dict] = None):
        """
        거래 기록 추가

        Args:
            profit: 수익금
            result: 'win' 또는 'loss'
            metadata: 추가 정보 (진입가, 청산가 등)
        """
        trade = {
            'profit': profit,
            'result': result,
            'timestamp': pd.Timestamp.now()
        }

        if metadata:
            trade.update(metadata)

        self.trade_history.append(trade)

    def get_trade_history_df(self) -> pd.DataFrame:
        """거래 히스토리 데이터프레임 반환"""
        if not self.trade_history:
            return pd.DataFrame()

        return pd.DataFrame(self.trade_history)


# === 사용 예시 ===
if __name__ == '__main__':
    calc = ExpectancyCalculator()

    # 예시 1: 과거 거래 데이터로 기대값 계산
    trades = pd.DataFrame({
        'profit': [1000, -300, 800, -200, 1200, -400, 500, -250, 1500, -350],
        'result': ['win', 'loss', 'win', 'loss', 'win', 'loss', 'win', 'loss', 'win', 'loss']
    })

    expectancy_results = calc.calculate_expectancy(trades)
    print(calc.generate_expectancy_report(expectancy_results))

    # 예시 2: 개별 포지션 기대값
    position_expectancy = calc.calculate_position_expectancy(
        p_final=0.65,
        entry_price=100,
        stop_loss=95,
        take_profit=110,
        position_size=100
    )

    print("Position Expectancy:")
    for key, value in position_expectancy.items():
        print(f"  {key}: {value:.2f}")

    # 예시 3: 최적 임계값 찾기
    predictions = np.random.rand(100)
    actual_outcomes = (np.random.rand(100) > 0.5).astype(int)
    profit_per_trade = np.where(actual_outcomes == 1,
                                 np.random.rand(100) * 1000,
                                 -np.random.rand(100) * 500)

    optimal_threshold, optimal_results = calc.optimize_threshold(
        predictions, actual_outcomes, profit_per_trade
    )

    print(f"\nOptimal Threshold: {optimal_threshold:.2f}")
    print(f"Expected Profit per Trade: {optimal_results['expectancy']:.2f}")
