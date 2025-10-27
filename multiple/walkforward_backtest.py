"""
walkforward_backtest.py
워크포워드 백테스팅 시스템

워크포워드 분석은 시계열 데이터에서 가장 현실적인 백테스팅 방법입니다.

프로세스:
1. 전체 기간을 여러 윈도우로 분할
2. 각 윈도우마다:
   - Training Period: 모델 학습/최적화
   - Testing Period: 실전 시뮬레이션
3. Out-of-Sample 성능 집계

예시:
데이터: 2020-01-01 ~ 2024-12-31 (5년)
윈도우: 6개월 학습 + 1개월 테스트

Window 1: Train 2020-01~06, Test 2020-07
Window 2: Train 2020-02~07, Test 2020-08
...
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
from logger_config import get_logger

logger = get_logger(__name__)


class WalkForwardBacktest:
    """
    워크포워드 백테스팅 엔진

    특징:
    - 고정 윈도우 (Fixed Window)
    - 확장 윈도우 (Expanding Window)
    - 앵커드 윈도우 (Anchored Window)
    """

    def __init__(self,
                 train_period_days: int = 180,
                 test_period_days: int = 30,
                 window_type: str = 'fixed',
                 min_train_samples: int = 100):
        """
        Args:
            train_period_days: 학습 기간 (일)
            test_period_days: 테스트 기간 (일)
            window_type: 'fixed', 'expanding', 'anchored'
            min_train_samples: 최소 학습 샘플 수
        """
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.window_type = window_type
        self.min_train_samples = min_train_samples

        self.windows = []
        self.results = []

        logger.info(f"WalkForwardBacktest initialized: {window_type} window, "
                   f"train={train_period_days}d, test={test_period_days}d")

    def generate_windows(self,
                         data: pd.DataFrame,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Dict]:
        """
        백테스팅 윈도우 생성

        Args:
            data: 전체 데이터 (날짜 인덱스)
            start_date: 백테스팅 시작일 (None이면 데이터 시작)
            end_date: 백테스팅 종료일 (None이면 데이터 끝)

        Returns:
            윈도우 리스트 [{'train_start', 'train_end', 'test_start', 'test_end'}, ...]
        """
        if start_date is None:
            start_date = data.index[0]
        if end_date is None:
            end_date = data.index[-1]

        windows = []
        anchor_date = start_date  # Anchored 모드용

        current_train_start = start_date

        while True:
            # 학습 기간 종료일
            train_end = current_train_start + timedelta(days=self.train_period_days)

            # 테스트 기간 시작/종료
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_period_days)

            # 종료 조건
            if test_end > end_date:
                break

            # 데이터에서 실제 날짜 찾기
            train_data = data.loc[current_train_start:train_end]
            test_data = data.loc[test_start:test_end]

            if len(train_data) < self.min_train_samples:
                logger.warning(f"Insufficient training data: {len(train_data)} < {self.min_train_samples}")
                break

            if len(test_data) == 0:
                logger.warning("No test data in window")
                break

            window = {
                'window_id': len(windows) + 1,
                'train_start': current_train_start,
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'train_samples': len(train_data),
                'test_samples': len(test_data)
            }

            windows.append(window)

            # 다음 윈도우로 이동
            if self.window_type == 'fixed':
                # 고정: 학습 시작일을 테스트 기간만큼 이동
                current_train_start = test_start

            elif self.window_type == 'expanding':
                # 확장: 학습 시작일 고정, 종료일만 이동
                # current_train_start는 그대로
                current_train_start = start_date  # 항상 처음부터
                # 실제로는 train_end를 test_end로 업데이트
                current_train_start = start_date
                # 다음 반복에서 train_end가 자동으로 늘어남
                # 실제 구현: train_end를 test_end로 설정
                self.train_period_days += self.test_period_days

            elif self.window_type == 'anchored':
                # 앵커드: 학습 시작일 고정, 테스트만 이동
                current_train_start = anchor_date
                # train_end를 test_end로 업데이트
                self.train_period_days = (test_end - anchor_date).days

            else:
                logger.warning(f"Unknown window type: {self.window_type}, using fixed")
                current_train_start = test_start

        self.windows = windows
        logger.info(f"Generated {len(windows)} windows")

        return windows

    def run_backtest(self,
                     data: pd.DataFrame,
                     train_func: Callable,
                     test_func: Callable,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        워크포워드 백테스팅 실행

        Args:
            data: 전체 데이터
            train_func: 학습 함수 (train_data) -> model
            test_func: 테스트 함수 (model, test_data) -> results_dict
            start_date: 시작일
            end_date: 종료일
            progress_callback: 진행 콜백 (current, total)

        Returns:
            결과 데이터프레임
        """
        # 윈도우 생성
        if not self.windows:
            self.generate_windows(data, start_date, end_date)

        self.results = []

        # 각 윈도우마다 학습/테스트
        for i, window in enumerate(self.windows):
            try:
                if progress_callback:
                    progress_callback(i + 1, len(self.windows))

                logger.info(f"Window {window['window_id']}/{len(self.windows)}: "
                           f"Train {window['train_start'].date()}~{window['train_end'].date()}, "
                           f"Test {window['test_start'].date()}~{window['test_end'].date()}")

                # 데이터 분할
                train_data = data.loc[window['train_start']:window['train_end']]
                test_data = data.loc[window['test_start']:window['test_end']]

                # 학습
                logger.debug("Training model...")
                model = train_func(train_data)

                # 테스트
                logger.debug("Testing model...")
                test_results = test_func(model, test_data)

                # 결과 저장
                window_result = {
                    **window,
                    **test_results
                }

                self.results.append(window_result)

                logger.info(f"Window {window['window_id']} complete: "
                           f"Return={test_results.get('return', 0):.2%}, "
                           f"Sharpe={test_results.get('sharpe', 0):.2f}")

            except Exception as e:
                logger.error(f"Window {window['window_id']} failed: {e}")
                continue

        # 결과 데이터프레임 생성
        results_df = pd.DataFrame(self.results)

        logger.info(f"Backtest complete: {len(self.results)}/{len(self.windows)} windows successful")

        return results_df

    def analyze_results(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        워크포워드 백테스팅 결과 분석

        Args:
            results_df: 백테스트 결과 데이터프레임

        Returns:
            종합 성과 지표
        """
        if len(results_df) == 0:
            logger.warning("No results to analyze")
            return {}

        # 전체 수익률 (복리)
        total_return = (1 + results_df['return']).prod() - 1

        # 평균 수익률
        avg_return = results_df['return'].mean()

        # 승률
        win_rate = (results_df['return'] > 0).sum() / len(results_df)

        # Sharpe Ratio (전체)
        if 'sharpe' in results_df.columns:
            avg_sharpe = results_df['sharpe'].mean()
        else:
            # 수익률 기반 계산
            returns = results_df['return'].values
            avg_sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        # 최대 낙폭 (MDD)
        cumulative_returns = (1 + results_df['return']).cumprod()
        cummax = cumulative_returns.cummax()
        drawdown = (cumulative_returns - cummax) / cummax
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 최고/최악 윈도우
        best_window = results_df.loc[results_df['return'].idxmax()]
        worst_window = results_df.loc[results_df['return'].idxmin()]

        # 일관성 (수익률 표준편차)
        consistency = results_df['return'].std()

        analysis = {
            'total_return': total_return,
            'avg_return_per_window': avg_return,
            'win_rate': win_rate,
            'avg_sharpe': avg_sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'consistency': consistency,
            'num_windows': len(results_df),
            'best_window_return': best_window['return'],
            'worst_window_return': worst_window['return']
        }

        return analysis

    def generate_report(self, results_df: pd.DataFrame) -> str:
        """
        워크포워드 백테스팅 리포트 생성
        """
        analysis = self.analyze_results(results_df)

        report = f"""
╔══════════════════════════════════════════════════════════╗
║          Walk-Forward Backtest Report                   ║
╚══════════════════════════════════════════════════════════╝

🔄 Backtest Configuration:
  • Window Type:                 {self.window_type}
  • Train Period:                {self.train_period_days} days
  • Test Period:                 {self.test_period_days} days
  • Total Windows:               {analysis['num_windows']}

📊 Overall Performance:
  • Total Return:                {analysis['total_return']:>12.2%}
  • Avg Return per Window:       {analysis['avg_return_per_window']:>12.2%}
  • Win Rate:                    {analysis['win_rate']:>12.1%}

📈 Risk-Adjusted Metrics:
  • Avg Sharpe Ratio:            {analysis['avg_sharpe']:>12.2f}
  • Max Drawdown:                {analysis['max_drawdown']:>12.2%}
  • Calmar Ratio:                {analysis['calmar_ratio']:>12.2f}
  • Consistency (Std):           {analysis['consistency']:>12.2%}

🏆 Best/Worst Windows:
  • Best Window Return:          {analysis['best_window_return']:>12.2%}
  • Worst Window Return:         {analysis['worst_window_return']:>12.2%}

"""

        # 윈도우별 상세 결과 (최근 5개)
        report += "📋 Recent Window Details:\n"
        recent_windows = results_df.tail(5)

        for _, row in recent_windows.iterrows():
            report += (f"  • Window {row['window_id']}: "
                      f"Test {row['test_start'].date()}~{row['test_end'].date()} | "
                      f"Return: {row['return']:>7.2%} | "
                      f"Sharpe: {row.get('sharpe', 0):>5.2f}\n")

        report += "\n" + "═" * 58 + "\n"

        return report

    def plot_results(self, results_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        결과 시각화 (matplotlib)

        Args:
            results_df: 결과 데이터프레임
            save_path: 저장 경로 (옵션)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            # 1. 누적 수익률
            cumulative_returns = (1 + results_df['return']).cumprod()
            test_dates = results_df['test_end']

            axes[0].plot(test_dates, cumulative_returns, marker='o', linestyle='-', linewidth=2)
            axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            axes[0].set_title('Cumulative Returns (Walk-Forward)', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Cumulative Return')
            axes[0].grid(True, alpha=0.3)
            axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            # 2. 윈도우별 수익률
            axes[1].bar(test_dates, results_df['return'], color=['green' if r > 0 else 'red' for r in results_df['return']], alpha=0.7)
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[1].set_title('Per-Window Returns', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Return (%)')
            axes[1].grid(True, alpha=0.3)
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            # 3. Drawdown
            cummax = cumulative_returns.cummax()
            drawdown = (cumulative_returns - cummax) / cummax

            axes[2].fill_between(test_dates, drawdown, 0, color='red', alpha=0.3)
            axes[2].set_title('Drawdown', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('Drawdown (%)')
            axes[2].set_xlabel('Date')
            axes[2].grid(True, alpha=0.3)
            axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")

            plt.show()

        except ImportError:
            logger.warning("Matplotlib not available, skipping plot")
        except Exception as e:
            logger.error(f"Plotting failed: {e}")


# === 사용 예시 ===
if __name__ == '__main__':
    # 더미 데이터 생성
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    prices = np.random.randn(len(dates)).cumsum() + 100
    data = pd.DataFrame({'price': prices}, index=dates)

    # 워크포워드 백테스트 설정
    wf_backtest = WalkForwardBacktest(
        train_period_days=180,  # 6개월 학습
        test_period_days=30,    # 1개월 테스트
        window_type='fixed'
    )

    # 학습/테스트 함수 정의
    def train_function(train_data):
        """간단한 이동평균 전략 학습 (여기서는 파라미터만 계산)"""
        # 최적 이동평균 기간 찾기 (예시)
        optimal_period = 20
        return {'ma_period': optimal_period}

    def test_function(model, test_data):
        """테스트 기간 성과 평가"""
        ma_period = model['ma_period']
        prices = test_data['price'].values

        # 간단한 MA 기반 수익률 계산 (예시)
        returns = np.diff(prices) / prices[:-1]
        total_return = np.sum(returns)

        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        return {
            'return': total_return,
            'sharpe': sharpe,
            'trades': len(returns)
        }

    # 백테스트 실행
    results = wf_backtest.run_backtest(
        data,
        train_func=train_function,
        test_func=test_function
    )

    # 결과 출력
    print(wf_backtest.generate_report(results))

    # 시각화
    # wf_backtest.plot_results(results, save_path='walkforward_results.png')
