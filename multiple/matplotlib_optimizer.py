"""
Matplotlib Memory Leak Fix and Optimization Guide
차트 메모리 누수 방지 및 최적화
"""

import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 대신 메모리 효율적인 백엔드 사용

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import gc
from typing import Optional
from contextlib import contextmanager
from logger_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# 1. 메모리 누수 패턴 (BAD Examples)
# ============================================================================

def create_chart_with_leak(data):
    """❌ BAD: 메모리 누수 발생"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data)
    ax.set_title('Stock Price')

    # 문제: figure를 닫지 않음!
    # 반복 호출 시 메모리 누적
    return fig


def create_multiple_charts_leak(data_list):
    """❌ BAD: 여러 차트 생성 시 누수"""
    for data in data_list:
        fig, ax = plt.subplots()
        ax.plot(data)
        # figure가 메모리에 계속 쌓임
        # plt.close()를 호출하지 않음!


# ============================================================================
# 2. 메모리 누수 방지 (GOOD Examples)
# ============================================================================

def create_chart_safe(data):
    """✅ GOOD: 메모리 안전한 차트 생성"""
    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        ax.plot(data)
        ax.set_title('Stock Price')
        return fig
    except Exception as e:
        # 에러 발생 시에도 figure 닫기
        plt.close(fig)
        raise e


def create_multiple_charts_safe(data_list):
    """✅ GOOD: 메모리 안전한 여러 차트 생성"""
    figures = []

    for data in data_list:
        fig, ax = plt.subplots()
        ax.plot(data)
        figures.append(fig)

        # 사용 후 즉시 닫기
        plt.close(fig)

    return figures


# ============================================================================
# 3. Context Manager 패턴 (Best Practice)
# ============================================================================

@contextmanager
def safe_figure(*args, **kwargs):
    """
    메모리 안전한 figure 생성 context manager

    Usage:
        with safe_figure(figsize=(10, 6)) as (fig, ax):
            ax.plot(data)
            ax.set_title('Chart')
            # 자동으로 닫힘
    """
    fig, ax = plt.subplots(*args, **kwargs)
    try:
        yield fig, ax
    finally:
        plt.close(fig)
        gc.collect()  # 가비지 컬렉션 강제 실행


@contextmanager
def safe_multiple_subplots(nrows=1, ncols=1, **kwargs):
    """
    여러 subplot을 안전하게 생성

    Usage:
        with safe_multiple_subplots(2, 2, figsize=(12, 8)) as (fig, axes):
            axes[0, 0].plot(data1)
            axes[0, 1].plot(data2)
            # 자동으로 정리됨
    """
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    try:
        yield fig, axes
    finally:
        plt.close(fig)
        gc.collect()


# ============================================================================
# 4. 차트 생성 클래스 (메모리 관리 포함)
# ============================================================================

class ChartManager:
    """메모리 안전한 차트 생성 관리자"""

    def __init__(self):
        self.active_figures = []

    def create_figure(self, figsize=(10, 6), **kwargs) -> tuple:
        """
        새 figure 생성 및 추적

        Returns:
            (fig, ax) 튜플
        """
        fig, ax = plt.subplots(figsize=figsize, **kwargs)
        self.active_figures.append(fig)
        return fig, ax

    def close_figure(self, fig):
        """특정 figure 닫기"""
        if fig in self.active_figures:
            self.active_figures.remove(fig)
        plt.close(fig)

    def close_all(self):
        """모든 활성 figure 닫기"""
        for fig in self.active_figures:
            plt.close(fig)
        self.active_figures.clear()
        plt.close('all')  # 추가 안전장치
        gc.collect()

    def __del__(self):
        """소멸자: 모든 figure 정리"""
        self.close_all()


# ============================================================================
# 5. 실전 예제: 주식 차트 생성
# ============================================================================

class StockChartOptimized:
    """최적화된 주식 차트 클래스"""

    def __init__(self):
        self.chart_manager = ChartManager()

    def create_price_chart(self, data, title="Stock Price"):
        """❌ OLD: 메모리 누수 가능"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Close'], label='Close Price')
        ax.set_title(title)
        ax.legend()
        # figure를 닫지 않음!
        return fig

    def create_price_chart_safe(self, data, title="Stock Price"):
        """✅ NEW: 메모리 안전"""
        with safe_figure(figsize=(12, 6)) as (fig, ax):
            ax.plot(data.index, data['Close'], label='Close Price')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # figure를 저장하거나 표시
            fig.savefig('chart.png', dpi=100, bbox_inches='tight')
            # context manager가 자동으로 닫음

    def create_multiple_indicators(self, data):
        """✅ 여러 지표를 안전하게 그리기"""
        with safe_multiple_subplots(3, 1, figsize=(12, 10), sharex=True) as (fig, axes):
            # 가격 차트
            axes[0].plot(data.index, data['Close'], label='Close')
            axes[0].set_title('Price')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 거래량
            axes[1].bar(data.index, data['Volume'], alpha=0.5)
            axes[1].set_title('Volume')
            axes[1].grid(True, alpha=0.3)

            # RSI
            if 'RSI' in data.columns:
                axes[2].plot(data.index, data['RSI'], color='purple')
                axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5)
                axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.5)
                axes[2].set_title('RSI')
                axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig('indicators.png', dpi=100, bbox_inches='tight')

    def cleanup(self):
        """모든 차트 정리"""
        self.chart_manager.close_all()


# ============================================================================
# 6. PyQt5 통합 시 메모리 관리
# ============================================================================

class PyQtChartWidget:
    """PyQt5에서 사용하는 차트 위젯 (메모리 최적화)"""

    def __init__(self):
        self.current_figure = None

    def update_chart(self, data):
        """차트 업데이트 (이전 차트 자동 정리)"""
        # 이전 figure 정리
        if self.current_figure is not None:
            plt.close(self.current_figure)
            self.current_figure = None

        # 새 figure 생성
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data)
        ax.set_title('Updated Chart')

        self.current_figure = fig
        return fig

    def clear_chart(self):
        """차트 지우기"""
        if self.current_figure is not None:
            plt.close(self.current_figure)
            self.current_figure = None
            gc.collect()

    def __del__(self):
        """소멸자: 메모리 정리"""
        self.clear_chart()


# ============================================================================
# 7. 메모리 사용량 모니터링
# ============================================================================

def get_matplotlib_memory_usage():
    """현재 matplotlib이 사용 중인 메모리 확인"""
    import sys

    # 활성 figure 수
    num_figures = len(plt.get_fignums())

    # 각 figure의 메모리 사용량 추정
    total_memory_mb = 0
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        # figure 객체의 대략적인 메모리 사용량
        memory_mb = sys.getsizeof(fig) / 1024 / 1024
        total_memory_mb += memory_mb

    return {
        'num_figures': num_figures,
        'total_memory_mb': total_memory_mb
    }


def print_matplotlib_stats():
    """matplotlib 메모리 통계 출력"""
    stats = get_matplotlib_memory_usage()
    logger.info("=" * 60)
    logger.info("Matplotlib Memory Stats")
    logger.info("=" * 60)
    logger.info(f"Active Figures: {stats['num_figures']}")
    logger.info(f"Total Memory: {stats['total_memory_mb']:.2f} MB")
    logger.info("=" * 60)


# ============================================================================
# 8. 정리 유틸리티 함수
# ============================================================================

def cleanup_all_matplotlib():
    """모든 matplotlib 리소스 정리"""
    plt.close('all')  # 모든 figure 닫기
    gc.collect()  # 가비지 컬렉션 강제 실행

    # 캐시 정리
    if hasattr(plt, '_cachedRenderer'):
        plt._cachedRenderer.clear()


def periodic_cleanup():
    """주기적으로 호출할 정리 함수"""
    # 10개 이상의 figure가 열려있으면 모두 닫기
    if len(plt.get_fignums()) > 10:
        logger.warning("Too many figures open. Cleaning up...")
        cleanup_all_matplotlib()


# ============================================================================
# 9. 최적화 체크리스트
# ============================================================================

"""
Matplotlib 메모리 누수 방지 체크리스트:

□ 사용 후 항상 plt.close(fig) 호출
□ context manager (with 구문) 사용
□ try-finally 블록에서 정리 코드 실행
□ ChartManager 같은 관리 클래스 사용
□ 주기적으로 plt.close('all') 호출
□ 가비지 컬렉션 gc.collect() 주기적 실행
□ figure 수 모니터링
□ 불필요한 figure 생성 최소화
□ savefig 후 즉시 figure 닫기
□ 클래스 소멸자에서 cleanup 구현
"""


# ============================================================================
# 10. 사용 예제
# ============================================================================

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    # 테스트 데이터
    dates = pd.date_range('2023-01-01', periods=100)
    data = pd.DataFrame({
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 100),
        'RSI': np.random.uniform(20, 80, 100)
    }, index=dates)

    logger.info("\nSafe chart creation with context manager:")
    with safe_figure(figsize=(10, 6)) as (fig, ax):
        ax.plot(data.index, data['Close'])
        ax.set_title('Stock Price - Safe')
        fig.savefig('safe_chart.png', dpi=100, bbox_inches='tight')
        logger.info("Chart saved: safe_chart.png")

    logger.info("\nUsing ChartManager:")
    chart_mgr = ChartManager()
    fig, ax = chart_mgr.create_figure(figsize=(10, 6))
    ax.plot(data.index, data['Volume'])
    ax.set_title('Volume')
    fig.savefig('volume_chart.png', dpi=100, bbox_inches='tight')
    chart_mgr.close_all()
    logger.info("Chart saved: volume_chart.png")

    logger.info("\nOptimized stock chart:")
    stock_chart = StockChartOptimized()
    stock_chart.create_price_chart_safe(data, "Test Stock")
    stock_chart.create_multiple_indicators(data)
    stock_chart.cleanup()
    logger.info("Charts created and cleaned up")

    # 메모리 통계
    print_matplotlib_stats()
