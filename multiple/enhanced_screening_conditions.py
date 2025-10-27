"""
Enhanced Screening Conditions
개선된 스크리닝 조건 모듈

주요 개선 사항:
1. 수익률 매도 조건 구현 (손절/익절/트레일링스톱)
2. BB+RSI 매수 조건 강화 (추세 확인 추가)
3. 거래량 급감 매도 조건 제거 → 명확한 손절로 대체
4. MACD+거래량 조건 강화
5. 모멘텀 매수 조건 개선 (과매수 필터)
"""

import pandas as pd
import numpy as np
from logger_config import get_logger

logger = get_logger(__name__)


class EnhancedScreeningConditions:
    """개선된 스크리닝 조건 클래스"""

    def __init__(self):
        """초기화"""
        # 손절/익절 설정 (사용자가 조정 가능)
        self.stop_loss_pct = -8.0  # -8% 손절
        self.take_profit_pct = 15.0  # +15% 익절
        self.trailing_stop_pct = 5.0  # 최고가 대비 -5% 트레일링 스톱

    # ==================== 매수 조건 ====================

    def check_ma_buy_condition_enhanced(self, data, current, prev):
        """
        이동평균 기술적 매수 조건 (강화)

        Args:
            data: 전체 데이터
            current: 현재 데이터
            prev: 전일 데이터

        Returns:
            (bool, str): (조건 만족 여부, 신호 이름)
        """
        try:
            # 기본 조건
            if not (current['MA60'] > current['MA120'] and current['Close'] > current['MA60']):
                return False, None

            # 강화 조건 1: 이동평균선 상승 중
            if not (current['MA60'] > prev['MA60'] and current['MA120'] > prev['MA120']):
                logger.debug("MA not rising")
                return False, None

            # 강화 조건 2: 주가가 60일선 근처 (3% 이내)
            distance_pct = abs(current['Close'] - current['MA60']) / current['MA60'] * 100
            if distance_pct > 3.0:
                logger.debug(f"Price too far from MA60: {distance_pct:.1f}%")
                return False, None

            # 강화 조건 3: RSI 과매수 방지
            if current['RSI'] > 75:
                logger.debug(f"RSI overbought: {current['RSI']:.1f}")
                return False, None

            # 개선 추가 1: 거래량 확인
            if current['Volume_Ratio'] < 1.0:
                logger.debug(f"Low volume: {current['Volume_Ratio']:.2f}")
                return False, None

            # 개선 추가 2: 추세 강도 확인
            trend_strength = (current['MA60'] - current['MA120']) / current['MA120'] * 100
            if trend_strength < 2.0:
                logger.debug(f"Weak trend: {trend_strength:.1f}%")
                return False, None

            # 개선 추가 3: 최근 모멘텀 확인 (5일)
            if len(data) >= 6:
                five_days_ago = data['Close'].iloc[-6]
                if current['Close'] <= five_days_ago:
                    logger.debug("No recent momentum")
                    return False, None

            return True, "강화된MA매수"

        except Exception as e:
            logger.error(f"Error in MA buy condition: {e}")
            return False, None

    def check_bb_rsi_buy_condition_enhanced(self, data, current, prev):
        """
        볼린저밴드 + RSI 매수 조건 (강화)

        중요: 추세 확인 필수! (하락장 함정 방지)

        Args:
            data: 전체 데이터
            current: 현재 데이터
            prev: 전일 데이터

        Returns:
            (bool, str): (조건 만족 여부, 신호 이름)
        """
        try:
            # 기본 조건: BB 하단 근처 (1.00, 기존 1.02에서 엄격하게)
            if not (current['Close'] <= current['BB_Lower'] * 1.00):
                return False, None

            # 기본 조건: RSI 과매도 (30, 기존 35에서 엄격하게)
            if not (current['RSI'] < 30):
                return False, None

            # ✨ 핵심 개선: 상승 추세 확인 필수!
            if not (current['MA60'] > current['MA120']):
                logger.debug("Not in uptrend - skipping BB buy")
                return False, None

            # 개선 추가 1: 거래량 급감 아님
            if current['Volume_Ratio'] < 0.8:
                logger.debug(f"Volume too low: {current['Volume_Ratio']:.2f}")
                return False, None

            # 개선 추가 2: 3일 연속 RSI < 35 확인 (일시적 과매도 제외)
            if len(data) >= 3:
                recent_rsi = data['RSI'].tail(3)
                if not all(recent_rsi < 35):
                    logger.debug("Not sustained oversold")
                    return False, None

            # 개선 추가 3: MACD 반등 조짐
            if 'MACD' in current and 'MACD_Signal' in current:
                if current['MACD'] < 0 and current['MACD'] <= prev['MACD']:
                    logger.debug("MACD still declining")
                    return False, None

            return True, "강화된BB+RSI매수"

        except Exception as e:
            logger.error(f"Error in BB+RSI buy condition: {e}")
            return False, None

    def check_macd_volume_buy_condition_enhanced(self, data, current, prev):
        """
        MACD 골든크로스 + 거래량 매수 조건 (강화)

        Args:
            data: 전체 데이터
            current: 현재 데이터
            prev: 전일 데이터

        Returns:
            (bool, str): (조건 만족 여부, 신호 이름)
        """
        try:
            # 기본 조건: MACD 골든크로스 (오늘 처음)
            if not (current['MACD'] > current['MACD_Signal'] and
                    prev['MACD'] <= prev['MACD_Signal']):
                return False, None

            # 기본 조건: 거래량 증가 (1.5로 강화, 기존 1.2)
            if not (current['Volume_Ratio'] > 1.5):
                logger.debug(f"Volume increase not strong enough: {current['Volume_Ratio']:.2f}")
                return False, None

            # 개선 추가 1: MACD 히스토그램 양수 (강한 모멘텀)
            if 'MACD_Hist' in current:
                if current['MACD_Hist'] <= 0:
                    logger.debug("MACD Histogram not positive")
                    return False, None

            # 개선 추가 2: 단기 추세도 상승
            if 'MA20' in current:
                if current['Close'] <= current['MA20']:
                    logger.debug("Price below MA20")
                    return False, None

            return True, "강화된MACD+거래량"

        except Exception as e:
            logger.error(f"Error in MACD+Volume buy condition: {e}")
            return False, None

    def check_momentum_buy_condition_enhanced(self, data, current, prev):
        """
        모멘텀 상승 매수 조건 (개선)

        개선:
        - 10일 기준으로 변경 (기존 21일은 너무 늦음)
        - 수익률 범위 제한 (3-8%, 너무 높으면 고점 위험)
        - RSI 상한선 추가 (65, 과매수 제외)
        - BB 상단 근처 제외

        Args:
            data: 전체 데이터
            current: 현재 데이터
            prev: 전일 데이터

        Returns:
            (bool, str): (조건 만족 여부, 신호 이름)
        """
        try:
            # 10일 모멘텀 계산 (기존 21일에서 변경)
            if len(data) < 11:
                return False, None

            price_10d_ago = data['Close'].iloc[-11]
            momentum_10d = (current['Close'] / price_10d_ago - 1) * 100

            # 모멘텀 범위: 3-8% (기존 >5%에서 변경)
            if not (3.0 < momentum_10d < 8.0):
                logger.debug(f"Momentum out of range: {momentum_10d:.1f}%")
                return False, None

            # RSI 범위: 50-65 (기존 >50에서 상한 추가)
            if not (50 < current['RSI'] < 65):
                logger.debug(f"RSI out of range: {current['RSI']:.1f}")
                return False, None

            # 추세 확인 필수
            if not (current['MA60'] > current['MA120']):
                logger.debug("Not in uptrend")
                return False, None

            # BB 상단 근처 아님 (과매수 구간 제외)
            if current['Close'] >= current['BB_Upper'] * 0.95:
                logger.debug("Near BB upper band")
                return False, None

            return True, "강화된모멘텀매수"

        except Exception as e:
            logger.error(f"Error in momentum buy condition: {e}")
            return False, None

    # ==================== 매도 조건 ====================

    def check_profit_sell_condition(self, current_price, buy_price, peak_price=None):
        """
        수익률 매도 조건 (손절/익절/트레일링스톱)

        Args:
            current_price: 현재 가격
            buy_price: 매수 가격
            peak_price: 매수 후 최고가 (트레일링 스톱용, 선택)

        Returns:
            (bool, str, float): (매도 여부, 매도 이유, 수익률)
        """
        if buy_price is None or buy_price <= 0:
            logger.warning("Invalid buy_price for profit sell check")
            return False, None, 0.0

        try:
            # 수익률 계산
            profit_pct = (current_price / buy_price - 1) * 100

            # 1. 손절 (-8%) - 부동소수점 오차 허용 (0.01%)
            if profit_pct <= (self.stop_loss_pct + 0.01):
                return True, f"손절({profit_pct:+.1f}%)", profit_pct

            # 2. 익절 (+15%) - 부동소수점 오차 허용 (0.01%)
            if profit_pct >= (self.take_profit_pct - 0.01):
                return True, f"익절({profit_pct:+.1f}%)", profit_pct

            # 3. 트레일링 스톱 (최고가 대비 -5%)
            if peak_price is not None and peak_price > buy_price:
                peak_profit_pct = (peak_price / buy_price - 1) * 100
                drawdown_from_peak = (current_price / peak_price - 1) * 100

                # 최고가에서 5% 이상 하락하면 매도
                if drawdown_from_peak <= -self.trailing_stop_pct:
                    return True, f"트레일링스톱(최고{peak_profit_pct:.1f}%→현재{profit_pct:.1f}%)", profit_pct

            return False, None, profit_pct

        except Exception as e:
            logger.error(f"Error in profit sell condition: {e}")
            return False, None, 0.0

    def check_technical_sell_condition_enhanced(self, data, current, prev):
        """
        기술적 매도 조건 (개선)

        개선:
        - 60일선 이탈 기준 3% → 5% (더 신중)
        - RSI 약세 확인 추가 (40 이하)
        - 2일 연속 조건 만족 확인 (일시적 오류 제거)

        Args:
            data: 전체 데이터
            current: 현재 데이터
            prev: 전일 데이터

        Returns:
            (bool, str): (조건 만족 여부, 신호 이름)
        """
        try:
            # 조건 1: MA60 < MA120 (추세 전환)
            condition1 = current['MA60'] < current['MA120']

            # 조건 2: MA60 하락 중
            condition2 = current['MA60'] < prev['MA60']

            # 조건 3: 주가 60일선 5% 이탈 (기존 3%에서 완화)
            condition3 = current['Close'] < current['MA60'] * 0.95

            # 조건 4: RSI 약세 (40 이하)
            condition4 = current['RSI'] < 40

            # 모든 조건 만족해야 매도
            if not (condition1 and condition2 and condition3 and condition4):
                return False, None

            # 개선: 2일 연속 조건 만족 확인 (일시적 오류 제거)
            if len(data) >= 2:
                prev2 = data.iloc[-2]
                prev_condition1 = prev['MA60'] < prev['MA120']
                prev_condition3 = prev['Close'] < prev['MA60'] * 0.95

                if not (prev_condition1 or prev_condition3):
                    logger.debug("Not sustained sell signal")
                    return False, None

            return True, "강화된기술적매도"

        except Exception as e:
            logger.error(f"Error in technical sell condition: {e}")
            return False, None

    def check_bb_rsi_sell_condition_enhanced(self, data, current, prev):
        """
        볼린저밴드 상단 + RSI 매도 조건 (개선)

        개선:
        - RSI 기준 70 → 75 (더 확실한 과매수)
        - 거래량 급증 확인 추가 (1.5배)
        - 2일 연속 조건 만족 확인

        Args:
            data: 전체 데이터
            current: 현재 데이터
            prev: 전일 데이터

        Returns:
            (bool, str): (조건 만족 여부, 신호 이름)
        """
        try:
            # 기본 조건: BB 상단 도달
            if not (current['Close'] >= current['BB_Upper']):
                return False, None

            # 기본 조건: RSI 과매수 (75, 기존 70에서 강화)
            if not (current['RSI'] > 75):
                return False, None

            # 개선 추가: 거래량 급증 (피로 신호)
            if not (current['Volume_Ratio'] > 1.5):
                logger.debug(f"No volume spike: {current['Volume_Ratio']:.2f}")
                return False, None

            # 개선: 2일 연속 조건 만족 확인
            if len(data) >= 2:
                if not (prev['Close'] >= prev['BB_Upper'] * 0.98 or prev['RSI'] > 70):
                    logger.debug("Not sustained overbought")
                    return False, None

            return True, "강화된BB+RSI매도"

        except Exception as e:
            logger.error(f"Error in BB+RSI sell condition: {e}")
            return False, None

    # ==================== 유틸리티 ====================

    def calculate_position_size(self, capital, current_price, buy_price=None, risk_per_trade=0.02):
        """
        포지션 크기 계산 (리스크 관리)

        Args:
            capital: 가용 자본
            current_price: 현재 가격
            buy_price: 매수 가격 (손절 라인 계산용, 선택)
            risk_per_trade: 거래당 리스크 (기본 2%)

        Returns:
            int: 매수 주식 수
        """
        try:
            if buy_price is None:
                buy_price = current_price

            # 손절 라인까지 거리
            stop_loss_price = buy_price * (1 + self.stop_loss_pct / 100)
            risk_per_share = abs(buy_price - stop_loss_price)

            # 리스크 금액
            risk_amount = capital * risk_per_trade

            # 최대 매수 가능 주식 수
            if risk_per_share > 0:
                position_size = int(risk_amount / risk_per_share)
            else:
                # 리스크가 0이면 자본의 25% 사용
                position_size = int(capital * 0.25 / current_price)

            return max(1, position_size)  # 최소 1주

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1

    def get_screening_summary(self):
        """
        스크리닝 조건 요약 정보 반환

        Returns:
            dict: 조건 요약
        """
        return {
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'trailing_stop_pct': self.trailing_stop_pct,
            'conditions': {
                'buy': [
                    '강화된MA매수 (MA60>MA120, 거래량, 추세강도, 모멘텀)',
                    '강화된BB+RSI매수 (BB하단, RSI<30, 상승추세 필수)',
                    '강화된MACD+거래량 (골든크로스, 거래량1.5배)',
                    '강화된모멘텀매수 (10일 3-8%, RSI 50-65)'
                ],
                'sell': [
                    '손절 (-8%)',
                    '익절 (+15%)',
                    '트레일링스톱 (최고가 -5%)',
                    '강화된기술적매도 (MA전환, 5%이탈, RSI<40)',
                    '강화된BB+RSI매도 (BB상단, RSI>75, 거래량급증)'
                ]
            }
        }


# ==================== 헬퍼 함수 ====================

def create_enhanced_screener():
    """
    개선된 스크리너 인스턴스 생성

    Returns:
        EnhancedScreeningConditions: 스크리너 인스턴스
    """
    return EnhancedScreeningConditions()


def test_enhanced_conditions():
    """
    개선된 조건 테스트
    """
    print("=" * 70)
    print("🧪 Enhanced Screening Conditions Test")
    print("=" * 70)

    screener = create_enhanced_screener()
    summary = screener.get_screening_summary()

    print("\n📊 Configuration:")
    print(f"   Stop Loss: {summary['stop_loss_pct']}%")
    print(f"   Take Profit: {summary['take_profit_pct']}%")
    print(f"   Trailing Stop: {summary['trailing_stop_pct']}%")

    print("\n✅ Buy Conditions:")
    for i, cond in enumerate(summary['conditions']['buy'], 1):
        print(f"   {i}. {cond}")

    print("\n🚨 Sell Conditions:")
    for i, cond in enumerate(summary['conditions']['sell'], 1):
        print(f"   {i}. {cond}")

    # 수익률 매도 조건 테스트
    print("\n" + "=" * 70)
    print("🧪 Profit Sell Condition Test")
    print("=" * 70)

    test_cases = [
        {'buy': 100, 'current': 92, 'peak': None, 'expected': '손절'},
        {'buy': 100, 'current': 115, 'peak': None, 'expected': '익절'},
        {'buy': 100, 'current': 110, 'peak': 120, 'expected': '트레일링스톱'},
        {'buy': 100, 'current': 105, 'peak': 108, 'expected': 'Hold'},
    ]

    for i, tc in enumerate(test_cases, 1):
        should_sell, reason, profit = screener.check_profit_sell_condition(
            tc['current'], tc['buy'], tc.get('peak')
        )
        status = "✅" if (tc['expected'] in reason if reason else False) or (tc['expected'] == 'Hold' and not should_sell) else "❌"
        result = reason if should_sell else "Hold"
        print(f"{status} Test {i}: Buy=${tc['buy']}, Current=${tc['current']}, "
              f"Peak=${tc.get('peak', 'N/A')} → {result} (Expected: {tc['expected']})")

    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_enhanced_conditions()
