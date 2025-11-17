"""
support_resistance.py
지지선/저항선 자동 감지 기능
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from logger_config import get_logger

logger = get_logger(__name__)


class SupportResistanceDetector:
    """지지선/저항선 감지기"""

    def detect_support_resistance(self, data, order=5, tolerance=0.02):
        """
        지지선과 저항선 감지

        Args:
            data: OHLCV 데이터프레임
            order: 극값 탐지 윈도우 크기
            tolerance: 가격대 그룹화 허용 오차 (2%)

        Returns:
            (list, list, dict): (지지선 리스트, 저항선 리스트, 상세 정보)
        """
        try:
            if data is None or len(data) < order * 2 + 1:
                return [], [], {}

            # 최근 120일 데이터 사용
            recent_data = data.tail(120).copy()

            # 저점 찾기 (지지선 후보)
            lows = recent_data['Low'].values
            support_indices = argrelextrema(lows, np.less, order=order)[0]

            # 고점 찾기 (저항선 후보)
            highs = recent_data['High'].values
            resistance_indices = argrelextrema(highs, np.greater, order=order)[0]

            # 지지선 가격들
            support_prices = lows[support_indices]

            # 저항선 가격들
            resistance_prices = highs[resistance_indices]

            # 비슷한 가격대끼리 그룹화
            support_levels = self._group_price_levels(support_prices, tolerance)
            resistance_levels = self._group_price_levels(resistance_prices, tolerance)

            # 강도 계산 (해당 레벨 근처에서 터치된 횟수)
            support_strength = self._calculate_strength(support_levels, lows, tolerance)
            resistance_strength = self._calculate_strength(resistance_levels, highs, tolerance)

            # 현재 가격
            current_price = recent_data['Close'].iloc[-1]

            # 현재 가격 기준으로 지지선/저항선 분류
            supports = []
            resistances = []

            for level, strength in zip(support_levels, support_strength):
                if level < current_price * (1 - tolerance):  # 현재가 아래
                    supports.append({'price': level, 'strength': strength, 'type': 'support'})

            for level, strength in zip(resistance_levels, resistance_strength):
                if level > current_price * (1 + tolerance):  # 현재가 위
                    resistances.append({'price': level, 'strength': strength, 'type': 'resistance'})

            # 강도순 정렬
            supports = sorted(supports, key=lambda x: x['strength'], reverse=True)
            resistances = sorted(resistances, key=lambda x: x['strength'], reverse=True)

            results = {
                'current_price': current_price,
                'support_count': len(supports),
                'resistance_count': len(resistances),
            }

            return supports[:5], resistances[:5], results  # 상위 5개만

        except Exception as e:
            logger.error(f"Support/Resistance detection error: {e}")
            return [], [], {}

    def _group_price_levels(self, prices, tolerance):
        """
        비슷한 가격대끼리 그룹화

        Args:
            prices: 가격 배열
            tolerance: 허용 오차 (비율)

        Returns:
            그룹화된 가격 레벨 리스트
        """
        if len(prices) == 0:
            return []

        # 정렬
        sorted_prices = np.sort(prices)

        levels = []
        current_group = [sorted_prices[0]]

        for price in sorted_prices[1:]:
            # 현재 그룹의 평균과 비교
            group_avg = np.mean(current_group)

            if abs(price - group_avg) / group_avg <= tolerance:
                # 같은 그룹
                current_group.append(price)
            else:
                # 새 그룹 시작
                levels.append(np.mean(current_group))
                current_group = [price]

        # 마지막 그룹 추가
        if current_group:
            levels.append(np.mean(current_group))

        return levels

    def _calculate_strength(self, levels, prices, tolerance):
        """
        각 레벨의 강도 계산 (터치 횟수)

        Args:
            levels: 가격 레벨 리스트
            prices: 전체 가격 배열
            tolerance: 허용 오차

        Returns:
            강도 리스트
        """
        strengths = []

        for level in levels:
            # 해당 레벨 근처의 가격 카운트
            touches = np.sum(np.abs(prices - level) / level <= tolerance)
            strengths.append(int(touches))

        return strengths

    def check_near_support_resistance(self, data, threshold=0.03):
        """
        현재 가격이 지지/저항선 근처인지 확인

        Args:
            data: OHLCV 데이터프레임
            threshold: 근접 판단 기준 (3%)

        Returns:
            (bool, str, dict): (신호 여부, 메시지, 상세 정보)
        """
        try:
            supports, resistances, info = self.detect_support_resistance(data)

            if not supports and not resistances:
                return None, "지지저항선없음", {}

            current_price = info.get('current_price', 0)

            # 가장 가까운 지지선 찾기
            nearest_support = None
            nearest_support_dist = float('inf')

            for sup in supports:
                dist = abs(current_price - sup['price']) / current_price
                if dist < nearest_support_dist:
                    nearest_support = sup
                    nearest_support_dist = dist

            # 가장 가까운 저항선 찾기
            nearest_resistance = None
            nearest_resistance_dist = float('inf')

            for res in resistances:
                dist = abs(current_price - res['price']) / current_price
                if dist < nearest_resistance_dist:
                    nearest_resistance = res
                    nearest_resistance_dist = dist

            results = {
                'current_price': current_price,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'support_distance': nearest_support_dist,
                'resistance_distance': nearest_resistance_dist,
            }

            # 판단
            if nearest_support and nearest_support_dist <= threshold:
                return True, f"지지선근처({nearest_support['price']:.2f})", results

            if nearest_resistance and nearest_resistance_dist <= threshold:
                return False, f"저항선근처({nearest_resistance['price']:.2f})", results

            return None, "중립구간", results

        except Exception as e:
            logger.error(f"Near support/resistance check error: {e}")
            return None, f"오류({str(e)})", {}


# 테스트 코드
if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime, timedelta

    print("=== 지지/저항선 감지 테스트 ===\n")

    detector = SupportResistanceDetector()

    # 테스트 종목
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f"종목: {symbol}")
    print(f"기간: {start_date.date()} ~ {end_date.date()}\n")

    # 데이터 다운로드
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)

    if data is not None and not data.empty:
        # 지지/저항선 감지
        supports, resistances, info = detector.detect_support_resistance(data)

        print(f"현재 가격: ${info.get('current_price', 0):.2f}\n")

        print(f"지지선 ({len(supports)}개):")
        for i, sup in enumerate(supports[:3], 1):
            print(f"  {i}. ${sup['price']:.2f} (강도: {sup['strength']})")

        print(f"\n저항선 ({len(resistances)}개):")
        for i, res in enumerate(resistances[:3], 1):
            print(f"  {i}. ${res['price']:.2f} (강도: {res['strength']})")

        # 근접 확인
        signal, msg, details = detector.check_near_support_resistance(data, threshold=0.03)

        print(f"\n근접 확인:")
        print(f"  신호: {signal}")
        print(f"  메시지: {msg}")

        if details.get('nearest_support'):
            print(f"  가장 가까운 지지선: ${details['nearest_support']['price']:.2f} "
                  f"({details['support_distance']*100:.1f}% 떨어짐)")

        if details.get('nearest_resistance'):
            print(f"  가장 가까운 저항선: ${details['nearest_resistance']['price']:.2f} "
                  f"({details['resistance_distance']*100:.1f}% 떨어짐)")

    print("\n✅ 테스트 완료")
