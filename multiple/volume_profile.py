"""
volume_profile.py
볼륨 프로파일 분석 기능
"""
import numpy as np
import pandas as pd
from logger_config import get_logger

logger = get_logger(__name__)


class VolumeProfileAnalyzer:
    """볼륨 프로파일 분석기"""

    def analyze_volume_profile(self, data, bins=20):
        """
        볼륨 프로파일 분석

        Args:
            data: OHLCV 데이터프레임
            bins: 가격 구간 개수

        Returns:
            (bool, str, dict): (신호 여부, 메시지, 상세 정보)
        """
        try:
            if data is None or len(data) < 20:
                return None, "데이터부족", {}

            # 최근 60일 데이터 사용
            recent_data = data.tail(60)

            # 가격 범위
            price_min = recent_data['Low'].min()
            price_max = recent_data['High'].max()

            # 가격 구간 생성
            price_bins = np.linspace(price_min, price_max, bins + 1)

            # 각 구간별 거래량 집계
            volume_by_price = np.zeros(bins)

            for idx, row in recent_data.iterrows():
                # 해당 봉의 가격 범위를 여러 구간에 분산
                low = row['Low']
                high = row['High']
                volume = row['Volume']

                # 해당 봉이 겹치는 모든 구간 찾기
                for i in range(bins):
                    bin_low = price_bins[i]
                    bin_high = price_bins[i + 1]

                    # 겹치는 부분 계산
                    overlap_low = max(low, bin_low)
                    overlap_high = min(high, bin_high)

                    if overlap_low < overlap_high:
                        # 겹치는 비율만큼 거래량 할당
                        overlap_ratio = (overlap_high - overlap_low) / (high - low) if high > low else 1.0
                        volume_by_price[i] += volume * overlap_ratio

            # POC (Point of Control) - 거래량이 가장 많은 가격대
            poc_idx = np.argmax(volume_by_price)
            poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2

            # VAH (Value Area High), VAL (Value Area Low) - 거래량의 70%가 집중된 구간
            total_volume = volume_by_price.sum()
            target_volume = total_volume * 0.70

            # POC부터 시작해서 양쪽으로 확장
            cumulative_volume = volume_by_price[poc_idx]
            val_idx = poc_idx
            vah_idx = poc_idx

            while cumulative_volume < target_volume and (val_idx > 0 or vah_idx < bins - 1):
                # 양쪽 중 거래량이 큰 쪽으로 확장
                left_vol = volume_by_price[val_idx - 1] if val_idx > 0 else 0
                right_vol = volume_by_price[vah_idx + 1] if vah_idx < bins - 1 else 0

                if left_vol >= right_vol and val_idx > 0:
                    val_idx -= 1
                    cumulative_volume += volume_by_price[val_idx]
                elif vah_idx < bins - 1:
                    vah_idx += 1
                    cumulative_volume += volume_by_price[vah_idx]
                else:
                    break

            val_price = price_bins[val_idx]
            vah_price = price_bins[vah_idx + 1]

            # 현재 가격
            current_price = data['Close'].iloc[-1]

            # 분석
            results = {
                'poc': poc_price,
                'val': val_price,
                'vah': vah_price,
                'current_price': current_price,
                'volume_distribution': volume_by_price.tolist(),
                'price_bins': price_bins.tolist(),
            }

            # 신호 판단
            # 1. 현재 가격이 VAL 근처 (매수 기회)
            if val_price <= current_price <= val_price * 1.02:
                return True, "VAL근처매수기회", results

            # 2. 현재 가격이 VAH 근처 (매도 고려)
            if vah_price * 0.98 <= current_price <= vah_price:
                return False, "VAH근처매도고려", results

            # 3. 현재 가격이 POC 근처 (균형 상태)
            if poc_price * 0.98 <= current_price <= poc_price * 1.02:
                return None, "POC근처균형", results

            # 4. 현재 가격이 Value Area 밖
            if current_price < val_price:
                return True, "ValueArea아래_강매수", results
            elif current_price > vah_price:
                return False, "ValueArea위_과매수", results

            return None, "정상범위", results

        except Exception as e:
            logger.error(f"Volume profile analysis error: {e}")
            return None, f"오류({str(e)})", {}

    def check_volume_breakout(self, data, threshold=2.0):
        """
        거래량 돌파 확인

        Args:
            data: OHLCV 데이터프레임
            threshold: 평균 대비 배수

        Returns:
            (bool, str, dict): (돌파 여부, 메시지, 상세 정보)
        """
        try:
            if data is None or len(data) < 20:
                return None, "데이터부족", {}

            # 최근 20일 평균 거래량
            avg_volume = data['Volume'].tail(20).mean()

            # 오늘 거래량
            today_volume = data['Volume'].iloc[-1]

            # 배수 계산
            volume_ratio = today_volume / avg_volume if avg_volume > 0 else 0

            results = {
                'today_volume': today_volume,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
            }

            if volume_ratio >= threshold:
                return True, f"거래량돌파({volume_ratio:.1f}x)", results
            else:
                return False, f"정상거래량({volume_ratio:.1f}x)", results

        except Exception as e:
            logger.error(f"Volume breakout check error: {e}")
            return None, f"오류({str(e)})", {}


# 테스트 코드
if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime, timedelta

    print("=== 볼륨 프로파일 분석 테스트 ===\n")

    analyzer = VolumeProfileAnalyzer()

    # 테스트 종목
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f"종목: {symbol}")
    print(f"기간: {start_date.date()} ~ {end_date.date()}\n")

    # 데이터 다운로드
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)

    if data is not None and not data.empty:
        # 볼륨 프로파일 분석
        signal, msg, details = analyzer.analyze_volume_profile(data)

        print(f"신호: {signal}")
        print(f"메시지: {msg}")
        print(f"\n상세 정보:")
        print(f"  POC (Point of Control): ${details.get('poc', 0):.2f}")
        print(f"  VAH (Value Area High): ${details.get('vah', 0):.2f}")
        print(f"  VAL (Value Area Low): ${details.get('val', 0):.2f}")
        print(f"  현재 가격: ${details.get('current_price', 0):.2f}")

        # 거래량 돌파 확인
        breakout, breakout_msg, breakout_details = analyzer.check_volume_breakout(data, threshold=2.0)

        print(f"\n거래량 돌파:")
        print(f"  신호: {breakout}")
        print(f"  메시지: {breakout_msg}")
        print(f"  오늘: {breakout_details.get('today_volume', 0):,.0f}")
        print(f"  평균: {breakout_details.get('avg_volume', 0):,.0f}")
        print(f"  비율: {breakout_details.get('volume_ratio', 0):.2f}x")

    print("\n✅ 테스트 완료")
