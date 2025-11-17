"""
Advanced Screening Engine
고급 스크리닝 엔진 - 다중 시간대, 시장 강도, 상대 강도 등
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from logger_config import get_logger
from sector_mapping import get_sector, get_sector_peers
from volume_profile import VolumeProfileAnalyzer
from support_resistance import SupportResistanceDetector

logger = get_logger(__name__)


class AdvancedScreeningEngine:
    """고급 스크리닝 엔진"""

    # 시장별 대표 지수 매핑
    MARKET_INDICES = {
        'KS': '^KS11',      # KOSPI (한국 거래소)
        'KQ': '^KQ11',      # KOSDAQ (한국 코스닥)
        'US': 'SPY',        # S&P 500 ETF (미국)
        'ST': '^OMX',       # OMXS30 (스웨덴)
        # 필요시 추가 가능:
        # 'HK': '^HSI',     # 항셍지수 (홍콩)
        # 'JP': '^N225',    # 닛케이225 (일본)
    }

    def __init__(self):
        """초기화"""
        self.market_cache = {}  # 시장 데이터 캐시
        self.volume_analyzer = VolumeProfileAnalyzer()
        self.sr_detector = SupportResistanceDetector()

    def get_market_index(self, symbol):
        """
        종목 심볼에서 시장 감지 후 적절한 시장 지수 반환

        Args:
            symbol: 종목 심볼 (예: '005930.KS', 'AAPL', 'ERIC-B.ST')

        Returns:
            str: 시장 지수 심볼 (예: '^KS11', 'SPY', '^OMX')
        """
        if '.KS' in symbol:
            return self.MARKET_INDICES.get('KS', 'SPY')
        elif '.KQ' in symbol:
            return self.MARKET_INDICES.get('KQ', 'SPY')
        elif '.ST' in symbol:
            return self.MARKET_INDICES.get('ST', 'SPY')
        else:
            # 미국 또는 기타 시장 - 기본 SPY
            return self.MARKET_INDICES.get('US', 'SPY')

    # ==================== 다중 시간대 확인 ====================

    def check_multi_timeframe(self, symbol, require_all=True):
        """
        다중 시간대 추세 확인

        Args:
            symbol: 종목 심볼
            require_all: 모든 시간대 일치 필요 여부

        Returns:
            (bool, str, dict): (통과 여부, 메시지, 상세 정보)
        """
        try:
            results = {}

            # 1. 일봉 (이미 체크됨)
            results['daily'] = True

            # 2. 주봉 추세
            weekly_data = yf.download(symbol, period='1y', interval='1wk', progress=False, auto_adjust=True)
            if weekly_data is not None and not weekly_data.empty and len(weekly_data) >= 50:
                close_col = weekly_data['Close']
                ma20_w = close_col.rolling(20).mean()
                ma50_w = close_col.rolling(50).mean()
                # Ensure scalar comparison
                ma20_val = ma20_w.iloc[-1].item() if hasattr(ma20_w.iloc[-1], 'item') else float(ma20_w.iloc[-1])
                ma50_val = ma50_w.iloc[-1].item() if hasattr(ma50_w.iloc[-1], 'item') else float(ma50_w.iloc[-1])
                weekly_trend = ma20_val > ma50_val
                results['weekly'] = weekly_trend
            else:
                results['weekly'] = None

            # 3. 월봉 추세
            monthly_data = yf.download(symbol, period='2y', interval='1mo', progress=False, auto_adjust=True)
            if monthly_data is not None and not monthly_data.empty and len(monthly_data) >= 24:
                close_col = monthly_data['Close']
                ma12_m = close_col.rolling(12).mean()
                ma24_m = close_col.rolling(24).mean()
                # Ensure scalar comparison
                ma12_val = ma12_m.iloc[-1].item() if hasattr(ma12_m.iloc[-1], 'item') else float(ma12_m.iloc[-1])
                ma24_val = ma24_m.iloc[-1].item() if hasattr(ma24_m.iloc[-1], 'item') else float(ma24_m.iloc[-1])
                monthly_trend = ma12_val > ma24_val
                results['monthly'] = monthly_trend
            else:
                results['monthly'] = None

            # 판단
            valid_results = {k: v for k, v in results.items() if v is not None}

            if require_all:
                # 모든 시간대 상승 추세
                if all(valid_results.values()):
                    return True, "다중시간대상승", results
                else:
                    failed = [k for k, v in valid_results.items() if not v]
                    return False, f"시간대불일치({','.join(failed)})", results
            else:
                # 2개 이상 상승 추세
                uptrend_count = sum(valid_results.values())
                if uptrend_count >= 2:
                    return True, f"다중시간대상승({uptrend_count}/3)", results
                else:
                    return False, f"시간대부족({uptrend_count}/3)", results

        except Exception as e:
            logger.error(f"Multi-timeframe check error for {symbol}: {e}")
            return False, f"오류({str(e)})", {}

    # ==================== 시장 강도 확인 ====================

    def check_market_strength(self, symbol=None, market_index=None, use_cache=True):
        """
        시장 강도 확인

        Args:
            symbol: 종목 심볼 (자동으로 시장 지수 감지, 우선순위 높음)
            market_index: 시장 지수 심볼 (직접 지정, symbol 없을 때만 사용)
            use_cache: 캐시 사용 여부

        Returns:
            (bool, str, dict): (통과 여부, 메시지, 상세 정보)
        """
        try:
            # 시장 지수 결정: symbol 우선, 없으면 market_index, 둘 다 없으면 SPY
            if symbol:
                market_index = self.get_market_index(symbol)
            elif not market_index:
                market_index = 'SPY'

            # 캐시 확인
            cache_key = f"{market_index}_strength"
            if use_cache and cache_key in self.market_cache:
                cached_data, cached_time = self.market_cache[cache_key]
                if datetime.now() - cached_time < timedelta(hours=1):
                    return cached_data

            # 시장 데이터 가져오기
            market_data = yf.download(market_index, period='3mo', progress=False, auto_adjust=True)

            if market_data is None or market_data.empty or len(market_data) < 50:
                return None, "시장데이터부족", {}

            results = {}

            # 1. 추세 확인 (MA20 > MA50)
            close_col = market_data['Close']
            ma20 = close_col.rolling(20).mean()
            ma50 = close_col.rolling(50).mean()
            # Ensure scalar comparison
            ma20_val = ma20.iloc[-1].item() if hasattr(ma20.iloc[-1], 'item') else float(ma20.iloc[-1])
            ma50_val = ma50.iloc[-1].item() if hasattr(ma50.iloc[-1], 'item') else float(ma50.iloc[-1])
            trend_up = ma20_val > ma50_val
            results['trend'] = trend_up

            # 2. 모멘텀 확인 (10일 수익률 > 0)
            if len(market_data) >= 11:
                close_now = close_col.iloc[-1].item() if hasattr(close_col.iloc[-1], 'item') else float(close_col.iloc[-1])
                close_10d_ago = close_col.iloc[-11].item() if hasattr(close_col.iloc[-11], 'item') else float(close_col.iloc[-11])
                returns_10d = (close_now / close_10d_ago - 1) * 100
                momentum_positive = returns_10d > 0
                results['momentum'] = momentum_positive
                results['returns_10d'] = returns_10d
            else:
                results['momentum'] = False
                results['returns_10d'] = 0.0

            # 3. VIX 확인 (미국 시장만 해당)
            if market_index in ['SPY', '^GSPC', '^DJI', '^IXIC']:
                try:
                    vix_data = yf.download('^VIX', period='5d', progress=False, auto_adjust=True)
                    if vix_data is not None and not vix_data.empty and len(vix_data) > 0:
                        vix_close = vix_data['Close'].iloc[-1]
                        vix_value = vix_close.item() if hasattr(vix_close, 'item') else float(vix_close)
                        vix_safe = vix_value < 25
                        results['vix'] = vix_value
                        results['vix_safe'] = vix_safe
                    else:
                        results['vix_safe'] = True  # VIX 없으면 통과
                except Exception as vix_err:
                    logger.debug(f"VIX data unavailable: {vix_err}")
                    results['vix_safe'] = True  # 오류 시 통과
            else:
                # 비미국 시장은 VIX 체크 스킵
                results['vix_safe'] = True
                results['vix'] = None

            # 종합 판단
            score = sum([
                results.get('trend', False),
                results.get('momentum', False),
                results.get('vix_safe', True)
            ])

            # 지수명 추가 (어떤 시장 지수를 사용했는지 표시)
            results['index_used'] = market_index

            if score >= 2:
                result = (True, f"강한시장({market_index},점수{score}/3)", results)
            elif score == 1:
                result = (None, f"중립시장({market_index},점수{score}/3)", results)
            else:
                result = (False, f"약한시장({market_index},점수{score}/3)", results)

            # 캐시 저장
            self.market_cache[cache_key] = (result, datetime.now())

            return result

        except Exception as e:
            logger.error(f"Market strength check error: {e}")
            return None, f"오류({str(e)})", {}

    # ==================== 상대 강도 비교 ====================

    def check_relative_strength(self, symbol, sector_symbols, period=60):
        """
        상대 강도 비교

        Args:
            symbol: 종목 심볼
            sector_symbols: 섹터 내 다른 종목 리스트
            period: 비교 기간 (일)

        Returns:
            (bool, str, dict): (통과 여부, 메시지, 상세 정보)
        """
        try:
            # 개별 종목 수익률
            stock_data = yf.download(symbol, period=f'{period}d', progress=False, auto_adjust=True)
            if stock_data is None or stock_data.empty or len(stock_data) < 2:
                return None, "데이터부족", {}

            close_col = stock_data['Close']
            close_first = close_col.iloc[0].item() if hasattr(close_col.iloc[0], 'item') else float(close_col.iloc[0])
            close_last = close_col.iloc[-1].item() if hasattr(close_col.iloc[-1], 'item') else float(close_col.iloc[-1])
            stock_return = (close_last / close_first - 1) * 100

            # 섹터 평균 수익률
            sector_returns = []
            for sec_symbol in sector_symbols[:10]:  # 최대 10개만
                try:
                    sec_data = yf.download(sec_symbol, period=f'{period}d', progress=False, auto_adjust=True)
                    if sec_data is not None and not sec_data.empty and len(sec_data) >= 2:
                        sec_close_col = sec_data['Close']
                        sec_first = sec_close_col.iloc[0].item() if hasattr(sec_close_col.iloc[0], 'item') else float(sec_close_col.iloc[0])
                        sec_last = sec_close_col.iloc[-1].item() if hasattr(sec_close_col.iloc[-1], 'item') else float(sec_close_col.iloc[-1])
                        sec_return = (sec_last / sec_first - 1) * 100
                        sector_returns.append(sec_return)
                except Exception as e:
                    logger.debug(f"Failed to get data for {sec_symbol}: {e}")
                    continue

            if len(sector_returns) == 0:
                return None, "섹터데이터없음", {}

            sector_avg = float(np.mean(sector_returns))
            relative_strength = stock_return - sector_avg

            results = {
                'stock_return': stock_return,
                'sector_avg': sector_avg,
                'relative_strength': relative_strength,
                'sector_count': len(sector_returns)
            }

            # 판단
            if relative_strength > 5:
                return True, f"강함(+{relative_strength:.1f}%)", results
            elif relative_strength > 0:
                return None, f"중립(+{relative_strength:.1f}%)", results
            else:
                return False, f"약함({relative_strength:.1f}%)", results

        except Exception as e:
            logger.error(f"Relative strength check error for {symbol}: {e}")
            return None, f"오류({str(e)})", {}

    def check_relative_strength_with_mapping(self, symbol, period=60):
        """
        섹터 매핑을 사용한 상대 강도 비교

        Args:
            symbol: 종목 심볼
            period: 비교 기간 (일)

        Returns:
            (bool, str, dict): (통과 여부, 메시지, 상세 정보)
        """
        try:
            # 섹터 확인
            sector = get_sector(symbol)

            if sector == 'Unknown':
                return None, "섹터미분류", {}

            # 동료 종목 가져오기
            peers = get_sector_peers(symbol)

            if not peers:
                return None, "섹터동료없음", {}

            # 상대 강도 비교
            return self.check_relative_strength(symbol, peers, period)

        except Exception as e:
            logger.error(f"Relative strength with mapping error for {symbol}: {e}")
            return None, f"오류({str(e)})", {}

    # ==================== 볼륨 프로파일 ====================

    def check_volume_profile(self, data):
        """
        볼륨 프로파일 분석

        Args:
            data: OHLCV 데이터프레임

        Returns:
            (bool, str, dict): (신호 여부, 메시지, 상세 정보)
        """
        return self.volume_analyzer.analyze_volume_profile(data)

    def check_volume_breakout(self, data, threshold=2.0):
        """
        거래량 돌파 확인

        Args:
            data: OHLCV 데이터프레임
            threshold: 평균 대비 배수

        Returns:
            (bool, str, dict): (돌파 여부, 메시지, 상세 정보)
        """
        return self.volume_analyzer.check_volume_breakout(data, threshold)

    # ==================== 지지/저항선 ====================

    def check_support_resistance(self, data):
        """
        지지/저항선 감지

        Args:
            data: OHLCV 데이터프레임

        Returns:
            (list, list, dict): (지지선 리스트, 저항선 리스트, 상세 정보)
        """
        return self.sr_detector.detect_support_resistance(data)

    def check_near_support_resistance(self, data, threshold=0.03):
        """
        현재 가격이 지지/저항선 근처인지 확인

        Args:
            data: OHLCV 데이터프레임
            threshold: 근접 판단 기준 (3%)

        Returns:
            (bool, str, dict): (신호 여부, 메시지, 상세 정보)
        """
        return self.sr_detector.check_near_support_resistance(data, threshold)

    # ==================== 통합 스크리닝 ====================

    def run_advanced_screening(self, symbol, data, conditions):
        """
        고급 조건 통합 스크리닝

        Args:
            symbol: 종목 심볼
            data: 일봉 데이터
            conditions: 활성화된 조건 리스트

        Returns:
            (bool, str, dict): (통과 여부, 신호명, 상세 결과)
        """
        try:
            results = {}
            passed = True
            messages = []

            # 1. 다중 시간대
            if 'multi_timeframe' in conditions:
                mtf_ok, mtf_msg, mtf_data = self.check_multi_timeframe(symbol)
                results['multi_timeframe'] = mtf_data
                if not mtf_ok:
                    passed = False
                messages.append(mtf_msg)

            # 2. 시장 강도
            if 'market_strength' in conditions:
                ms_ok, ms_msg, ms_data = self.check_market_strength()
                results['market_strength'] = ms_data
                if ms_ok is False:  # None은 통과
                    passed = False
                messages.append(ms_msg)

            # 3. 상대 강도 (섹터 정보 필요 - 단순화)
            if 'relative_strength' in conditions:
                # TODO: 섹터별 종목 리스트 구현 필요
                # 현재는 스킵
                messages.append("상대강도(스킵)")

            # 종합 판단
            if passed:
                signal = "고급스크리닝통과"
                return True, signal, results
            else:
                signal = "고급조건미달"
                return False, signal, results

        except Exception as e:
            logger.error(f"Advanced screening error for {symbol}: {e}")
            return False, f"오류({str(e)})", {}


# ==================== 헬퍼 함수 ====================

def create_advanced_engine():
    """고급 엔진 인스턴스 생성"""
    return AdvancedScreeningEngine()


# ==================== 테스트 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Advanced Screening Engine Test")
    print("=" * 60)

    engine = create_advanced_engine()

    # 1. 다중 시간대 테스트
    print("\n1. 다중 시간대 확인 테스트")
    result, msg, data = engine.check_multi_timeframe('AAPL')
    print(f"   AAPL: {msg}")
    print(f"   상세: {data}")

    # 2. 시장 강도 테스트
    print("\n2. 시장 강도 확인 테스트")
    result, msg, data = engine.check_market_strength()
    print(f"   시장: {msg}")
    print(f"   상세: {data}")

    print("\n" + "=" * 60)
    print("✅ 테스트 완료")
