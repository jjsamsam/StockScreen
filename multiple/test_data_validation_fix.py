"""
데이터 검증 수정 테스트
000100.KS 종목의 데이터 검증 오류를 확인하고 수정된 코드를 테스트합니다.
"""

import sys
import logging
from cache_manager import StockDataCache

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

def test_data_validation():
    """데이터 검증 테스트"""
    print("=" * 60)
    print("🧪 데이터 검증 수정 테스트")
    print("=" * 60)

    # 문제가 있던 종목 테스트
    test_symbols = [
        "000100.KS",  # 문제가 있던 종목
        "005930.KS",  # 삼성전자 (비교용)
        "AAPL",       # 애플 (미국 주식)
    ]

    cache_manager = StockDataCache()

    results = []
    for symbol in test_symbols:
        print(f"\n{'=' * 60}")
        print(f"📊 Testing: {symbol}")
        print(f"{'=' * 60}")

        # 캐시 강제 새로고침 + 검증 활성화
        data = cache_manager.get_stock_data(
            symbol,
            period='6mo',
            force_refresh=True,
            validate_cache=True
        )

        if data is not None and not data.empty:
            print(f"✅ {symbol}: 데이터 검증 통과!")
            print(f"   - 데이터 포인트: {len(data)}개")
            print(f"   - 날짜 범위: {data.index[0]} ~ {data.index[-1]}")

            # 가격 범위 확인
            print(f"   - 가격 범위: ${data['Low'].min():.2f} ~ ${data['High'].max():.2f}")

            # Close가 High/Low 범위 내에 있는지 확인
            close_in_range = ((data['Close'] >= data['Low']) & (data['Close'] <= data['High'])).all()
            print(f"   - Close 범위 검사: {'✅ 통과' if close_in_range else '❌ 실패'}")

            # 샘플 데이터 (최근 5개)
            print(f"\n   📈 최근 5일 데이터:")
            recent = data.tail(5)[['Open', 'High', 'Low', 'Close', 'Volume']]
            for idx, row in recent.iterrows():
                date_str = idx.strftime('%Y-%m-%d')
                print(f"      {date_str}: O={row['Open']:.2f} H={row['High']:.2f} L={row['Low']:.2f} C={row['Close']:.2f}")

                # 검증
                if row['Close'] > row['High'] or row['Close'] < row['Low']:
                    print(f"         ⚠️ WARNING: Close outside High-Low range!")

            results.append((symbol, True, len(data)))
        else:
            print(f"❌ {symbol}: 데이터 가져오기 실패 또는 검증 실패")
            results.append((symbol, False, 0))

    # 최종 요약
    print(f"\n{'=' * 60}")
    print("📊 최종 결과 요약")
    print(f"{'=' * 60}")

    success_count = sum(1 for _, success, _ in results if success)
    total_count = len(results)

    for symbol, success, data_points in results:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{symbol:15s} {status:10s} ({data_points} 데이터 포인트)")

    print(f"\n성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

    if success_count == total_count:
        print("\n🎉 모든 테스트 통과! 데이터 검증 수정이 성공적으로 작동합니다.")
        return True
    else:
        print("\n⚠️ 일부 테스트 실패. 추가 수정이 필요할 수 있습니다.")
        return False

if __name__ == "__main__":
    try:
        success = test_data_validation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
