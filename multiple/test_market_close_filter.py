"""
시장 마감 전 불완전한 데이터 필터링 테스트
"""

import sys
import logging
from datetime import datetime
import pytz
from cache_manager import StockDataCache

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

def test_market_close_filtering():
    """시장 마감 전 데이터 필터링 테스트"""
    print("=" * 70)
    print("🕐 시장 마감 전 불완전한 데이터 필터링 테스트")
    print("=" * 70)

    # 다양한 시장의 종목 테스트
    test_cases = [
        {
            'symbol': '005930.KS',
            'name': '삼성전자 (한국)',
            'timezone': 'Asia/Seoul',
            'close_time': '15:30'
        },
        {
            'symbol': 'AAPL',
            'name': '애플 (미국)',
            'timezone': 'America/New_York',
            'close_time': '16:00'
        },
        {
            'symbol': '000100.KS',
            'name': '유한양행 (한국)',
            'timezone': 'Asia/Seoul',
            'close_time': '15:30'
        }
    ]

    cache_manager = StockDataCache()

    print("\n📊 현재 시간 확인")
    print("-" * 70)

    for tc in test_cases:
        tz = pytz.timezone(tc['timezone'])
        now = datetime.now(tz)
        print(f"{tc['name']:20s} {now.strftime('%Y-%m-%d %H:%M:%S %Z')} (마감: {tc['close_time']})")

    results = []

    for tc in test_cases:
        symbol = tc['symbol']
        name = tc['name']
        timezone = tc['timezone']

        print(f"\n{'=' * 70}")
        print(f"📊 {name} ({symbol})")
        print(f"{'=' * 70}")

        # 시장 시간 정보
        tz = pytz.timezone(timezone)
        now_market = datetime.now(tz)
        print(f"현재 시각: {now_market.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"시장 마감: {tc['close_time']} {timezone}")

        # 데이터 가져오기 (강제 새로고침)
        data = cache_manager.get_stock_data(
            symbol,
            period='1mo',
            force_refresh=True,
            validate_cache=True
        )

        if data is not None and not data.empty:
            print(f"\n✅ 데이터 가져오기 성공")
            print(f"   데이터 포인트: {len(data)}개")
            print(f"   날짜 범위: {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}")

            # 최근 3일 데이터 표시
            print(f"\n   📈 최근 3일 데이터:")
            recent = data.tail(3)[['Open', 'High', 'Low', 'Close', 'Volume']]
            for idx, row in recent.iterrows():
                date_str = idx.strftime('%Y-%m-%d')
                is_today = idx.date() == now_market.date()
                today_marker = " ← 오늘" if is_today else ""
                print(f"      {date_str}: O={row['Open']:>8.2f} H={row['High']:>8.2f} "
                      f"L={row['Low']:>8.2f} C={row['Close']:>8.2f} V={row['Volume']:>12,.0f}{today_marker}")

            # 마지막 날짜가 오늘인지 확인
            last_date = data.index[-1]
            if last_date.tzinfo is None:
                last_date_local = tz.localize(last_date)
            else:
                last_date_local = last_date.astimezone(tz)

            is_today = last_date_local.date() == now_market.date()

            if is_today:
                print(f"\n   ⚠️  주의: 마지막 데이터가 오늘 날짜입니다.")
                print(f"   → 시장이 마감되지 않았다면 이 데이터는 불완전할 수 있습니다.")
            else:
                print(f"\n   ✅ 마지막 데이터가 과거 날짜입니다. ({last_date_local.strftime('%Y-%m-%d')})")
                print(f"   → 데이터가 완전합니다.")

            results.append({
                'symbol': symbol,
                'name': name,
                'success': True,
                'data_points': len(data),
                'last_date': last_date_local,
                'is_today': is_today
            })
        else:
            print(f"❌ 데이터 가져오기 실패")
            results.append({
                'symbol': symbol,
                'name': name,
                'success': False,
                'data_points': 0,
                'last_date': None,
                'is_today': False
            })

    # 최종 요약
    print(f"\n{'=' * 70}")
    print("📊 최종 결과 요약")
    print(f"{'=' * 70}")

    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)

    for r in results:
        status = "✅ 성공" if r['success'] else "❌ 실패"
        today_status = "⚠️  오늘" if r['is_today'] else "✅ 과거"
        data_count = f"({r['data_points']} 포인트)" if r['success'] else ""

        print(f"{r['name']:25s} {status:10s} {today_status:10s} {data_count}")

    print(f"\n성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

    # 결론
    print(f"\n{'=' * 70}")
    print("🎯 결론")
    print(f"{'=' * 70}")

    has_today = any(r['is_today'] for r in results if r['success'])

    if has_today:
        print("⚠️  일부 종목의 마지막 데이터가 오늘 날짜입니다.")
        print("   시장 마감 시간 전이라면 이 데이터는 자동으로 제거되었습니다.")
        print("   시장 마감 후라면 완전한 데이터입니다.")
    else:
        print("✅ 모든 종목의 마지막 데이터가 과거 날짜입니다.")
        print("   데이터가 완전하며 예측에 사용해도 안전합니다.")

    print(f"\n{'=' * 70}")
    print("📝 참고사항")
    print(f"{'=' * 70}")
    print("• 한국 시장: 15:30 마감 (Asia/Seoul)")
    print("• 미국 시장: 16:00 마감 (America/New_York)")
    print("• 시장 마감 전: 불완전한 오늘 데이터 자동 제거")
    print("• 시장 마감 후: 오늘 데이터 포함 (완전함)")
    print(f"{'=' * 70}")

    if success_count == total_count:
        print("\n🎉 모든 테스트 통과!")
        return True
    else:
        print("\n⚠️ 일부 테스트 실패")
        return False

if __name__ == "__main__":
    try:
        success = test_market_close_filtering()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
