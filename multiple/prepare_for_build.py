# prepare_for_build.py - 빌드 전 실행할 스크립트 (개선된 버전)
import os
import sys
import pandas as pd
import argparse
from pathlib import Path

def create_all_master_csvs(force_overwrite=False, quiet=False):
    """빌드용 마스터 CSV 파일들 생성
    
    Args:
        force_overwrite (bool): True면 기존 파일 덮어쓰기, False면 건너뛰기
        quiet (bool): True면 출력 최소화
    
    Returns:
        dict: 생성된 파일 정보
    """
    if not quiet:
        print("🗂️ 빌드용 마스터 CSV 생성 중...")
    
    os.makedirs('stock_data', exist_ok=True)
    
    results = {}
    
    # 각 국가별 마스터 CSV 생성 (중복 체크 포함)
    results['korea'] = create_korea_master_backup(force_overwrite, quiet)
    results['usa'] = create_usa_master_backup(force_overwrite, quiet) 
    results['sweden'] = create_sweden_master_backup(force_overwrite, quiet)
    
    if not quiet:
        print("✅ 모든 마스터 CSV 생성 완료!")
    
    return results

def check_and_create_csv(df, file_path, description, force_overwrite=False, quiet=False):
    """CSV 파일 존재 여부 확인 후 생성
    
    Args:
        df (DataFrame): 저장할 데이터프레임
        file_path (str): 저장할 파일 경로
        description (str): 파일 설명 (로그용)
        force_overwrite (bool): 강제 덮어쓰기 여부
        quiet (bool): 출력 최소화 여부
    
    Returns:
        bool: 파일이 생성되었으면 True, 건너뛰었으면 False
    """
    if os.path.exists(file_path) and not force_overwrite:
        if not quiet:
            print(f"⭐️ {description} - 이미 존재함 (건너뜀): {file_path}")
        return False
    
    try:
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        status = "덮어씀" if os.path.exists(file_path) and force_overwrite else "생성됨"
        if not quiet:
            print(f"✅ {description} - {status}: {file_path}")
        return True
    except Exception as e:
        if not quiet:
            print(f"❌ {description} - 생성 실패: {e}")
        return False

def create_korea_master_backup(force_overwrite=False, quiet=False):
    """한국 마스터 CSV (100개)"""
    korea_data = [
        # 시총 상위 100개 (2024년 기준, 단위: 원)
        ('005930.KS', '삼성전자', '반도체', 300000000000000, 'KOSPI'),
        ('000660.KS', 'SK하이닉스', '반도체', 80000000000000, 'KOSPI'),
        ('035420.KS', '네이버', 'IT서비스', 40000000000000, 'KOSPI'),
        ('207940.KS', '삼성바이오로직스', '바이오', 35000000000000, 'KOSPI'),
        ('006400.KS', '삼성SDI', '배터리', 30000000000000, 'KOSPI'),
        ('051910.KS', 'LG화학', '화학', 28000000000000, 'KOSPI'),
        ('035720.KS', '카카오', 'IT서비스', 25000000000000, 'KOSPI'),
        ('068270.KS', '셀트리온', '바이오', 24000000000000, 'KOSPI'),
        ('005380.KS', '현대차', '자동차', 22000000000000, 'KOSPI'),
        ('373220.KS', 'LG에너지솔루션', '배터리', 20000000000000, 'KOSPI'),
        
        ('323410.KS', '카카오뱅크', '금융', 18000000000000, 'KOSPI'),
        ('000270.KS', '기아', '자동차', 17000000000000, 'KOSPI'),
        ('066570.KS', 'LG전자', '전자', 16000000000000, 'KOSPI'),
        ('003550.KS', 'LG', '지주회사', 15000000000000, 'KOSPI'),
        ('015760.KS', '한국전력', '전력', 14000000000000, 'KOSPI'),
        ('017670.KS', 'SK텔레콤', '통신', 13000000000000, 'KOSPI'),
        ('034730.KS', 'SK', '지주회사', 12000000000000, 'KOSPI'),
        ('096770.KS', 'SK이노베이션', '에너지', 11000000000000, 'KOSPI'),
        ('086790.KS', '하나금융지주', '금융', 10000000000000, 'KOSPI'),
        ('105560.KS', 'KB금융', '금융', 9500000000000, 'KOSPI'),
        
        # 추가 80개 종목들...
        ('012330.KS', '현대모비스', '자동차부품', 9000000000000, 'KOSPI'),
        ('032830.KS', '삼성생명', '보험', 8800000000000, 'KOSPI'),
        ('009150.KS', '삼성전기', '전자부품', 8500000000000, 'KOSPI'),
        ('000810.KS', '삼성화재', '보험', 8200000000000, 'KOSPI'),
        ('251270.KS', '넷마블', '게임', 8000000000000, 'KOSPI'),
        
        # KOSDAQ 상위 종목들
        ('042700.KQ', '한미반도체', '반도체', 1500000000000, 'KOSDAQ'),
        ('065350.KQ', '신성통상', '섬유', 1200000000000, 'KOSDAQ'),
        ('058470.KQ', '리노공업', '반도체', 1000000000000, 'KOSDAQ'),
        ('067310.KQ', '하나마이크론', '반도체', 900000000000, 'KOSDAQ'),
        ('137310.KQ', '에스디바이오센서', '바이오', 800000000000, 'KOSDAQ'),
        
        # 나머지 종목들을 위한 더미 데이터
        *[(f"{6000+i:06d}.KS", f"종목{i}", "기타", 1000000000*(100-i), 'KOSPI') 
          for i in range(70)]  # 총 100개가 되도록 조정
    ]
    
    df = pd.DataFrame(korea_data, columns=['ticker', 'name', 'sector', 'market_cap', 'exchange'])
    
    # 마스터 CSV와 작업용 CSV 모두 체크해서 생성
    master_created = check_and_create_csv(df, 'stock_data/korea_stocks_master.csv', '한국 마스터 CSV', force_overwrite, quiet)
    work_created = check_and_create_csv(df.head(20), 'stock_data/korea_stocks.csv', '한국 작업용 CSV', force_overwrite, quiet)
    
    return {'master': master_created, 'work': work_created, 'count': len(df)}

def create_usa_master_backup(force_overwrite=False, quiet=False):
    """나스닥 마스터 CSV (100개)"""
    usa_top_100 = [
        # 시총 상위 100개 (2024년 기준, 단위: USD)
        ('AAPL', 'Apple Inc', 'Technology', 3000000000000, 'NASDAQ'),
        ('MSFT', 'Microsoft Corp', 'Technology', 2800000000000, 'NASDAQ'),
        ('GOOGL', 'Alphabet Inc Class A', 'Technology', 1700000000000, 'NASDAQ'),
        ('AMZN', 'Amazon.com Inc', 'Consumer Discretionary', 1500000000000, 'NASDAQ'),
        ('NVDA', 'NVIDIA Corp', 'Technology', 1900000000000, 'NASDAQ'),
        ('TSLA', 'Tesla Inc', 'Consumer Discretionary', 800000000000, 'NASDAQ'),
        ('META', 'Meta Platforms Inc', 'Technology', 750000000000, 'NASDAQ'),
        ('BRK-B', 'Berkshire Hathaway Inc Class B', 'Financial Services', 700000000000, 'NYSE'),
        ('UNH', 'UnitedHealth Group Inc', 'Healthcare', 450000000000, 'NYSE'),
        ('JNJ', 'Johnson & Johnson', 'Healthcare', 420000000000, 'NYSE'),
        
        ('V', 'Visa Inc Class A', 'Financial Services', 400000000000, 'NYSE'),
        ('PG', 'Procter & Gamble Co', 'Consumer Staples', 380000000000, 'NYSE'),
        ('JPM', 'JPMorgan Chase & Co', 'Financial Services', 450000000000, 'NYSE'),
        ('HD', 'Home Depot Inc', 'Consumer Discretionary', 350000000000, 'NYSE'),
        ('MA', 'Mastercard Inc Class A', 'Financial Services', 340000000000, 'NYSE'),
        ('BAC', 'Bank of America Corp', 'Financial Services', 300000000000, 'NYSE'),
        ('XOM', 'Exxon Mobil Corp', 'Energy', 280000000000, 'NYSE'),
        ('CVX', 'Chevron Corp', 'Energy', 270000000000, 'NYSE'),
        ('ABBV', 'AbbVie Inc', 'Healthcare', 260000000000, 'NYSE'),
        ('WMT', 'Walmart Inc', 'Consumer Staples', 450000000000, 'NYSE'),
        
        # 나머지 80개를 위한 추가 데이터
        *[(f"STCK{i:02d}", f"Stock {i}", "Technology", 1000000000*(80-i), 'NASDAQ') 
          for i in range(80)]
    ]
    
    df = pd.DataFrame(usa_top_100, columns=['ticker', 'name', 'sector', 'market_cap', 'exchange'])
    
    master_created = check_and_create_csv(df, 'stock_data/usa_stocks_master.csv', '미국 마스터 CSV', force_overwrite, quiet)
    work_created = check_and_create_csv(df.head(20), 'stock_data/usa_stocks.csv', '미국 작업용 CSV', force_overwrite, quiet)
    
    return {'master': master_created, 'work': work_created, 'count': len(df)}

def create_sweden_master_backup(force_overwrite=False, quiet=False):
    """스웨덴 마스터 CSV (100개)"""
    sweden_top_100 = [
        # 시총 상위 종목들 (2024년 기준, 단위: SEK)
        ('INVE-B.ST', 'Investor AB Class B', 'Financial Services', 800000000000, 'OMX Stockholm'),
        ('VOLV-B.ST', 'Volvo AB Class B', 'Industrials', 450000000000, 'OMX Stockholm'),
        ('SAND.ST', 'Sandvik AB', 'Industrials', 400000000000, 'OMX Stockholm'),
        ('ATCO-A.ST', 'Atlas Copco AB Class A', 'Industrials', 400000000000, 'OMX Stockholm'),
        ('ASSA-B.ST', 'ASSA ABLOY AB Class B', 'Industrials', 350000000000, 'OMX Stockholm'),
        ('HEXA-B.ST', 'Hexagon AB Class B', 'Technology', 350000000000, 'OMX Stockholm'),
        ('SWED-A.ST', 'Swedbank AB Class A', 'Financial Services', 300000000000, 'OMX Stockholm'),
        ('ERIC-B.ST', 'Telefonaktiebolaget LM Ericsson Class B', 'Technology', 300000000000, 'OMX Stockholm'),
        ('ALFA.ST', 'Alfa Laval AB', 'Industrials', 300000000000, 'OMX Stockholm'),
        ('SEB-A.ST', 'Skandinaviska Enskilda Banken AB Class A', 'Financial Services', 280000000000, 'OMX Stockholm'),
        
        # 나머지 90개를 위한 추가 데이터
        *[(f"SWE{i:02d}.ST", f"Swedish Stock {i}", "Industrials", 1000000000*(90-i), 'OMX Stockholm') 
          for i in range(90)]
    ]
    
    df = pd.DataFrame(sweden_top_100, columns=['ticker', 'name', 'sector', 'market_cap', 'exchange'])
    
    master_created = check_and_create_csv(df, 'stock_data/sweden_stocks_master.csv', '스웨덴 마스터 CSV', force_overwrite, quiet)
    work_created = check_and_create_csv(df.head(20), 'stock_data/sweden_stocks.csv', '스웨덴 작업용 CSV', force_overwrite, quiet)
    
    return {'master': master_created, 'work': work_created, 'count': len(df)}

def check_required_files():
    """필수 CSV 파일들이 존재하는지 확인"""
    required_files = [
        'stock_data/korea_stocks.csv',
        'stock_data/usa_stocks.csv',
        'stock_data/sweden_stocks.csv'
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    return existing_files, missing_files

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='빌드용 마스터 CSV 파일 생성')
    parser.add_argument('--force', '-f', action='store_true', 
                        help='기존 파일이 있어도 덮어쓰기')
    parser.add_argument('--check', '-c', action='store_true',
                        help='파일 존재 여부만 확인')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='출력 최소화')
    
    args = parser.parse_args()
    
    if args.check:
        existing_files, missing_files = check_required_files()
        
        if not args.quiet:
            print("📋 파일 존재 여부 확인:")
            for file_path in existing_files:
                print(f"  ✅ 존재: {file_path}")
            for file_path in missing_files:
                print(f"  ❌ 없음: {file_path}")
        
        # 누락된 파일이 있으면 종료 코드 1 반환
        return 1 if missing_files else 0
    else:
        results = create_all_master_csvs(force_overwrite=args.force, quiet=args.quiet)
        
        if not args.quiet:
            print("\n📊 생성 결과 요약:")
            total_created = 0
            for market, result in results.items():
                created_count = sum([result['master'], result['work']])
                total_created += created_count
                print(f"  {market}: {created_count}개 파일 생성/확인 (총 {result['count']}개 종목)")
            
            print(f"\n✅ 총 {total_created}개 파일 처리 완료!")
        
        return 0

if __name__ == "__main__":
    sys.exit(main())