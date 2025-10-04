"""
logger_config.py
중앙 집중식 로깅 설정
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# 로깅 레벨 매핑
LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# 전역 로깅 레벨 (런타임에 변경 가능)
_current_level = logging.INFO


def setup_logging(level='INFO', log_to_file=False, log_dir='logs'):
    """
    로깅 시스템 초기화

    Args:
        level: 로깅 레벨 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_to_file: 파일로도 로그를 저장할지 여부
        log_dir: 로그 파일 저장 디렉토리
    """
    global _current_level
    _current_level = LEVEL_MAP.get(level.upper(), logging.INFO)

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(_current_level)

    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(_current_level)

    # 포맷 설정 (레벨에 따라 다른 포맷)
    if _current_level == logging.DEBUG:
        # DEBUG 모드: 상세 정보 (시간, 파일명, 라인, 함수명)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d %(funcName)s()] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        # INFO 이상: 간단한 포맷
        formatter = logging.Formatter(
            '%(levelname)s - %(name)s - %(message)s'
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 파일 핸들러 추가 (옵션)
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_path / f'stock_screener_{timestamp}.log'

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(_current_level)

        # 파일에는 항상 상세 포맷 사용
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"로그 파일 생성: {log_file}")


def set_level(level):
    """
    런타임에 로깅 레벨 변경

    Args:
        level: 로깅 레벨 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    global _current_level
    _current_level = LEVEL_MAP.get(level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(_current_level)

    # 모든 핸들러의 레벨도 변경
    for handler in root_logger.handlers:
        handler.setLevel(_current_level)


def get_logger(name):
    """
    모듈별 로거 가져오기

    Args:
        name: 로거 이름 (보통 __name__ 사용)

    Returns:
        logging.Logger: 로거 인스턴스
    """
    return logging.getLogger(name)


# 초기 설정 (INFO 레벨, 콘솔만)
setup_logging(level='INFO', log_to_file=False)
