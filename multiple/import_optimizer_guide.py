"""
Import Optimization Guide
이 파일은 프로젝트의 import 최적화 가이드와 권장 패턴을 제공합니다.
"""

# ============================================================================
# 1. 권장 Import 패턴 (Best Practices)
# ============================================================================

# ✅ GOOD: 명시적 import (Explicit imports)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton,
    QLabel, QVBoxLayout, QHBoxLayout, QTableWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QColor

# ❌ BAD: Wildcard imports
# from PyQt5.QtWidgets import *
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *

# ============================================================================
# 2. 표준 라이브러리 Import (Standard Library)
# ============================================================================

# 그룹별로 정리 (PEP 8 권장)
import os
import sys
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ============================================================================
# 3. 서드파티 라이브러리 Import (Third-party Libraries)
# ============================================================================

import numpy as np
import pandas as pd
import yfinance as yf

# 선택적 import (Optional imports with availability check)
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# ============================================================================
# 4. 로컬 모듈 Import (Local Modules)
# ============================================================================

# 프로젝트 내부 모듈은 마지막에
from cache_manager import get_stock_data, get_ticker_info, get_cache_instance
from unified_search import search_stocks, clear_search_cache
from technical_indicators import get_indicators
from csv_manager import read_csv, write_csv, get_csv_manager

# ============================================================================
# 5. Lazy Import 패턴 (필요한 시점에만 import)
# ============================================================================

def heavy_computation():
    """무거운 라이브러리는 함수 내부에서 import"""
    # ❌ BAD: 파일 상단에서 import
    # import tensorflow as tf  # 사용하지 않을 수도 있는데 항상 로드됨

    # ✅ GOOD: 필요할 때만 import
    import tensorflow as tf
    # ... 연산 수행 ...
    return result


# ============================================================================
# 6. Import 정리 규칙 (Import Organization Rules)
# ============================================================================

"""
Import 순서 (PEP 8):
1. 표준 라이브러리
2. 서드파티 라이브러리
3. 로컬 모듈

각 그룹 사이에 빈 줄 하나씩 추가
"""

# ============================================================================
# 7. 기존 코드 최적화 예시
# ============================================================================

# ❌ BEFORE (비효율적)
"""
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pandas as pd
import numpy as np

def some_function():
    import os  # 함수 내부 import (간단한 라이브러리는 상단에)
    import sys
    # ...
"""

# ✅ AFTER (최적화)
"""
import os
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QPushButton, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

def some_function():
    # 이미 상단에 import됨
    # ...
"""

# ============================================================================
# 8. 조건부 Import 최적화
# ============================================================================

# ✅ GOOD: 버전 체크와 함께
import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

# ============================================================================
# 9. 순환 Import 방지 (Avoiding Circular Imports)
# ============================================================================

# 순환 import 문제가 있을 때:
# 1. 타입 힌팅만 필요한 경우 TYPE_CHECKING 사용

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chart_window import ChartWindow  # 타입 힌트용

# 2. 함수 내부에서 import
def show_chart():
    from chart_window import ChartWindow  # 실제 사용 시점에 import
    chart = ChartWindow()
    chart.show()

# ============================================================================
# 10. Import 별칭 사용 (Import Aliases)
# ============================================================================

# ✅ GOOD: 명확한 별칭
import numpy as np  # 관례적 별칭
import pandas as pd
import matplotlib.pyplot as plt

# ❌ BAD: 혼란스러운 별칭
# import numpy as n  # 너무 짧음
# import pandas as dataframe  # 너무 김

# ============================================================================
# 프로젝트별 Import 최적화 체크리스트
# ============================================================================

"""
□ Wildcard import (*) 제거
□ 사용하지 않는 import 제거
□ Import 순서 정리 (표준 > 서드파티 > 로컬)
□ 명시적 import 사용
□ 선택적 라이브러리는 try-except로 처리
□ 무거운 라이브러리는 lazy import 고려
□ 순환 import 확인 및 해결
□ 함수 내부 import 최소화 (간단한 라이브러리는 상단에)
"""

# ============================================================================
# 자동 Import 정리 도구
# ============================================================================

def analyze_imports(file_path: str):
    """파일의 import 분석 및 최적화 제안"""
    import ast
    import os

    if not os.path.exists(file_path):
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                if alias.name == '*':
                    print(f"⚠️ Wildcard import found: from {module} import *")
                imports.append(f"from {module} import {alias.name}")

    print(f"\n📊 Total imports: {len(imports)}")
    return imports


if __name__ == "__main__":
    # 사용 예시
    print("Import Optimization Guide for Stock Screener Project")
    print("=" * 60)

    # 프로젝트의 주요 파일들 분석
    files_to_check = [
        'enhanced_screener.py',
        'screener.py',
        'prediction_window.py',
        'chart_window.py',
    ]

    for file in files_to_check:
        if os.path.exists(file):
            print(f"\n분석: {file}")
            analyze_imports(file)
