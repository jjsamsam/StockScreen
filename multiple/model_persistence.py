#!/usr/bin/env python3
"""
model_persistence.py
ML 모델 저장/로드 및 증분 학습 시스템
- LSTM/Transformer: .h5 (Keras HDF5)
- XGBoost/LightGBM/RF: .pkl (Pickle)
- 메타데이터: .json
- 버전 관리 및 성능 추적
"""

import os
import json
import pickle
import joblib
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import hashlib

from logger_config import get_logger
logger = get_logger(__name__)


class ModelPersistence:
    """모델 저장/로드 및 버전 관리 시스템"""

    def __init__(self, base_dir: str = "models"):
        """
        Args:
            base_dir: 모델 저장 기본 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def _get_model_dir(self, ticker: str) -> Path:
        """티커별 모델 디렉토리 생성/반환"""
        model_dir = self.base_dir / ticker
        model_dir.mkdir(exist_ok=True)
        return model_dir

    def _generate_version(self) -> str:
        """버전 생성 (타임스탬프 기반)"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_model_path(self, ticker: str, model_type: str, version: Optional[str] = None) -> Path:
        """모델 파일 경로 반환"""
        model_dir = self._get_model_dir(ticker)

        if version is None:
            # 최신 버전 찾기
            pattern = f"{model_type}_*.pkl" if model_type in ['xgboost', 'lightgbm', 'random_forest'] else f"{model_type}_*.h5"
            existing = sorted(model_dir.glob(pattern))
            if existing:
                return existing[-1]
            else:
                version = self._generate_version()

        # 파일 확장자 결정
        if model_type in ['lstm', 'transformer']:
            ext = 'h5'
        else:
            ext = 'pkl'

        return model_dir / f"{model_type}_{version}.{ext}"

    def _get_metadata_path(self, model_path: Path) -> Path:
        """메타데이터 파일 경로"""
        return model_path.with_suffix('.json')

    # ========== Keras 모델 (LSTM/Transformer) ==========

    def save_keras_model(self, model, ticker: str, model_type: str,
                        metadata: Optional[Dict[str, Any]] = None,
                        scaler=None) -> str:
        """
        Keras 모델 저장 (.h5)

        Args:
            model: Keras 모델
            ticker: 주식 티커
            model_type: 'lstm' 또는 'transformer'
            metadata: 추가 메타데이터 (성능, 파라미터 등)
            scaler: MinMaxScaler 객체

        Returns:
            저장된 파일 경로
        """
        version = self._generate_version()
        model_path = self._get_model_path(ticker, model_type, version)

        try:
            # 모델 저장
            model.save(str(model_path))
            logger.info(f"✅ {model_type.upper()} 모델 저장: {model_path}")

            # Scaler 저장
            if scaler is not None:
                scaler_path = model_path.with_suffix('.scaler.pkl')
                joblib.dump(scaler, scaler_path)
                logger.info(f"✅ Scaler 저장: {scaler_path}")

            # 메타데이터 저장
            meta = {
                'ticker': ticker,
                'model_type': model_type,
                'version': version,
                'saved_at': datetime.now().isoformat(),
                'framework': 'tensorflow/keras',
                'file_path': str(model_path)
            }

            if metadata:
                meta.update(metadata)

            meta_path = self._get_metadata_path(model_path)
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            logger.info(f"✅ 메타데이터 저장: {meta_path}")

            # 훈련 히스토리 저장
            self._save_training_history(ticker, model_type, version, metadata)

            return str(model_path)

        except Exception as e:
            logger.error(f"❌ Keras 모델 저장 실패: {e}")
            raise

    def load_keras_model(self, ticker: str, model_type: str,
                        version: Optional[str] = None) -> tuple:
        """
        Keras 모델 로드 (.h5)

        Returns:
            (model, metadata, scaler)
        """
        try:
            # TensorFlow Lazy Import
            import tensorflow as tf
            from tensorflow import keras

            model_path = self._get_model_path(ticker, model_type, version)

            if not model_path.exists():
                logger.warning(f"모델 파일 없음: {model_path}")
                return None, None, None

            # 모델 로드
            model = keras.models.load_model(str(model_path))
            logger.info(f"✅ {model_type.upper()} 모델 로드: {model_path}")

            # Scaler 로드
            scaler = None
            scaler_path = model_path.with_suffix('.scaler.pkl')
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                logger.info(f"✅ Scaler 로드: {scaler_path}")

            # 메타데이터 로드
            meta_path = self._get_metadata_path(model_path)
            metadata = None
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"✅ 메타데이터 로드: {meta_path}")

            return model, metadata, scaler

        except Exception as e:
            logger.error(f"❌ Keras 모델 로드 실패: {e}")
            return None, None, None

    # ========== Scikit-learn/XGBoost/LightGBM 모델 ==========

    def save_sklearn_model(self, model, ticker: str, model_type: str,
                          metadata: Optional[Dict[str, Any]] = None,
                          scaler=None) -> str:
        """
        Scikit-learn/XGBoost/LightGBM 모델 저장 (.pkl)

        Args:
            model: 모델 객체
            ticker: 주식 티커
            model_type: 'xgboost', 'lightgbm', 'random_forest'
            metadata: 추가 메타데이터
            scaler: Scaler 객체

        Returns:
            저장된 파일 경로
        """
        version = self._generate_version()
        model_path = self._get_model_path(ticker, model_type, version)

        try:
            # 모델 저장 (joblib이 pickle보다 빠름)
            joblib.dump(model, model_path)
            logger.info(f"✅ {model_type.upper()} 모델 저장: {model_path}")

            # Scaler 저장
            if scaler is not None:
                scaler_path = model_path.with_suffix('.scaler.pkl')
                joblib.dump(scaler, scaler_path)
                logger.info(f"✅ Scaler 저장: {scaler_path}")

            # 메타데이터 저장
            meta = {
                'ticker': ticker,
                'model_type': model_type,
                'version': version,
                'saved_at': datetime.now().isoformat(),
                'framework': self._get_framework_name(model),
                'file_path': str(model_path)
            }

            if metadata:
                meta.update(metadata)

            # Feature importance 저장 (가능한 경우)
            if hasattr(model, 'feature_importances_'):
                meta['has_feature_importance'] = True

            meta_path = self._get_metadata_path(model_path)
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            logger.info(f"✅ 메타데이터 저장: {meta_path}")

            # 훈련 히스토리 저장
            self._save_training_history(ticker, model_type, version, metadata)

            return str(model_path)

        except Exception as e:
            logger.error(f"❌ {model_type} 모델 저장 실패: {e}")
            raise

    def load_sklearn_model(self, ticker: str, model_type: str,
                          version: Optional[str] = None) -> tuple:
        """
        Scikit-learn/XGBoost/LightGBM 모델 로드 (.pkl)

        Returns:
            (model, metadata, scaler)
        """
        try:
            model_path = self._get_model_path(ticker, model_type, version)

            if not model_path.exists():
                logger.warning(f"모델 파일 없음: {model_path}")
                return None, None, None

            # 모델 로드
            model = joblib.load(model_path)
            logger.info(f"✅ {model_type.upper()} 모델 로드: {model_path}")

            # Scaler 로드
            scaler = None
            scaler_path = model_path.with_suffix('.scaler.pkl')
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                logger.info(f"✅ Scaler 로드: {scaler_path}")

            # 메타데이터 로드
            meta_path = self._get_metadata_path(model_path)
            metadata = None
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"✅ 메타데이터 로드: {meta_path}")

            return model, metadata, scaler

        except Exception as e:
            logger.error(f"❌ {model_type} 모델 로드 실패: {e}")
            return None, None, None

    # ========== 유틸리티 함수 ==========

    def _get_framework_name(self, model) -> str:
        """모델의 프레임워크 이름 반환"""
        model_class = type(model).__name__
        module = type(model).__module__

        if 'xgboost' in module:
            return 'xgboost'
        elif 'lightgbm' in module:
            return 'lightgbm'
        elif 'sklearn' in module:
            return 'scikit-learn'
        else:
            return 'unknown'

    def _save_training_history(self, ticker: str, model_type: str,
                               version: str, metadata: Optional[Dict] = None):
        """훈련 히스토리 CSV 파일에 추가"""
        model_dir = self._get_model_dir(ticker)
        history_path = model_dir / "training_history.csv"

        # 히스토리 레코드 생성
        record = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'version': version,
        }

        if metadata:
            # 주요 메트릭 추출
            for key in ['train_loss', 'val_loss', 'rmse', 'mae', 'confidence_score']:
                if key in metadata:
                    record[key] = metadata[key]

        # CSV 파일에 추가
        df = pd.DataFrame([record])

        if history_path.exists():
            df.to_csv(history_path, mode='a', header=False, index=False)
        else:
            df.to_csv(history_path, index=False)

        logger.debug(f"훈련 히스토리 업데이트: {history_path}")

    def get_training_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """훈련 히스토리 조회"""
        model_dir = self._get_model_dir(ticker)
        history_path = model_dir / "training_history.csv"

        if history_path.exists():
            return pd.read_csv(history_path)
        else:
            return None

    def list_models(self, ticker: str) -> List[Dict[str, Any]]:
        """저장된 모델 목록 조회"""
        model_dir = self._get_model_dir(ticker)

        if not model_dir.exists():
            return []

        models = []
        for meta_file in model_dir.glob("*.json"):
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    models.append(meta)
            except:
                continue

        # 최신순 정렬
        models.sort(key=lambda x: x.get('saved_at', ''), reverse=True)
        return models

    def delete_old_models(self, ticker: str, keep_latest: int = 5):
        """오래된 모델 삭제 (최신 N개만 유지)"""
        models = self.list_models(ticker)

        # 모델 타입별로 그룹화
        by_type = {}
        for model in models:
            mtype = model['model_type']
            if mtype not in by_type:
                by_type[mtype] = []
            by_type[mtype].append(model)

        # 각 타입별로 오래된 것 삭제
        deleted_count = 0
        for mtype, mlist in by_type.items():
            if len(mlist) > keep_latest:
                to_delete = mlist[keep_latest:]
                for model_meta in to_delete:
                    try:
                        # 모델 파일 삭제
                        model_path = Path(model_meta['file_path'])
                        if model_path.exists():
                            model_path.unlink()

                        # 메타데이터 삭제
                        meta_path = self._get_metadata_path(model_path)
                        if meta_path.exists():
                            meta_path.unlink()

                        # Scaler 삭제
                        scaler_path = model_path.with_suffix('.scaler.pkl')
                        if scaler_path.exists():
                            scaler_path.unlink()

                        deleted_count += 1
                        logger.info(f"🗑️  오래된 모델 삭제: {model_path.name}")

                    except Exception as e:
                        logger.error(f"모델 삭제 실패: {e}")

        logger.info(f"✅ {deleted_count}개 오래된 모델 삭제 완료")
        return deleted_count

    # ========== 증분 학습 (Incremental Learning) ==========

    def supports_incremental_learning(self, model_type: str) -> bool:
        """모델이 증분 학습을 지원하는지 확인"""
        # XGBoost, LightGBM은 기존 모델에서 계속 학습 가능
        incremental_models = ['xgboost', 'lightgbm']
        return model_type in incremental_models

    def incremental_train_xgboost(self, ticker: str, X_new, y_new,
                                  n_estimators_add: int = 50) -> Any:
        """
        XGBoost 증분 학습

        Args:
            ticker: 주식 티커
            X_new: 새로운 학습 데이터
            y_new: 새로운 타겟
            n_estimators_add: 추가할 트리 개수

        Returns:
            업데이트된 모델
        """
        try:
            import xgboost as xgb

            # 기존 모델 로드
            model, metadata, scaler = self.load_sklearn_model(ticker, 'xgboost')

            if model is None:
                logger.warning("기존 XGBoost 모델 없음, 새로 훈련해야 합니다")
                return None

            logger.info(f"XGBoost 증분 학습 시작... (기존 트리: {model.n_estimators})")

            # 새 데이터로 추가 학습
            # XGBoost는 xgb_model 파라미터로 기존 모델 전달
            new_model = xgb.XGBRegressor(
                n_estimators=model.n_estimators + n_estimators_add,
                max_depth=model.max_depth,
                learning_rate=model.learning_rate,
                random_state=42,
                verbosity=0
            )

            # Scaler 적용 (있으면)
            if scaler is not None:
                X_new_scaled = scaler.transform(X_new)
            else:
                X_new_scaled = X_new

            # 기존 모델에서 계속 학습
            new_model.fit(X_new_scaled, y_new, xgb_model=model.get_booster())

            logger.info(f"✅ XGBoost 증분 학습 완료 (총 트리: {new_model.n_estimators})")

            # 업데이트된 모델 저장
            new_metadata = metadata.copy() if metadata else {}
            new_metadata['incremental_learning'] = True
            new_metadata['previous_version'] = metadata.get('version', 'unknown')
            new_metadata['trees_added'] = n_estimators_add

            self.save_sklearn_model(new_model, ticker, 'xgboost', new_metadata, scaler)

            return new_model

        except Exception as e:
            logger.error(f"❌ XGBoost 증분 학습 실패: {e}")
            return None

    def incremental_train_lightgbm(self, ticker: str, X_new, y_new,
                                   n_estimators_add: int = 50) -> Any:
        """
        LightGBM 증분 학습
        """
        try:
            import lightgbm as lgb

            # 기존 모델 로드
            model, metadata, scaler = self.load_sklearn_model(ticker, 'lightgbm')

            if model is None:
                logger.warning("기존 LightGBM 모델 없음, 새로 훈련해야 합니다")
                return None

            logger.info(f"LightGBM 증분 학습 시작... (기존 트리: {model.n_estimators})")

            # 새 데이터로 추가 학습
            new_model = lgb.LGBMRegressor(
                n_estimators=model.n_estimators + n_estimators_add,
                max_depth=model.max_depth,
                learning_rate=model.learning_rate,
                random_state=42,
                verbosity=-1
            )

            # Scaler 적용
            if scaler is not None:
                X_new_scaled = scaler.transform(X_new)
            else:
                X_new_scaled = X_new

            # 기존 모델에서 계속 학습
            new_model.fit(
                X_new_scaled, y_new,
                init_model=model.booster_
            )

            logger.info(f"✅ LightGBM 증분 학습 완료 (총 트리: {new_model.n_estimators})")

            # 업데이트된 모델 저장
            new_metadata = metadata.copy() if metadata else {}
            new_metadata['incremental_learning'] = True
            new_metadata['previous_version'] = metadata.get('version', 'unknown')
            new_metadata['trees_added'] = n_estimators_add

            self.save_sklearn_model(new_model, ticker, 'lightgbm', new_metadata, scaler)

            return new_model

        except Exception as e:
            logger.error(f"❌ LightGBM 증분 학습 실패: {e}")
            return None


# ========== 전역 인스턴스 ==========
_model_persistence = None

def get_model_persistence() -> ModelPersistence:
    """전역 ModelPersistence 인스턴스 반환 (싱글톤)"""
    global _model_persistence
    if _model_persistence is None:
        _model_persistence = ModelPersistence()
    return _model_persistence


if __name__ == "__main__":
    # 테스트 코드
    persistence = ModelPersistence()

    print("=== Model Persistence 테스트 ===")
    print(f"Base directory: {persistence.base_dir}")

    # 모델 목록 조회 예제
    models = persistence.list_models("AAPL")
    print(f"\n저장된 AAPL 모델 개수: {len(models)}")

    if models:
        print("\n최근 모델:")
        for model in models[:3]:
            print(f"  - {model['model_type']}: {model['version']} ({model['saved_at']})")
