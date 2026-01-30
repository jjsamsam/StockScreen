"""
task_manager.py
비동기 작업 관리자 - AI 예측 등 CPU 집약적 작업을 백그라운드에서 처리

특징:
- 작업 상태 추적 (pending, running, completed, failed, cancelled)
- 타임아웃 지원 (기본 120초)
- 진행률 업데이트
- 작업 취소 기능
- 결과 캐싱 (6시간)
"""

import asyncio
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import traceback

import sys
import os

# 프로젝트 루트 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
webapp_dir = os.path.dirname(backend_dir)
project_root = os.path.dirname(webapp_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logger_config import get_logger

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """작업 정보"""
    task_id: str
    task_type: str  # "prediction", "screening" 등
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0  # 0-100
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    params: Dict[str, Any] = field(default_factory=dict)
    _cancelled: bool = False

    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self):
        self._cancelled = True
        self.status = TaskStatus.CANCELLED
        self.message = "작업이 취소되었습니다"
        self.completed_at = datetime.now()


class TaskManager:
    """비동기 작업 관리자 - 싱글톤"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.tasks: Dict[str, Task] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)  # 동시 작업 제한
        self.lock = threading.Lock()
        
        # 캐시 설정
        self.cache_duration = timedelta(hours=6)
        self.max_tasks = 100  # 최대 저장 작업 수
        
        # 타임아웃 설정
        self.default_timeout = 120  # 초
        
        self._initialized = True
        logger.info("TaskManager 초기화 완료")
    
    def create_task(self, task_type: str, params: Dict[str, Any]) -> str:
        """새 작업 생성"""
        task_id = str(uuid.uuid4())[:8]  # 짧은 ID
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            params=params,
            message="작업 대기 중..."
        )
        
        with self.lock:
            # 오래된 작업 정리
            self._cleanup_old_tasks()
            self.tasks[task_id] = task
        
        logger.info(f"작업 생성: {task_id} ({task_type})")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """작업 조회"""
        return self.tasks.get(task_id)
    
    def update_task(self, task_id: str, 
                    status: Optional[TaskStatus] = None,
                    progress: Optional[int] = None,
                    message: Optional[str] = None,
                    result: Optional[Dict] = None,
                    error: Optional[str] = None):
        """작업 상태 업데이트"""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        with self.lock:
            if status:
                task.status = status
                if status == TaskStatus.RUNNING:
                    task.started_at = datetime.now()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task.completed_at = datetime.now()
            
            if progress is not None:
                task.progress = min(100, max(0, progress))
            
            if message:
                task.message = message
            
            if result is not None:
                task.result = result
            
            if error:
                task.error = error
    
    def cancel_task(self, task_id: str) -> bool:
        """작업 취소"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        task.cancel()
        logger.info(f"작업 취소: {task_id}")
        return True
    
    async def run_prediction_async(self, task_id: str, 
                                     ticker: str, 
                                     forecast_days: int,
                                     predict_func: Callable,
                                     mode: str = "standard") -> Dict:
        """비동기 예측 실행"""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"작업을 찾을 수 없습니다: {task_id}")
        
        # 상태 업데이트: 실행 중
        self.update_task(task_id, 
                         status=TaskStatus.RUNNING, 
                         progress=10,
                         message=f"{ticker} 데이터 준비 중...")
        
        try:
            # ThreadPoolExecutor에서 CPU 집약적 작업 실행
            loop = asyncio.get_event_loop()
            
            # 예측 함수 래퍼 (취소 체크 포함)
            def run_with_cancellation_check():
                # 진행률 업데이트 콜백
                def progress_callback(progress: int, message: str):
                    if task.is_cancelled():
                        raise InterruptedError("작업이 취소되었습니다")
                    self.update_task(task_id, progress=progress, message=message)
                
                # 시작 알림
                progress_callback(20, f"{ticker} 모델 학습 중...")
                
                # 실제 예측 실행 - predict_func returns dict with 'success', 'data'/'error'
                result_dict = predict_func(ticker, forecast_days, mode)
                
                if task.is_cancelled():
                    raise InterruptedError("작업이 취소되었습니다")
                
                return result_dict
            
            # 타임아웃 적용
            try:
                result_dict = await asyncio.wait_for(
                    loop.run_in_executor(self.executor, run_with_cancellation_check),
                    timeout=self.default_timeout
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"예측 시간 초과 ({self.default_timeout}초)")
            
            # result_dict is {'success': True/False, 'data': ..., 'error': ...}
            if not result_dict.get('success'):
                error = result_dict.get('error', '알 수 없는 오류')
                self.update_task(task_id,
                                 status=TaskStatus.FAILED,
                                 progress=100,
                                 message="예측 실패",
                                 error=error)
                return {"success": False, "error": error}
            
            # 성공
            result_data = result_dict.get('data')
            self.update_task(task_id,
                             status=TaskStatus.COMPLETED,
                             progress=100,
                             message="예측 완료!",
                             result=result_data)
            
            logger.info(f"예측 완료: {task_id} ({ticker})")
            return {"success": True, "data": result_data}
        
        except InterruptedError as e:
            self.update_task(task_id,
                             status=TaskStatus.CANCELLED,
                             message=str(e))
            return {"success": False, "error": str(e)}
        
        except TimeoutError as e:
            self.update_task(task_id,
                             status=TaskStatus.FAILED,
                             progress=100,
                             message="시간 초과",
                             error=str(e))
            return {"success": False, "error": str(e)}
        
        except Exception as e:
            error_msg = f"예측 중 오류: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.update_task(task_id,
                             status=TaskStatus.FAILED,
                             progress=100,
                             message="예측 실패",
                             error=error_msg)
            return {"success": False, "error": error_msg}
    
    def _cleanup_old_tasks(self):
        """오래된 작업 정리"""
        now = datetime.now()
        expired = []
        
        for task_id, task in self.tasks.items():
            # 완료된 작업 중 캐시 기간 지난 것
            if task.completed_at and (now - task.completed_at) > self.cache_duration:
                expired.append(task_id)
            # 생성 후 24시간 지난 미완료 작업
            elif (now - task.created_at) > timedelta(hours=24):
                expired.append(task_id)
        
        for task_id in expired:
            del self.tasks[task_id]
        
        # 최대 개수 초과 시 오래된 것부터 삭제
        if len(self.tasks) > self.max_tasks:
            sorted_tasks = sorted(self.tasks.items(), key=lambda x: x[1].created_at)
            for task_id, _ in sorted_tasks[:len(self.tasks) - self.max_tasks]:
                del self.tasks[task_id]
        
        if expired:
            logger.debug(f"오래된 작업 {len(expired)}개 정리")
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """작업 상태 조회 (API 응답용)"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        # 소요 시간 계산
        elapsed = None
        if task.started_at:
            end_time = task.completed_at or datetime.now()
            elapsed = (end_time - task.started_at).total_seconds()
        
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status.value,
            "progress": task.progress,
            "message": task.message,
            "error": task.error,
            "elapsed_seconds": elapsed,
            "has_result": task.result is not None,
            "created_at": task.created_at.isoformat(),
            "params": task.params
        }
    
    def get_task_result(self, task_id: str) -> Optional[Dict]:
        """작업 결과 조회 (API 응답용)"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        if task.status != TaskStatus.COMPLETED:
            return {
                "success": False,
                "error": f"작업이 아직 완료되지 않았습니다 (상태: {task.status.value})"
            }
        
        return {
            "success": True,
            "data": task.result
        }


# 전역 인스턴스
task_manager = TaskManager()
