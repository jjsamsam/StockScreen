import { useState, useEffect, useRef } from 'react'
import api from '../api'
import './PredictionPanel.css'
import { Language, translations } from '../translations'

interface PredictionResult {
    ticker: string
    current_price: number
    predicted_price: number
    expected_return: number
    confidence: number
    recommendation: string
    confidence_note: string
    forecast_days: number
}

interface TaskStatus {
    task_id: string
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
    progress: number
    message: string
    error?: string
    elapsed_seconds?: number
}

interface PredictionPanelProps {
    language: Language
    onProcessStart?: () => void
    onProcessEnd?: () => void
}

function PredictionPanel({ language, onProcessStart, onProcessEnd }: PredictionPanelProps) {
    const [ticker, setTicker] = useState('')
    const [forecastDays, setForecastDays] = useState(7)
    const [predictionMode, setPredictionMode] = useState<'fast' | 'standard' | 'precise'>('fast')
    const [loading, setLoading] = useState(false)
    const [searching, setSearching] = useState(false)
    const [searchResults, setSearchResults] = useState<any[]>([])
    const [result, setResult] = useState<PredictionResult | null>(null)
    const [error, setError] = useState('')

    // 비동기 상태
    const [taskId, setTaskId] = useState<string | null>(null)
    const [taskStatus, setTaskStatus] = useState<TaskStatus | null>(null)
    const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)

    const t = translations[language];

    // 컴포넌트 언마운트 시 폴링 정리
    useEffect(() => {
        return () => {
            if (pollingRef.current) {
                clearInterval(pollingRef.current)
            }
        }
    }, [])

    const handleSearch = async (query: string) => {
        setTicker(query)
        if (query.length < 2) {
            setSearchResults([])
            return
        }

        setSearching(true)
        try {
            const response = await api.get('/search', {
                params: { q: query, limit: 5 }
            })
            if (response.data.success) {
                setSearchResults(response.data.results)
            }
        } catch (error) {
            console.error('Search failed:', error)
        } finally {
            setSearching(false)
        }
    }

    const selectStock = (selectedTicker: string) => {
        setTicker(selectedTicker)
        setSearchResults([])
    }

    // 비동기 예측 시작
    const handlePredictAsync = async (targetTicker?: string) => {
        const finalTicker = targetTicker || ticker
        if (!finalTicker.trim()) {
            setError(language === 'ko' ? '종목 코드를 입력해주세요' : 'Please enter a stock ticker')
            return
        }

        setLoading(true)
        if (onProcessStart) onProcessStart()
        setError('')
        setResult(null)
        setSearchResults([])
        setTaskStatus(null)

        try {
            // 비동기 예측 요청
            const response = await api.post('/predict/async', {
                ticker: finalTicker.toUpperCase(),
                forecast_days: forecastDays,
                mode: predictionMode
            })

            if (response.data.success) {
                const newTaskId = response.data.task_id
                setTaskId(newTaskId)

                // 초기 상태 설정
                setTaskStatus({
                    task_id: newTaskId,
                    status: 'pending',
                    progress: 0,
                    message: t.predictionQueued
                })

                // 폴링 시작
                startPolling(newTaskId)
            } else {
                setError(response.data.error || (language === 'ko' ? '예측 시작 실패' : 'Failed to start prediction'))
                setLoading(false)
                if (onProcessEnd) onProcessEnd()
            }
        } catch (err: any) {
            console.error('예측 시작 실패:', err)
            setError(err.response?.data?.detail || (language === 'ko' ? '예측 요청 중 오류가 발생했습니다' : 'An error occurred while starting prediction'))
            setLoading(false)
            if (onProcessEnd) onProcessEnd()
        }
    }

    // 상태 폴링
    const startPolling = (taskIdToCheck: string) => {
        // 기존 폴링 중지
        if (pollingRef.current) {
            clearInterval(pollingRef.current)
        }

        const poll = async () => {
            try {
                const response = await api.get(`/predict/status/${taskIdToCheck}`)
                const status = response.data as TaskStatus

                setTaskStatus(status)

                // 완료 상태 처리
                if (status.status === 'completed') {
                    stopPolling()
                    await fetchResult(taskIdToCheck)
                } else if (status.status === 'failed' || status.status === 'cancelled') {
                    stopPolling()
                    setError(status.error || status.message)
                    setLoading(false)
                    if (onProcessEnd) onProcessEnd()
                }
            } catch (err) {
                console.error('상태 조회 실패:', err)
                // 에러가 5회 이상 발생하면 중지
            }
        }

        // 즉시 한 번 실행
        poll()

        // 1초마다 폴링
        pollingRef.current = setInterval(poll, 1000)
    }

    const stopPolling = () => {
        if (pollingRef.current) {
            clearInterval(pollingRef.current)
            pollingRef.current = null
        }
    }

    // 결과 조회
    const fetchResult = async (taskIdToFetch: string) => {
        try {
            const response = await api.get(`/predict/result/${taskIdToFetch}`)

            if (response.data.success && response.data.data) {
                setResult(response.data.data)
            } else {
                setError(response.data.error || (language === 'ko' ? '결과를 가져올 수 없습니다' : 'Could not fetch result'))
            }
        } catch (err: any) {
            setError(err.response?.data?.detail || (language === 'ko' ? '결과 조회 중 오류가 발생했습니다' : 'Error fetching result'))
        } finally {
            setLoading(false)
            if (onProcessEnd) onProcessEnd()
        }
    }

    // 예측 취소
    const handleCancel = async () => {
        if (!taskId) return

        try {
            await api.post(`/predict/cancel/${taskId}`)
            stopPolling()
            setTaskStatus(prev => prev ? { ...prev, status: 'cancelled', message: t.predictionCancelled } : null)
            setLoading(false)
            if (onProcessEnd) onProcessEnd()
        } catch (err) {
            console.error('취소 실패:', err)
        }
    }

    const getReturnColor = (returnValue: number) => {
        if (returnValue > 0.02) return 'var(--up)'
        if (returnValue < -0.02) return 'var(--down)'
        return 'var(--warning)'
    }

    const downloadCSV = () => {
        if (!result) return

        const headers = ['Ticker', 'Current Price', 'Predicted Price', 'Expected Return', 'Confidence', 'Recommendation', 'Forecast Days']
        const row = [
            result.ticker,
            result.current_price,
            result.predicted_price,
            `${(result.expected_return * 100).toFixed(2)}%`,
            `${(result.confidence * 100).toFixed(1)}%`,
            `"${result.recommendation}"`,
            result.forecast_days
        ]

        const csvContent = [headers.join(','), row.join(',')].join('\n')
        const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' })
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.setAttribute('href', url)
        link.setAttribute('download', `prediction_${result.ticker}_${new Date().toISOString().slice(0, 10)}.csv`)
        link.style.visibility = 'hidden'
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
    }

    const translateRecommendation = (rec: string) => {
        if (!rec) return '';
        if (rec.includes('🚀') || rec.includes('매수')) return t.recommendBuy;
        if (rec.includes('📉') || rec.includes('매도')) return t.recommendSell;
        if (rec.includes('⏸️') || rec.includes('관망')) return t.recommendHold;
        if (rec.includes('⚠️') && rec.includes('매수')) return t.recommendConservativeBuy;
        if (rec.includes('⚠️') && rec.includes('매도')) return t.recommendConservativeSell;
        if (rec.includes('⚠️') && rec.includes('관망')) return t.recommendHold;
        return rec;
    };

    const translateNote = (note: string) => {
        if (!note) return '';
        if (note.includes('높은 신뢰도')) return t.noteHighConfidence;
        if (note.includes('낮은 신뢰도') && note.includes('신중')) return t.noteLowConfidenceCautious;
        if (note.includes('낮은 신뢰도') && note.includes('불확실')) return t.noteLowConfidenceUncertain;
        return note;
    };

    // 진행률 바 색상
    const getProgressColor = (progress: number) => {
        if (progress < 30) return 'var(--primary)'
        if (progress < 70) return 'var(--warning)'
        return 'var(--success)'
    }

    return (
        <div className="prediction-panel">
            <h2>{t.analysisAndPrediction}</h2>

            <div className="input-section">
                <div className="input-group search-container">
                    <label>{language === 'ko' ? '종목 검색' : 'Stock Search'}</label>
                    <div className="search-input-wrapper">
                        <input
                            type="text"
                            value={ticker}
                            onChange={(e) => handleSearch(e.target.value.toUpperCase())}
                            placeholder={t.enterTickerOrName}
                            onKeyPress={(e) => e.key === 'Enter' && handlePredictAsync()}
                            disabled={loading}
                        />
                        {searching && <div className="searching-spinner small"></div>}
                    </div>
                    {searchResults.length > 0 && (
                        <div className="prediction-search-results">
                            {searchResults.map((stock, idx) => (
                                <div
                                    key={idx}
                                    className="search-item"
                                    onClick={() => selectStock(stock.symbol)}
                                >
                                    <span className="item-symbol">{stock.symbol}</span>
                                    <span className="item-name">{stock.name}</span>
                                </div>
                            ))}
                        </div>
                    )}
                    <small>{language === 'ko' ? '미국: AAPL, 한국: Samsung, 스웨덴: Volvo' : 'USA: Apple, KR: Samsung, SE: Volvo'}</small>
                </div>

                <div className="input-group">
                    <label>{t.forecastDays} ({t.days})</label>
                    <select
                        value={forecastDays}
                        onChange={(e) => setForecastDays(Number(e.target.value))}
                        disabled={loading}
                    >
                        <option value={1}>1{t.days} ({language === 'ko' ? '초단기' : 'V.Short'})</option>
                        <option value={3}>3{t.days} ({language === 'ko' ? '단기' : 'Short'})</option>
                        <option value={7}>7{t.days} (1{language === 'ko' ? '주' : 'w'})</option>
                        <option value={14}>14{t.days} (2{language === 'ko' ? '주' : 'w'})</option>
                        <option value={30}>30{t.days} (1{language === 'ko' ? '개월' : 'm'})</option>
                    </select>
                </div>

                {/* 🆕 예측 모드 선택 */}
                <div className="input-group">
                    <label>{language === 'ko' ? '예측 모드' : 'Mode'}</label>
                    <div className="mode-selector">
                        <button
                            className={`mode-btn ${predictionMode === 'fast' ? 'active' : ''}`}
                            onClick={() => setPredictionMode('fast')}
                            disabled={loading}
                            title={language === 'ko' ? 'XGBoost만 사용 (5-15초)' : 'XGBoost only (5-15s)'}
                        >
                            ⚡ {language === 'ko' ? '빠름' : 'Fast'}
                        </button>
                        <button
                            className={`mode-btn ${predictionMode === 'standard' ? 'active' : ''}`}
                            onClick={() => setPredictionMode('standard')}
                            disabled={loading}
                            title={language === 'ko' ? '3개 모델 (15-40초)' : '3 models (15-40s)'}
                        >
                            📊 {language === 'ko' ? '표준' : 'Std'}
                        </button>
                        <button
                            className={`mode-btn ${predictionMode === 'precise' ? 'active' : ''}`}
                            onClick={() => setPredictionMode('precise')}
                            disabled={loading}
                            title={language === 'ko' ? '5개 모델 (40-90초)' : '5 models (40-90s)'}
                        >
                            🎯 {language === 'ko' ? '정밀' : 'Full'}
                        </button>
                    </div>
                </div>

                <button
                    className="predict-btn"
                    onClick={() => handlePredictAsync()}
                    disabled={loading}
                >
                    {loading ? t.analysisInProgress : `🔮 ${t.startAnalysis}`}
                </button>
            </div>

            {/* 진행률 표시 (비동기 예측 중) */}
            {loading && taskStatus && (
                <div className="progress-box">
                    <div className="progress-header">
                        <span className="progress-title">{t.predictionProgress}</span>
                        <button className="cancel-btn" onClick={handleCancel}>
                            ✕ {t.cancelPrediction}
                        </button>
                    </div>

                    <div className="progress-bar-container">
                        <div
                            className="progress-bar-fill"
                            style={{
                                width: `${taskStatus.progress}%`,
                                background: getProgressColor(taskStatus.progress)
                            }}
                        />
                    </div>

                    <div className="progress-info">
                        <span className="progress-message">{taskStatus.message}</span>
                        <span className="progress-percent">{taskStatus.progress}%</span>
                    </div>

                    {taskStatus.elapsed_seconds !== undefined && taskStatus.elapsed_seconds > 0 && (
                        <div className="progress-elapsed">
                            {t.elapsedTime}: {Math.round(taskStatus.elapsed_seconds)}{t.seconds}
                        </div>
                    )}
                </div>
            )}

            {error && (
                <div className="error-box">
                    ❌ {error}
                    {(taskStatus?.status === 'failed' || taskStatus?.status === 'cancelled') && (
                        <button className="retry-btn" onClick={() => handlePredictAsync()}>
                            🔄 {t.retryPrediction}
                        </button>
                    )}
                </div>
            )}

            {result && (
                <div className="result-box">
                    <div className="result-header">
                        <h3>{result.ticker}</h3>
                        <span className="forecast-badge">{result.forecast_days}{t.days} {language === 'ko' ? '예측' : 'Forecast'}</span>
                    </div>

                    <div className="result-grid">
                        <div className="result-item">
                            <span className="label">{language === 'ko' ? '현재가' : 'Price'}</span>
                            <span className="value">${result.current_price.toFixed(2)}</span>
                        </div>

                        <div className="result-item">
                            <span className="label">{t.targetPrice}</span>
                            <span className="value">${result.predicted_price.toFixed(2)}</span>
                        </div>

                        <div className="result-item">
                            <span className="label">{t.expectedReturn}</span>
                            <span
                                className="value large"
                                style={{ color: getReturnColor(result.expected_return) }}
                            >
                                {(result.expected_return * 100).toFixed(2)}%
                            </span>
                        </div>

                        <div className="result-item">
                            <span className="label">{t.aiConfidence}</span>
                            <span className="value">{(result.confidence * 100).toFixed(1)}%</span>
                        </div>
                    </div>

                    <div className="recommendation-box">
                        <div className="recommendation">
                            {translateRecommendation(result.recommendation)}
                        </div>
                        <div className="confidence-note">
                            {translateNote(result.confidence_note)}
                        </div>
                    </div>

                    <div className="confidence-bar">
                        <div
                            className="confidence-fill"
                            style={{
                                width: `${result.confidence * 100}%`,
                                background: result.confidence > 0.6 ? 'var(--success)' : 'var(--warning)'
                            }}
                        />
                    </div>

                    <div className="result-actions">
                        <button className="download-btn-compact" onClick={downloadCSV}>
                            {t.savePredictionCsv}
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}

export default PredictionPanel
