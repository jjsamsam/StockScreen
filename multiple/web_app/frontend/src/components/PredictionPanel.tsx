import { useState } from 'react'
import axios from 'axios'
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

interface PredictionPanelProps {
    language: Language
}

function PredictionPanel({ language }: PredictionPanelProps) {
    const [ticker, setTicker] = useState('')
    const [forecastDays, setForecastDays] = useState(7)
    const [loading, setLoading] = useState(false)
    const [searching, setSearching] = useState(false)
    const [searchResults, setSearchResults] = useState<any[]>([])
    const [result, setResult] = useState<PredictionResult | null>(null)
    const [error, setError] = useState('')
    const t = translations[language];

    const handleSearch = async (query: string) => {
        setTicker(query)
        if (query.length < 2) {
            setSearchResults([])
            return
        }

        setSearching(true)
        try {
            const response = await axios.get('/api/search', {
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

    const handlePredict = async (targetTicker?: string) => {
        const finalTicker = targetTicker || ticker
        if (!finalTicker.trim()) {
            setError(language === 'ko' ? '종목 코드를 입력해주세요' : 'Please enter a stock ticker')
            return
        }

        setLoading(true)
        setError('')
        setResult(null)
        setSearchResults([])

        try {
            const response = await axios.post('/api/predict', {
                ticker: finalTicker.toUpperCase(),
                forecast_days: forecastDays
            })

            setResult(response.data.data)
        } catch (err: any) {
            console.error('예측 실패:', err)
            setError(err.response?.data?.detail || (language === 'ko' ? '예측 중 오류가 발생했습니다' : 'An error occurred during prediction'))
        } finally {
            setLoading(false)
        }
    }

    const getReturnColor = (returnValue: number) => {
        if (returnValue > 0.02) return 'var(--success)'
        if (returnValue < -0.02) return 'var(--danger)'
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
                            onKeyPress={(e) => e.key === 'Enter' && handlePredict()}
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
                    <small>{language === 'ko' ? '미국: AAPL, 한국: 005930, 스웨덴: VOLV-B.ST' : 'USA: AAPL, KR: 005930, SE: VOLV-B.ST'}</small>
                </div>

                <div className="input-group">
                    <label>{t.forecastDays} ({t.days})</label>
                    <select
                        value={forecastDays}
                        onChange={(e) => setForecastDays(Number(e.target.value))}
                    >
                        <option value={1}>1{t.days} ({language === 'ko' ? '초단기' : 'V.Short'})</option>
                        <option value={3}>3{t.days} ({language === 'ko' ? '단기' : 'Short'})</option>
                        <option value={7}>7{t.days} (1{language === 'ko' ? '주' : 'w'})</option>
                        <option value={14}>14{t.days} (2{language === 'ko' ? '주' : 'w'})</option>
                        <option value={30}>30{t.days} (1{language === 'ko' ? '개월' : 'm'})</option>
                    </select>
                </div>

                <button
                    className="predict-btn"
                    onClick={() => handlePredict()}
                    disabled={loading}
                >
                    {loading ? t.analysisInProgress : `🔮 ${t.startAnalysis}`}
                </button>
            </div>

            {error && (
                <div className="error-box">
                    ❌ {error}
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
