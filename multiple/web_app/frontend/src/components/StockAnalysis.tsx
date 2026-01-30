import { useState, useEffect } from 'react'
import api from '../api'
import './StockAnalysis.css'

interface StockAnalysisProps {
    ticker: string
    language: 'ko' | 'en'
}

interface AnalysisData {
    symbol: string
    last_update: string
    price: {
        current_price: number
        prev_close: number
        change: number
        change_percent: number
        high: number
        low: number
    }
    rsi: {
        value: number
        signal: string
        description: string
    }
    macd: {
        macd: number
        signal_line: number
        histogram: number
        signal: string
        description: string
    }
    bollinger: {
        upper: number
        middle: number
        lower: number
        position: number
        signal: string
        description: string
    }
    moving_averages: {
        ma20: number
        ma60: number
        ma120: number
        signal: string
        description: string
        trend_strength: string
    }
    volume: {
        current: number
        average_20d: number
        ratio: number
        signal: string
        description: string
    }
    trend: {
        adx: number
        plus_di: number
        minus_di: number
        atr: number
        signal: string
        description: string
        direction: string
    }
    summary: {
        bullish_points: number
        bearish_points: number
        signal: string
        description: string
    }
    risk_management: {
        stop_loss: number
        take_profit: number
        risk_reward_ratio: number
    }
}

function StockAnalysis({ ticker, language }: StockAnalysisProps) {
    const [loading, setLoading] = useState(false)
    const [data, setData] = useState<AnalysisData | null>(null)
    const [error, setError] = useState('')
    const [period, setPeriod] = useState('6mo')

    const t = {
        title: language === 'ko' ? '📊 기술적 분석' : '📊 Technical Analysis',
        loading: language === 'ko' ? '분석 중...' : 'Analyzing...',
        noData: language === 'ko' ? '종목을 선택하세요' : 'Select a stock',
        price: language === 'ko' ? '💰 가격 정보' : '💰 Price Info',
        indicators: language === 'ko' ? '📈 기술적 지표' : '📈 Technical Indicators',
        trend: language === 'ko' ? '📊 추세 분석' : '📊 Trend Analysis',
        summary: language === 'ko' ? '💡 종합 의견' : '💡 Summary',
        risk: language === 'ko' ? '⚠️ 리스크 관리' : '⚠️ Risk Management',
        period: language === 'ko' ? '분석 기간' : 'Period',
        lastUpdate: language === 'ko' ? '최종 업데이트' : 'Last Update',
        currentPrice: language === 'ko' ? '현재가' : 'Current',
        change: language === 'ko' ? '전일대비' : 'Change',
        high: language === 'ko' ? '고가' : 'High',
        low: language === 'ko' ? '저가' : 'Low',
        stopLoss: language === 'ko' ? '손절가' : 'Stop Loss',
        takeProfit: language === 'ko' ? '목표가' : 'Take Profit',
        riskReward: language === 'ko' ? '손익비' : 'Risk/Reward',
        bullishPoints: language === 'ko' ? '매수 신호' : 'Bullish',
        bearishPoints: language === 'ko' ? '매도 신호' : 'Bearish',
        volume: language === 'ko' ? '거래량' : 'Volume',
        volumeRatio: language === 'ko' ? '평균 대비' : 'vs Avg',
        bollingerBand: language === 'ko' ? '볼린저밴드' : 'Bollinger Band',
        movingAverage: language === 'ko' ? '이동평균선' : 'Moving Avg',
        bullishDominant: language === 'ko' ? '📈 상승 우세' : '📈 Bullish',
        bearishDominant: language === 'ko' ? '📉 하락 우세' : '📉 Bearish',
    }

    // 백엔드에서 오는 한글 description을 영어로 변환
    const translateDescription = (desc: string): string => {
        if (language === 'ko') return desc;
        const translations: { [key: string]: string } = {
            // RSI
            '극도 과매수 (즉시 매도 고려)': 'Extreme Overbought (Consider Selling)',
            '과매수 (매도 준비)': 'Overbought (Prepare to Sell)',
            '강세 구간 (상승 지속 가능)': 'Bullish Zone (Uptrend Likely)',
            '중립 구간 (방향성 애매)': 'Neutral Zone (No Clear Direction)',
            '약세 구간 (하락 지속 가능)': 'Bearish Zone (Downtrend Likely)',
            '과매도 (매수 준비)': 'Oversold (Prepare to Buy)',
            '극도 과매도 (적극 매수 고려)': 'Extreme Oversold (Consider Buying)',
            // MACD
            '골든크로스 발생 (강력한 매수 신호)': 'Golden Cross (Strong Buy Signal)',
            '데드크로스 발생 (강력한 매도 신호)': 'Death Cross (Strong Sell Signal)',
            'MACD > Signal (상승 모멘텀)': 'MACD > Signal (Bullish Momentum)',
            'MACD < Signal (하락 모멘텀)': 'MACD < Signal (Bearish Momentum)',
            // Bollinger
            '상단 근접 (매도 관심)': 'Near Upper Band (Sell Interest)',
            '하단 근접 (매수 관심)': 'Near Lower Band (Buy Interest)',
            '중앙 영역 (관망)': 'Middle Zone (Wait & See)',
            // MA
            '완전 정배열 (강한 상승 추세)': 'Perfect Alignment (Strong Uptrend)',
            '부분 정배열 (단기 상승 추세)': 'Partial Alignment (Short-term Uptrend)',
            '완전 역배열 (강한 하락 추세)': 'Reverse Alignment (Strong Downtrend)',
            '부분 역배열 (단기 하락 추세)': 'Partial Reverse (Short-term Downtrend)',
            '혼재 (방향성 불분명)': 'Mixed (No Clear Direction)',
            // Volume
            '대량 거래 (주목 필요)': 'Heavy Volume (Attention Needed)',
            '높은 거래량 (관심 증가)': 'High Volume (Rising Interest)',
            '보통 이상 거래량': 'Above Average Volume',
            '보통 거래량': 'Normal Volume',
            '낮은 거래량 (관심 부족)': 'Low Volume (Low Interest)',
            // Trend
            '강한 추세': 'Strong Trend',
            '약한 추세 (횡보)': 'Weak Trend (Sideways)',
            // Summary
            '강력 매수 추천': 'Strong Buy',
            '매수 관심 구간': 'Buy Interest Zone',
            '강력 매도 추천': 'Strong Sell',
            '매도 관심 구간': 'Sell Interest Zone',
            '중립/관망 구간': 'Neutral/Wait Zone',
        };
        return translations[desc] || desc;
    }

    useEffect(() => {
        if (ticker) {
            fetchAnalysis()
        }
    }, [ticker, period])

    const fetchAnalysis = async () => {
        if (!ticker) return

        setLoading(true)
        setError('')

        try {
            const response = await api.get(`/analysis/${ticker}`, {
                params: { period }
            })

            if (response.data.success) {
                setData(response.data.data)
            } else {
                setError(response.data.error || 'Analysis failed')
            }
        } catch (err: any) {
            setError(err.response?.data?.detail || err.message || 'Error fetching analysis')
        } finally {
            setLoading(false)
        }
    }

    const getSignalEmoji = (signal: string) => {
        const emojiMap: { [key: string]: string } = {
            strong_buy: '🟢',
            buy: '🟢',
            bullish: '🟢',
            strong_bullish: '🟢',
            golden_cross: '🟢',
            oversold: '🟢',
            extreme_oversold: '🔵',
            neutral: '⚪',
            middle: '⚪',
            normal: '⚪',
            sell: '🔴',
            strong_sell: '🔴',
            bearish: '🔴',
            strong_bearish: '🔴',
            death_cross: '🔴',
            overbought: '🟠',
            extreme_overbought: '🔴',
            upper: '🔴',
            lower: '🟢',
            high: '📈',
            extreme_high: '🔥',
            above_average: '📊',
            low: '📉',
        }
        return emojiMap[signal] || '⚪'
    }

    const formatNumber = (num: number, decimals = 2) => {
        if (num === undefined || num === null || isNaN(num)) return '-'
        return num.toLocaleString(undefined, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        })
    }

    const formatVolume = (num: number) => {
        if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B'
        if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M'
        if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K'
        return num.toFixed(0)
    }

    if (!ticker) {
        return (
            <div className="stock-analysis empty">
                <p>{t.noData}</p>
            </div>
        )
    }

    if (loading) {
        return (
            <div className="stock-analysis loading">
                <div className="analysis-spinner"></div>
                <p>{t.loading}</p>
            </div>
        )
    }

    if (error) {
        return (
            <div className="stock-analysis error">
                <p>❌ {error}</p>
                <button onClick={fetchAnalysis}>🔄 Retry</button>
            </div>
        )
    }

    if (!data) {
        return null
    }

    return (
        <div className="stock-analysis">
            {/* 헤더 */}
            <div className="analysis-header">
                <h3>{t.title} - {data.symbol}</h3>
                <div className="period-selector">
                    <label>{t.period}:</label>
                    <select value={period} onChange={(e) => setPeriod(e.target.value)}>
                        <option value="1mo">1M</option>
                        <option value="3mo">3M</option>
                        <option value="6mo">6M</option>
                        <option value="1y">1Y</option>
                        <option value="2y">2Y</option>
                    </select>
                </div>
            </div>
            <small className="last-update">{t.lastUpdate}: {data.last_update}</small>

            {/* 가격 정보 */}
            <section className="analysis-section price-section">
                <h4>{t.price}</h4>
                <div className="price-grid">
                    <div className="price-main">
                        <span className="label">{t.currentPrice}</span>
                        <span className="value">{formatNumber(data.price.current_price)}</span>
                    </div>
                    <div className={`price-change ${data.price.change >= 0 ? 'positive' : 'negative'}`}>
                        <span className="label">{t.change}</span>
                        <span className="value">
                            {data.price.change >= 0 ? '+' : ''}{formatNumber(data.price.change)}
                            ({data.price.change_percent >= 0 ? '+' : ''}{formatNumber(data.price.change_percent)}%)
                        </span>
                    </div>
                    <div className="price-range">
                        <span>{t.high}: {formatNumber(data.price.high)}</span>
                        <span>{t.low}: {formatNumber(data.price.low)}</span>
                    </div>
                </div>
            </section>

            {/* 종합 의견 */}
            <section className="analysis-section summary-section">
                <h4>{t.summary}</h4>
                <div className={`summary-signal ${data.summary.signal}`}>
                    {getSignalEmoji(data.summary.signal)} {translateDescription(data.summary.description)}
                </div>
                <div className="summary-points">
                    <span className="bullish">{t.bullishPoints}: {data.summary.bullish_points}</span>
                    <span className="bearish">{t.bearishPoints}: {data.summary.bearish_points}</span>
                </div>
            </section>

            {/* 기술적 지표 */}
            <section className="analysis-section indicators-section">
                <h4>{t.indicators}</h4>

                <div className="indicator-row">
                    <span className="indicator-name">RSI ({formatNumber(data.rsi.value, 1)})</span>
                    <span className="indicator-signal">
                        {getSignalEmoji(data.rsi.signal)} {translateDescription(data.rsi.description)}
                    </span>
                </div>

                <div className="indicator-row">
                    <span className="indicator-name">MACD</span>
                    <span className="indicator-signal">
                        {getSignalEmoji(data.macd.signal)} {translateDescription(data.macd.description)}
                    </span>
                </div>

                <div className="indicator-row">
                    <span className="indicator-name">{t.bollingerBand} ({(data.bollinger.position * 100).toFixed(0)}%)</span>
                    <span className="indicator-signal">
                        {getSignalEmoji(data.bollinger.signal)} {translateDescription(data.bollinger.description)}
                    </span>
                </div>

                <div className="indicator-row">
                    <span className="indicator-name">{t.movingAverage}</span>
                    <span className="indicator-signal">
                        {getSignalEmoji(data.moving_averages.signal)} {translateDescription(data.moving_averages.description)}
                    </span>
                </div>
            </section>

            {/* 추세 분석 */}
            <section className="analysis-section trend-section">
                <h4>{t.trend}</h4>
                <div className="trend-grid">
                    <div className="trend-item">
                        <span>ADX: {formatNumber(data.trend.adx, 1)}</span>
                        <span>{getSignalEmoji(data.trend.signal)} {translateDescription(data.trend.description)}</span>
                    </div>
                    <div className="trend-item">
                        <span>+DI: {formatNumber(data.trend.plus_di, 1)}</span>
                        <span>-DI: {formatNumber(data.trend.minus_di, 1)}</span>
                    </div>
                    <div className="trend-direction">
                        {data.trend.direction === 'bullish' ? t.bullishDominant : t.bearishDominant}
                    </div>
                </div>
            </section>

            {/* 거래량 */}
            <section className="analysis-section volume-section">
                <h4>{t.volume}</h4>
                <div className="volume-info">
                    <span>{formatVolume(data.volume.current)}</span>
                    <span>{t.volumeRatio}: {formatNumber(data.volume.ratio, 2)}x</span>
                    <span>{getSignalEmoji(data.volume.signal)} {translateDescription(data.volume.description)}</span>
                </div>
            </section>

            {/* 리스크 관리 */}
            <section className="analysis-section risk-section">
                <h4>{t.risk}</h4>
                <div className="risk-grid">
                    <div className="risk-item">
                        <span className="label">{t.stopLoss}</span>
                        <span className="value stop-loss">{formatNumber(data.risk_management.stop_loss)}</span>
                    </div>
                    <div className="risk-item">
                        <span className="label">{t.takeProfit}</span>
                        <span className="value take-profit">{formatNumber(data.risk_management.take_profit)}</span>
                    </div>
                    <div className="risk-item">
                        <span className="label">{t.riskReward}</span>
                        <span className="value">1:{formatNumber(data.risk_management.risk_reward_ratio, 1)}</span>
                    </div>
                </div>
            </section>
        </div>
    )
}

export default StockAnalysis
