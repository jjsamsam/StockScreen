import { useState } from 'react'
import api from '../api'
import './ScreeningPanel.css'
import { Language, translations } from '../translations'

interface ScreeningPanelProps {
    market: string
    onResults: (results: any) => void
    language: Language
    onProcessStart?: () => void
    onProcessEnd?: () => void
}

function ScreeningPanel({ market, onResults, language, onProcessStart, onProcessEnd }: ScreeningPanelProps) {
    const [buyConditions, setBuyConditions] = useState<string[]>([])
    const [sellConditions, setSellConditions] = useState<string[]>([])
    const [loading, setLoading] = useState(false)
    const [stockLimit, setStockLimit] = useState(100)
    const [matchMode, setMatchMode] = useState<'all' | 'any'>('any')
    const [universeSource, setUniverseSource] = useState<'csv' | 'dynamic' | 'hybrid'>('csv')
    const t = translations[language];

    const availableConditions = {
        buy: [
            { id: 'golden_cross', label: t.cond_golden_cross },
            { id: 'rsi_oversold', label: t.cond_rsi_oversold },
            { id: 'volume_surge', label: t.cond_volume_surge },
            { id: 'enhanced_ma_buy', label: t.cond_enhanced_ma_buy },
            { id: 'enhanced_bb_rsi_buy', label: t.cond_enhanced_bb_rsi_buy },
            { id: 'enhanced_macd_volume_buy', label: t.cond_enhanced_macd_volume_buy },
            { id: 'enhanced_momentum_buy', label: t.cond_enhanced_momentum_buy },
            { id: 'balanced_buy', label: t.cond_balanced_buy },
            { id: 'ichimoku_bullish', label: t.cond_ichimoku_bullish }
        ],
        sell: [
            { id: 'death_cross', label: t.cond_death_cross },
            { id: 'rsi_overbought', label: t.cond_rsi_overbought },
            { id: 'enhanced_technical_sell', label: t.cond_enhanced_technical_sell },
            { id: 'enhanced_bb_rsi_sell', label: t.cond_enhanced_bb_rsi_sell },
            { id: 'balanced_sell', label: t.cond_balanced_sell },
            { id: 'ichimoku_bearish', label: t.cond_ichimoku_bearish }
        ]
    }

    const toggleCondition = (type: 'buy' | 'sell', condition: string) => {
        if (type === 'buy') {
            setBuyConditions(prev =>
                prev.includes(condition)
                    ? prev.filter(c => c !== condition)
                    : [...prev, condition]
            )
        } else {
            setSellConditions(prev =>
                prev.includes(condition)
                    ? prev.filter(c => c !== condition)
                    : [...prev, condition]
            )
        }
    }

    const handleScreen = async () => {
        if (buyConditions.length === 0 && sellConditions.length === 0) {
            alert(language === 'ko' ? '최소 하나의 조건을 선택해주세요' : 'Please select at least one condition')
            return
        }

        setLoading(true)
        if (onProcessStart) onProcessStart()
        try {
            // 먼저 종목 리스트 가져오기
            const stocksResponse = await api.get(`/stocks/${market}`, {
                params: { limit: stockLimit, source: universeSource }
            })

            console.log('Stocks response:', stocksResponse.data)

            // ✅ 응답 구조 확인 및 수정
            if (!stocksResponse.data.success || !stocksResponse.data.stocks) {
                alert(language === 'ko' ? '종목 데이터를 가져올 수 없습니다' : 'Could not fetch stock data')
                return
            }

            const symbols = stocksResponse.data.stocks.map((s: any) => s.ticker)

            console.log('Symbols:', symbols)
            console.log('Buy conditions:', buyConditions)
            console.log('Sell conditions:', sellConditions)

            // ✅ null 대신 undefined 사용 (FastAPI가 선택적 필드로 인식)
            const requestData: any = {
                symbols,
                period: '1y',
                match_mode: matchMode
            }

            if (buyConditions.length > 0) {
                requestData.buy_conditions = buyConditions
            }

            if (sellConditions.length > 0) {
                requestData.sell_conditions = sellConditions
            }

            console.log('Screening request:', requestData)

            // 스크리닝 실행
            const screeningResponse = await api.post('/screen', requestData)

            console.log('Screening response:', screeningResponse.data)
            onResults(screeningResponse.data)
        } catch (error: any) {
            console.error('스크리닝 실패:', error)
            console.error('Error response:', error.response?.data)

            // ✅ 에러 메시지 개선
            let errorMessage = language === 'ko' ? '스크리닝 중 오류가 발생했습니다' : 'An error occurred during screening'
            if (error.response?.data?.detail) {
                if (Array.isArray(error.response.data.detail)) {
                    errorMessage = error.response.data.detail.map((e: any) =>
                        `${e.loc?.join('.')}: ${e.msg}`
                    ).join('\n')
                } else {
                    errorMessage = error.response.data.detail
                }
            }
            alert(errorMessage)
        } finally {
            setLoading(false)
            if (onProcessEnd) onProcessEnd()
        }
    }

    return (
        <div className="screening-panel">
            <h2>{t.screeningSettings}</h2>

            <div className="settings-row">
                <div className="setting-item">
                    <label>{t.stockLimit}</label>
                    <div className="limit-input-group">
                        <input
                            type="number"
                            value={stockLimit}
                            onChange={(e) => setStockLimit(Number(e.target.value))}
                            min={10}
                            max={10000}
                            step={10}
                        />
                        <button
                            className={`limit-all-btn ${stockLimit >= 10000 ? 'active' : ''}`}
                            onClick={() => setStockLimit(10000)}
                        >
                            {t.limitAll}
                        </button>
                    </div>
                </div>

                <div className="setting-item">
                    <label>{t.matchMode}</label>
                    <div className="match-mode-selector">
                        <button
                            className={`mode-btn ${matchMode === 'any' ? 'active' : ''}`}
                            onClick={() => setMatchMode('any')}
                            title={language === 'ko' ? '선택한 조건 중 하나라도 맞으면 추출' : 'Extract if any of the selected conditions match'}
                        >
                            {t.matchAny}
                        </button>
                        <button
                            className={`mode-btn ${matchMode === 'all' ? 'active' : ''}`}
                            onClick={() => setMatchMode('all')}
                            title={language === 'ko' ? '선택한 모든 조건이 맞아야 추출' : 'Extract only if all selected conditions match'}
                        >
                            {t.matchAll}
                        </button>
                    </div>
                </div>

                <div className="setting-item">
                    <label>{t.universeSource}</label>
                    <div className="match-mode-selector">
                        <button
                            className={`mode-btn ${universeSource === 'csv' ? 'active' : ''}`}
                            onClick={() => setUniverseSource('csv')}
                            title={language === 'ko' ? '로컬 마스터 CSV 사용' : 'Use local master CSV'}
                        >
                            {t.sourceCsv}
                        </button>
                        <button
                            className={`mode-btn ${universeSource === 'dynamic' ? 'active' : ''}`}
                            onClick={() => setUniverseSource('dynamic')}
                            title={language === 'ko' ? '대표 종목을 실시간 거래대금으로 정렬' : 'Rank a live representative universe by turnover'}
                        >
                            {t.sourceDynamic}
                        </button>
                        <button
                            className={`mode-btn ${universeSource === 'hybrid' ? 'active' : ''}`}
                            onClick={() => setUniverseSource('hybrid')}
                            title={language === 'ko' ? '동적 소스 실패 시 CSV로 전환' : 'Use dynamic source with CSV fallback'}
                        >
                            {t.sourceHybrid}
                        </button>
                    </div>
                </div>
            </div>

            <div className="conditions-section">
                <h3>{t.buyConditions}</h3>
                <div className="condition-grid">
                    {availableConditions.buy.map(cond => (
                        <button
                            key={cond.id}
                            className={`condition-btn ${buyConditions.includes(cond.id) ? 'active' : ''}`}
                            onClick={() => toggleCondition('buy', cond.id)}
                        >
                            {cond.label}
                        </button>
                    ))}
                </div>
            </div>

            <div className="conditions-section">
                <h3>{t.sellConditions}</h3>
                <div className="condition-grid">
                    {availableConditions.sell.map(cond => (
                        <button
                            key={cond.id}
                            className={`condition-btn ${sellConditions.includes(cond.id) ? 'active' : ''}`}
                            onClick={() => toggleCondition('sell', cond.id)}
                        >
                            {cond.label}
                        </button>
                    ))}
                </div>
            </div>

            <button
                className="screen-btn"
                onClick={handleScreen}
                disabled={loading}
            >
                {loading ? t.screeningInProgress : `🔍 ${t.startScreening}`}
            </button>
        </div>
    )
}

export default ScreeningPanel
