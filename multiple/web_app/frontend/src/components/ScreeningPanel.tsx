import { useState } from 'react'
import axios from 'axios'
import './ScreeningPanel.css'

interface ScreeningPanelProps {
    market: string
    onResults: (results: any) => void
}

function ScreeningPanel({ market, onResults }: ScreeningPanelProps) {
    const [buyConditions, setBuyConditions] = useState<string[]>([])
    const [sellConditions, setSellConditions] = useState<string[]>([])
    const [loading, setLoading] = useState(false)
    const [stockLimit, setStockLimit] = useState(100)
    const [matchMode, setMatchMode] = useState<'all' | 'any'>('any')

    const availableConditions = {
        buy: [
            { id: 'golden_cross', label: '골든 크로스' },
            { id: 'rsi_oversold', label: 'RSI 과매도' },
            { id: 'volume_surge', label: '거래량 급증' },
            { id: 'enhanced_ma_buy', label: '강화된 MA 매수' },
            { id: 'enhanced_bb_rsi_buy', label: '강화된 BB+RSI 매수' },
            { id: 'enhanced_macd_volume_buy', label: '강화된 MACD+거래량' },
            { id: 'enhanced_momentum_buy', label: '강화된 모멘텀 매수' }
        ],
        sell: [
            { id: 'death_cross', label: '데드 크로스' },
            { id: 'rsi_overbought', label: 'RSI 과매수' },
            { id: 'enhanced_technical_sell', label: '강화된 기술적 매도' },
            { id: 'enhanced_bb_rsi_sell', label: '강화된 BB+RSI 매도' }
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
            alert('최소 하나의 조건을 선택해주세요')
            return
        }

        setLoading(true)
        try {
            // 먼저 종목 리스트 가져오기
            const stocksResponse = await axios.get(`/api/stocks/${market}`, {
                params: { limit: stockLimit }
            })

            console.log('Stocks response:', stocksResponse.data)

            // ✅ 응답 구조 확인 및 수정
            if (!stocksResponse.data.success || !stocksResponse.data.stocks) {
                alert('종목 데이터를 가져올 수 없습니다')
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
            const screeningResponse = await axios.post('/api/screen', requestData)

            console.log('Screening response:', screeningResponse.data)
            onResults(screeningResponse.data)
        } catch (error: any) {
            console.error('스크리닝 실패:', error)
            console.error('Error response:', error.response?.data)

            // ✅ 에러 메시지 개선
            let errorMessage = '스크리닝 중 오류가 발생했습니다'
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
        }
    }

    return (
        <div className="screening-panel">
            <h2>📊 스크리닝 조건 설정</h2>

            <div className="settings-row">
                <div className="setting-item">
                    <label>종목 수 제한</label>
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
                            전체
                        </button>
                    </div>
                </div>

                <div className="setting-item">
                    <label>매칭 방식</label>
                    <div className="match-mode-selector">
                        <button
                            className={`mode-btn ${matchMode === 'any' ? 'active' : ''}`}
                            onClick={() => setMatchMode('any')}
                            title="선택한 조건 중 하나라도 맞으면 추출"
                        >
                            하나라도 만족 (OR)
                        </button>
                        <button
                            className={`mode-btn ${matchMode === 'all' ? 'active' : ''}`}
                            onClick={() => setMatchMode('all')}
                            title="선택한 모든 조건이 맞아야 추출"
                        >
                            전부 만족 (AND)
                        </button>
                    </div>
                </div>
            </div>

            <div className="conditions-section">
                <h3>🚀 매수 조건</h3>
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
                <h3>📉 매도 조건</h3>
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
                {loading ? '스크리닝 중...' : '🔍 스크리닝 시작'}
            </button>
        </div>
    )
}

export default ScreeningPanel
