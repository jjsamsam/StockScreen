import { useEffect, useState } from 'react'
import axios from 'axios'
import './MarketSelector.css'

interface MarketSelectorProps {
    selectedMarket: string
    onMarketChange: (market: string) => void
}

function MarketSelector({ selectedMarket, onMarketChange }: MarketSelectorProps) {
    const [markets, setMarkets] = useState<string[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetchMarkets()
    }, [])

    const fetchMarkets = async () => {
        try {
            const response = await axios.get('/api/markets')
            setMarkets(response.data.markets)
        } catch (error) {
            console.error('시장 목록 조회 실패:', error)
            setMarkets(['korea', 'usa', 'sweden']) // 기본값
        } finally {
            setLoading(false)
        }
    }

    const marketNames: Record<string, string> = {
        korea: '🇰🇷 한국',
        usa: '🇺🇸 미국',
        sweden: '🇸🇪 스웨덴'
    }

    if (loading) {
        return <div className="market-selector loading">로딩 중...</div>
    }

    return (
        <div className="market-selector">
            <label>시장 선택</label>
            <div className="market-buttons">
                {markets.map(market => (
                    <button
                        key={market}
                        className={`market-btn ${selectedMarket === market ? 'active' : ''}`}
                        onClick={() => onMarketChange(market)}
                    >
                        {marketNames[market] || market}
                    </button>
                ))}
            </div>
        </div>
    )
}

export default MarketSelector
