import { useEffect, useState } from 'react'
import axios from 'axios'
import './MarketSelector.css'
import { Language, translations } from '../translations'

interface MarketSelectorProps {
    selectedMarket: string
    onMarketChange: (market: string) => void
    language: Language
}

function MarketSelector({ selectedMarket, onMarketChange, language }: MarketSelectorProps) {
    const [markets, setMarkets] = useState<string[]>([])
    const [loading, setLoading] = useState(true)
    const t = translations[language];

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
        korea: `🇰🇷 ${t.marketKorea}`,
        usa: `🇺🇸 ${t.marketUsa}`,
        sweden: `🇸🇪 ${t.marketSweden}`
    }

    if (loading) {
        return <div className="market-selector loading">{language === 'ko' ? '로딩 중...' : 'Loading...'}</div>
    }

    return (
        <div className="market-selector">
            <label>{t.marketSelection}</label>
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
