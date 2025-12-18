import { useState } from 'react'
import axios from 'axios'
import './StockSearch.css'
import ChartView from './ChartView'
import { Language, translations } from '../translations'

interface StockSearchProps {
    language: Language
}

function StockSearch({ language }: StockSearchProps) {
    const [query, setQuery] = useState('')
    const [results, setResults] = useState<any[]>([])
    const [loading, setLoading] = useState(false)
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
    const t = translations[language];

    const handleSearch = async () => {
        if (!query.trim()) {
            return
        }

        setLoading(true)
        try {
            const response = await axios.get('/api/search', {
                params: { q: query, limit: 20 }
            })

            if (response.data.success) {
                setResults(response.data.results)
            }
        } catch (error) {
            console.error('검색 실패:', error)
        } finally {
            setLoading(false)
        }
    }

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            handleSearch()
        }
    }

    const handleDirectInput = () => {
        if (query.trim()) {
            setSelectedSymbol(query.toUpperCase())
        }
    }

    return (
        <div className="stock-search">
            <h2>🔍 {t.searchAndCharts}</h2>

            <div className="search-box">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder={t.enterTickerOrName}
                />
                <button onClick={handleSearch} disabled={loading}>
                    {loading ? (language === 'ko' ? '검색 중...' : 'Searching...') : t.search}
                </button>
            </div>

            <div className="search-hint">
                {language === 'ko' ? (
                    <>
                        💡 팁: 한국 종목은 영문 이름(Samsung) 또는 티커 코드(005930.KS)로 검색하세요
                        <br />
                        📌 티커 코드를 정확히 알고 있다면 바로 입력 후 "직접 입력" 버튼을 클릭하세요
                    </>
                ) : (
                    <>
                        💡 Tip: Search for Korean stocks using English names (Samsung) or Ticker (005930.KS)
                        <br />
                        📌 If you know the ticker, enter it and click "Direct Input" to view the chart
                    </>
                )}
            </div>

            <div className="direct-input-section">
                <button
                    className="direct-input-btn"
                    onClick={handleDirectInput}
                    disabled={!query.trim()}
                >
                    🎯 "{query}" {language === 'ko' ? '직접 입력하여 차트 보기' : 'Direct Input (View Chart)'}
                </button>
            </div>

            {results.length > 0 && (
                <div className="search-results">
                    <h3>{language === 'ko' ? `검색 결과 (${results.length}개)` : `Search Results (${results.length})`}</h3>
                    <div className="results-grid">
                        {results.map((stock, index) => (
                            <div
                                key={index}
                                className="stock-card"
                                onClick={() => setSelectedSymbol(stock.symbol)}
                            >
                                <div className="stock-symbol">{stock.symbol}</div>
                                <div className="stock-name">{stock.name}</div>
                                <div className="stock-market">{stock.market}</div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {!loading && query && results.length === 0 && (
                <div className="no-results">
                    {language === 'ko' ? '검색 결과가 없습니다. "직접 입력" 버튼을 사용해보세요.' : 'No results found. Try the "Direct Input" button.'}
                </div>
            )}

            {selectedSymbol && (
                <ChartView
                    symbol={selectedSymbol}
                    onClose={() => setSelectedSymbol(null)}
                    language={language}
                />
            )}
        </div>
    )
}

export default StockSearch
