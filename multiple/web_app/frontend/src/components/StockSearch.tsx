import { useState } from 'react'
import api from '../api'
import './StockSearch.css'
import ChartView from './ChartView'
import { Language, translations } from '../translations'

interface StockSearchProps {
    language: Language
}

interface QuoteData {
    symbol: string
    price: number
    change: number
    change_percent: number
    volume: number
}

function StockSearch({ language }: StockSearchProps) {
    const [query, setQuery] = useState('')
    const [results, setResults] = useState<any[]>([])
    const [loading, setLoading] = useState(false)
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
    const [quotes, setQuotes] = useState<{ [symbol: string]: QuoteData }>({})
    const [loadingQuotes, setLoadingQuotes] = useState<{ [symbol: string]: boolean }>({})
    const t = translations[language];

    const handleSearch = async () => {
        if (!query.trim()) {
            return
        }

        setLoading(true)
        setQuotes({}) // 이전 시세 정보 초기화
        try {
            const response = await api.get('/search', {
                params: { q: query, limit: 20 }
            })

            if (response.data.success) {
                setResults(response.data.results)
                // 검색 결과에 대해 시세 정보 비동기 로드
                response.data.results.forEach((stock: any) => {
                    fetchQuote(stock.symbol)
                })
            }
        } catch (error) {
            console.error('검색 실패:', error)
        } finally {
            setLoading(false)
        }
    }

    const fetchQuote = async (symbol: string) => {
        if (quotes[symbol] || loadingQuotes[symbol]) return

        setLoadingQuotes(prev => ({ ...prev, [symbol]: true }))
        try {
            const response = await api.get(`/quote/${symbol}`)
            if (response.data.success) {
                setQuotes(prev => ({
                    ...prev,
                    [symbol]: response.data.data
                }))
            }
        } catch (error) {
            console.error(`시세 조회 실패 (${symbol}):`, error)
        } finally {
            setLoadingQuotes(prev => ({ ...prev, [symbol]: false }))
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

    const formatPrice = (price: number) => {
        if (!price) return '-'
        return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    }

    const formatVolume = (vol: number) => {
        if (!vol) return '-'
        if (vol >= 1e9) return (vol / 1e9).toFixed(1) + 'B'
        if (vol >= 1e6) return (vol / 1e6).toFixed(1) + 'M'
        if (vol >= 1e3) return (vol / 1e3).toFixed(1) + 'K'
        return vol.toFixed(0)
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
                        {results.map((stock, index) => {
                            const quote = quotes[stock.symbol]
                            const isLoadingQuote = loadingQuotes[stock.symbol]

                            return (
                                <div
                                    key={index}
                                    className="stock-card"
                                    onClick={() => setSelectedSymbol(stock.symbol)}
                                >
                                    <div className="card-header">
                                        <div className="stock-symbol">{stock.symbol}</div>
                                        <div className="stock-market">{stock.market}</div>
                                    </div>
                                    <div className="stock-name">{stock.name}</div>

                                    {/* 시세 정보 */}
                                    <div className="stock-quote">
                                        {isLoadingQuote ? (
                                            <div className="quote-loading">
                                                <span className="loading-dot">●</span>
                                            </div>
                                        ) : quote ? (
                                            <>
                                                <div className="quote-price">{formatPrice(quote.price)}</div>
                                                <div className={`quote-change ${quote.change >= 0 ? 'positive' : 'negative'}`}>
                                                    {quote.change >= 0 ? '+' : ''}{formatPrice(quote.change)}
                                                    <span className="quote-percent">
                                                        ({quote.change_percent >= 0 ? '+' : ''}{quote.change_percent.toFixed(2)}%)
                                                    </span>
                                                </div>
                                                <div className="quote-volume">
                                                    {language === 'ko' ? '거래량' : 'Vol'}: {formatVolume(quote.volume)}
                                                </div>
                                            </>
                                        ) : (
                                            <div className="quote-unavailable">-</div>
                                        )}
                                    </div>
                                </div>
                            )
                        })}
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
