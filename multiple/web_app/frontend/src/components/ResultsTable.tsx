import { useState } from 'react'
import './ResultsTable.css'
import ChartView from './ChartView'

interface ResultsTableProps {
    results: {
        buy_signals?: any[]
        sell_signals?: any[]
        total_screened?: number
    }
}

function ResultsTable({ results }: ResultsTableProps) {
    const { buy_signals = [], sell_signals = [], total_screened = 0 } = results
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)

    return (
        <div className="results-table">
            <div className="results-header">
                <h2>📊 스크리닝 결과</h2>
                <div className="stats">
                    <span className="stat">
                        전체: <strong>{total_screened}</strong>
                    </span>
                    <span className="stat buy">
                        매수: <strong>{buy_signals.length}</strong>
                    </span>
                    <span className="stat sell">
                        매도: <strong>{sell_signals.length}</strong>
                    </span>
                </div>
            </div>

            {buy_signals.length > 0 && (
                <div className="signal-section">
                    <h3 className="section-title buy">🚀 매수 신호</h3>
                    <div className="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>종목</th>
                                    <th>현재가</th>
                                    <th>거래량</th>
                                    <th>조건</th>
                                </tr>
                            </thead>
                            <tbody>
                                {buy_signals.map((signal, index) => (
                                    <tr
                                        key={index}
                                        onClick={() => setSelectedSymbol(signal.symbol)}
                                        style={{ cursor: 'pointer' }}
                                        title="클릭하여 차트 보기"
                                    >
                                        <td className="symbol">{signal.symbol}</td>
                                        <td className="price">${signal.current_price.toFixed(2)}</td>
                                        <td className="volume">{signal.volume.toLocaleString()}</td>
                                        <td className="conditions">
                                            {signal.matched_conditions.join(', ')}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {sell_signals.length > 0 && (
                <div className="signal-section">
                    <h3 className="section-title sell">📉 매도 신호</h3>
                    <div className="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>종목</th>
                                    <th>현재가</th>
                                    <th>거래량</th>
                                    <th>조건</th>
                                </tr>
                            </thead>
                            <tbody>
                                {sell_signals.map((signal, index) => (
                                    <tr
                                        key={index}
                                        onClick={() => setSelectedSymbol(signal.symbol)}
                                        style={{ cursor: 'pointer' }}
                                        title="클릭하여 차트 보기"
                                    >
                                        <td className="symbol">{signal.symbol}</td>
                                        <td className="price">${signal.current_price.toFixed(2)}</td>
                                        <td className="volume">{signal.volume.toLocaleString()}</td>
                                        <td className="conditions">
                                            {signal.matched_conditions.join(', ')}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {buy_signals.length === 0 && sell_signals.length === 0 && (
                <div className="no-results">
                    조건에 맞는 종목이 없습니다
                </div>
            )}

            {selectedSymbol && (
                <ChartView
                    symbol={selectedSymbol}
                    onClose={() => setSelectedSymbol(null)}
                />
            )}
        </div>
    )
}

export default ResultsTable
