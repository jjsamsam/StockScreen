import { useState } from 'react'
import './ResultsTable.css'
import ChartView from './ChartView'
import { Language, translations } from '../translations'

interface ResultsTableProps {
    results: {
        buy_signals?: any[]
        sell_signals?: any[]
        total_screened?: number
    }
    language: Language
}

function ResultsTable({ results, language }: ResultsTableProps) {
    const { buy_signals = [], sell_signals = [], total_screened = 0 } = results
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
    const t = translations[language];

    const getConditionLabel = (id: string, fallback: string) => {
        const key = `cond_${id}` as keyof typeof t;
        return (t[key] as string) || fallback;
    };

    const downloadCSV = () => {
        const headers = [
            t.colStock + ' (Ticker)',
            language === 'ko' ? '종목명' : 'Stock Name',
            t.colPrice,
            t.colVolume,
            t.colCondition,
            'Type'
        ]
        const rows = [
            ...buy_signals.map(s => [
                s.symbol,
                `"${s.name.replace(/"/g, '""')}"`,
                s.current_price,
                s.volume,
                `"${(s.matched_ids ? s.matched_ids.map((id: string) => getConditionLabel(id, id)) : s.matched_conditions).join(', ')}"`,
                'BUY'
            ]),
            ...sell_signals.map(s => [
                s.symbol,
                `"${s.name.replace(/"/g, '""')}"`,
                s.current_price,
                s.volume,
                `"${(s.matched_ids ? s.matched_ids.map((id: string) => getConditionLabel(id, id)) : s.matched_conditions).join(', ')}"`,
                'SELL'
            ])
        ]

        const csvContent = [
            headers.join(','),
            ...rows.map(row => row.join(','))
        ].join('\n')

        const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' })
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.setAttribute('href', url)
        link.setAttribute('download', `${language === 'ko' ? '스크리닝_결과' : 'screening_results'}_${new Date().toISOString().slice(0, 10)}.csv`)
        link.style.visibility = 'hidden'
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
    }

    return (
        <div className="results-table">
            <div className="results-header">
                <div className="header-left">
                    <h2>{t.screeningResults}</h2>
                    <div className="stats">
                        <span className="stat">
                            {t.statTotal}: <strong>{total_screened}</strong>
                        </span>
                        <span className="stat buy">
                            {t.statBuy}: <strong>{buy_signals.length}</strong>
                        </span>
                        <span className="stat sell">
                            {t.statSell}: <strong>{sell_signals.length}</strong>
                        </span>
                    </div>
                </div>
                <button
                    className="download-btn"
                    onClick={downloadCSV}
                    disabled={buy_signals.length === 0 && sell_signals.length === 0}
                >
                    {t.downloadCsv}
                </button>
            </div>

            {buy_signals.length > 0 && (
                <div className="signal-section">
                    <h3 className="section-title buy">{t.buySignals}</h3>
                    <div className="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>{t.colStock}</th>
                                    <th>{t.colPrice}</th>
                                    <th>{t.colVolume}</th>
                                    <th>{t.colCondition}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {buy_signals.map((signal, index) => (
                                    <tr
                                        key={index}
                                        onClick={() => setSelectedSymbol(signal.symbol)}
                                        style={{ cursor: 'pointer' }}
                                        title={t.clickChart}
                                    >
                                        <td className="symbol-cell">
                                            <div className="stock-name">{signal.name}</div>
                                            <div className="stock-ticker">{signal.symbol}</div>
                                        </td>
                                        <td className="price">{language === 'ko' ? '' : '$'}{signal.current_price.toLocaleString(undefined, { minimumFractionDigits: language === 'ko' ? 0 : 2 })}{language === 'ko' ? '원' : ''}</td>
                                        <td className="volume">{signal.volume.toLocaleString()}</td>
                                        <td className="conditions">
                                            <div className="condition-wrapper">
                                                <span className="condition-count">{signal.matched_conditions.length}</span>
                                                <div className="condition-list">
                                                    {(signal.matched_ids ? signal.matched_ids : signal.matched_conditions).map((cond: string, i: number) => (
                                                        <span key={i} className="condition-badge">
                                                            {signal.matched_ids ? getConditionLabel(cond, cond) : cond}
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
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
                    <h3 className="section-title sell">{t.sellSignals}</h3>
                    <div className="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>{t.colStock}</th>
                                    <th>{t.colPrice}</th>
                                    <th>{t.colVolume}</th>
                                    <th>{t.colCondition}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {sell_signals.map((signal, index) => (
                                    <tr
                                        key={index}
                                        onClick={() => setSelectedSymbol(signal.symbol)}
                                        style={{ cursor: 'pointer' }}
                                        title={t.clickChart}
                                    >
                                        <td className="symbol-cell">
                                            <div className="stock-name">{signal.name}</div>
                                            <div className="stock-ticker">{signal.symbol}</div>
                                        </td>
                                        <td className="price">{language === 'ko' ? '' : '$'}{signal.current_price.toLocaleString(undefined, { minimumFractionDigits: language === 'ko' ? 0 : 2 })}{language === 'ko' ? '원' : ''}</td>
                                        <td className="volume">{signal.volume.toLocaleString()}</td>
                                        <td className="conditions">
                                            <div className="condition-wrapper">
                                                <span className="condition-count">{signal.matched_conditions.length}</span>
                                                <div className="condition-list">
                                                    {(signal.matched_ids ? signal.matched_ids : signal.matched_conditions).map((cond: string, i: number) => (
                                                        <span key={i} className="condition-badge">
                                                            {signal.matched_ids ? getConditionLabel(cond, cond) : cond}
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
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
                    {t.noResults}
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

export default ResultsTable
