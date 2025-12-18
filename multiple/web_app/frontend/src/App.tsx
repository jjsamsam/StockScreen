import { useState } from 'react'
import './App.css'
import Header from './components/Header'
import MarketSelector from './components/MarketSelector'
import ScreeningPanel from './components/ScreeningPanel'
import PredictionPanel from './components/PredictionPanel'
import StockSearch from './components/StockSearch'
import ResultsTable from './components/ResultsTable'

function App() {
    const [activeTab, setActiveTab] = useState<'screening' | 'prediction' | 'chart'>('screening')
    const [selectedMarket, setSelectedMarket] = useState<string>('korea')
    const [screeningResults, setScreeningResults] = useState<any>(null)

    return (
        <div className="app">
            <Header />

            <div className="container">
                <div className="tabs">
                    <button
                        className={`tab ${activeTab === 'screening' ? 'active' : ''}`}
                        onClick={() => setActiveTab('screening')}
                    >
                        📊 스크리닝
                    </button>
                    <button
                        className={`tab ${activeTab === 'prediction' ? 'active' : ''}`}
                        onClick={() => setActiveTab('prediction')}
                    >
                        🤖 AI 예측
                    </button>
                    <button
                        className={`tab ${activeTab === 'chart' ? 'active' : ''}`}
                        onClick={() => setActiveTab('chart')}
                    >
                        📈 차트 보기
                    </button>
                </div>

                <div className="content">
                    {activeTab === 'screening' ? (
                        <>
                            <MarketSelector
                                selectedMarket={selectedMarket}
                                onMarketChange={setSelectedMarket}
                            />
                            <ScreeningPanel
                                market={selectedMarket}
                                onResults={setScreeningResults}
                            />
                            {screeningResults && (
                                <ResultsTable results={screeningResults} />
                            )}
                        </>
                    ) : activeTab === 'prediction' ? (
                        <PredictionPanel />
                    ) : (
                        <StockSearch />
                    )}
                </div>
            </div>
        </div>
    )
}

export default App
