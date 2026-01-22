import { useState } from 'react'
import './App.css'
import Header from './components/Header'
import MarketSelector from './components/MarketSelector'
import ScreeningPanel from './components/ScreeningPanel'
import PredictionPanel from './components/PredictionPanel'
import StockSearch from './components/StockSearch'
import ResultsTable from './components/ResultsTable'
import { Language, translations } from './translations'

function App() {
    const [activeTab, setActiveTab] = useState<'screening' | 'prediction' | 'chart'>('screening')
    const [selectedMarket, setSelectedMarket] = useState<string>('korea')
    const [screeningResults, setScreeningResults] = useState<any>(null)
    const [language, setLanguage] = useState<Language>('ko')
    const [isProcessing, setIsProcessing] = useState<boolean>(false)

    const t = translations[language];

    return (
        <div className="app">
            <Header language={language} setLanguage={setLanguage} isProcessing={isProcessing} />

            <div className="container">
                <div className="tabs">
                    <button
                        className={`tab ${activeTab === 'screening' ? 'active' : ''}`}
                        onClick={() => setActiveTab('screening')}
                    >
                        {t.tabScreening}
                    </button>
                    <button
                        className={`tab ${activeTab === 'prediction' ? 'active' : ''}`}
                        onClick={() => setActiveTab('prediction')}
                    >
                        {t.tabPrediction}
                    </button>
                    <button
                        className={`tab ${activeTab === 'chart' ? 'active' : ''}`}
                        onClick={() => setActiveTab('chart')}
                    >
                        {t.tabChart}
                    </button>
                </div>

                <div className="content">
                    {activeTab === 'screening' ? (
                        <>
                            <MarketSelector
                                selectedMarket={selectedMarket}
                                onMarketChange={setSelectedMarket}
                                language={language}
                            />
                            <ScreeningPanel
                                market={selectedMarket}
                                onResults={setScreeningResults}
                                language={language}
                                onProcessStart={() => setIsProcessing(true)}
                                onProcessEnd={() => setIsProcessing(false)}
                            />
                            {screeningResults && (
                                <ResultsTable results={screeningResults} language={language} />
                            )}
                        </>
                    ) : activeTab === 'prediction' ? (
                        <PredictionPanel
                            language={language}
                            onProcessStart={() => setIsProcessing(true)}
                            onProcessEnd={() => setIsProcessing(false)}
                        />
                    ) : (
                        <StockSearch language={language} />
                    )}
                </div>
            </div>
        </div>
    )
}

export default App
