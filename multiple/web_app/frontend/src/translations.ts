export type Language = 'ko' | 'en';

export const translations = {
    ko: {
        // App
        tabScreening: '📊 스크리닝',
        tabPrediction: '🤖 AI 예측',
        tabChart: '📈 차트 보기',

        headerSubtitle: 'AI 기반 주식 스크리닝 & 예측 시스템',

        // Backend Status
        backendStatus: '서버 상태',
        backendOnline: '정상',
        backendOffline: '연결 안됨',

        // MarketSelector
        marketSelection: '시장 선택',
        marketKorea: '한국',
        marketUsa: '미국',
        marketSweden: '스웨덴',

        // ScreeningPanel
        screeningSettings: '스크리닝 설정',
        matchMode: '매칭 방식',
        matchAll: '모든 조건 일치 (AND)',
        matchAny: '하나라도 일치 (OR)',
        stockLimit: '종목 수 제한',
        limitAll: '전체',
        buyConditions: '매수 조건',
        sellConditions: '매도 조건',
        startScreening: '스크리닝 시작',
        screeningInProgress: '스크리닝 중...',

        // ResultsTable
        screeningResults: '📊 스크리닝 결과',
        statTotal: '전체',
        statBuy: '매수',
        statSell: '매도',
        downloadCsv: '📥 CSV 다운로드',
        buySignals: '🚀 매수 신호',
        sellSignals: '📉 매도 신호',
        colStock: '종목',
        colPrice: '현재가',
        colVolume: '거래량',
        colCondition: '조건',
        noResults: '조건에 맞는 종목이 없습니다',
        clickChart: '클릭하여 차트 보기',

        // PredictionPanel
        analysisAndPrediction: '종목 분석 및 예측',
        enterTickerOrName: '종목 코드 또는 이름 입력',
        startAnalysis: '분석 시작',
        analysisInProgress: '분석 중...',
        currentProposal: '현재 제안',
        targetPrice: '목표가',
        expectedReturn: '예상 수익률',
        aiConfidence: 'AI 확신도',
        forecastDays: '예측 기간',
        days: '일',
        savePredictionCsv: '📥 결과 CSV 저장',
        recommendBuy: '매수 추천',
        recommendSell: '매도 고려',
        recommendHold: '관망',
        recommendConservativeBuy: '보수적 매수 고려',
        recommendConservativeSell: '보수적 매도 고려',
        recommendWait: '관망 권장',
        noteHighConfidence: '높은 신뢰도',
        noteLowConfidenceCautious: '낮은 신뢰도 - 신중 판단 필요',
        noteLowConfidenceUncertain: '낮은 신뢰도 - 불확실한 예측',

        // StockSearch
        searchAndCharts: '종목 검색 및 차트',
        enterTicker: '종목 코드 입력',
        search: '검색',

        // ChartView
        close: '닫기',
        period1M: '1개월',
        period3M: '3개월',
        period6M: '6개월',
        period1Y: '1년',
        period3Y: '3년',
        movingAverages: '이동평균선 (MA20: 황색, MA60: 청색, MA120: 분홍, MA240: 청록)',
        bollingerBands: '볼린저 밴드 (상단/하단: 보라색 점선)',
        volume: '거래량 (상승: 빨강, 하락: 파랑)',
        rsiTitle: 'RSI 지수 (RSI: 황색, 70: 빨강, 30: 파랑)',

        // Conditions
        cond_golden_cross: '골든 크로스',
        cond_rsi_oversold: 'RSI 과매도',
        cond_volume_surge: '거래량 급증',
        cond_enhanced_ma_buy: '강화된 MA 매수',
        cond_enhanced_bb_rsi_buy: '강화된 BB+RSI 매수',
        cond_enhanced_macd_volume_buy: '강화된 MACD+거래량',
        cond_enhanced_momentum_buy: '강화된 모멘텀 매수',
        cond_death_cross: '데드 크로스',
        cond_rsi_overbought: 'RSI 과매수',
        cond_enhanced_technical_sell: '강화된 기술적 매도',
        cond_enhanced_bb_rsi_sell: '강화된 BB+RSI 매도',
    },
    en: {
        // App
        tabScreening: '📊 Screening',
        tabPrediction: '🤖 AI Prediction',
        tabChart: '📈 Chart View',

        headerSubtitle: 'AI-Powered Stock Screening & Prediction System',

        // Backend Status
        backendStatus: 'Backend Status',
        backendOnline: 'Online',
        backendOffline: 'Offline',

        // MarketSelector
        marketSelection: 'Select Market',
        marketKorea: 'Korea',
        marketUsa: 'USA',
        marketSweden: 'Sweden',

        // ScreeningPanel
        screeningSettings: 'Screening Settings',
        matchMode: 'Match Mode',
        matchAll: 'All Conditions (AND)',
        matchAny: 'Any Condition (OR)',
        stockLimit: 'Stock Limit',
        limitAll: 'All',
        buyConditions: 'Buy Conditions',
        sellConditions: 'Sell Conditions',
        startScreening: 'Start Screening',
        screeningInProgress: 'Screening...',

        // ResultsTable
        screeningResults: '📊 Screening Results',
        statTotal: 'Total',
        statBuy: 'Buy',
        statSell: 'Sell',
        downloadCsv: '📥 Download CSV',
        buySignals: '🚀 Buy Signals',
        sellSignals: '📉 Sell Signals',
        colStock: 'Stock',
        colPrice: 'Price',
        colVolume: 'Volume',
        colCondition: 'Conditions',
        noResults: 'No stocks match the conditions',
        clickChart: 'Click to view chart',

        // PredictionPanel
        analysisAndPrediction: 'Stock Analysis & Prediction',
        enterTickerOrName: 'Enter ticker or name',
        startAnalysis: 'Analyze',
        analysisInProgress: 'Analyzing...',
        currentProposal: 'Current Proposal',
        targetPrice: 'Target',
        expectedReturn: 'Exp. Return',
        aiConfidence: 'AI Confidence',
        forecastDays: 'Forecast',
        days: 'Days',
        savePredictionCsv: '📥 Save CSV',
        recommendBuy: 'BUY Recommendation',
        recommendSell: 'SELL Consideration',
        recommendHold: 'HOLD',
        recommendConservativeBuy: 'Conservative Buy Consideration',
        recommendConservativeSell: 'Conservative Sell Consideration',
        recommendWait: 'WAIT Recommendation',
        noteHighConfidence: 'High Confidence',
        noteLowConfidenceCautious: 'Low Confidence - Cautious approach needed',
        noteLowConfidenceUncertain: 'Low Confidence - Uncertain prediction',

        // StockSearch
        searchAndCharts: 'Stock Search & Charts',
        enterTicker: 'Enter ticker',
        search: 'Search',

        // ChartView
        close: 'Close',
        period1M: '1M',
        period3M: '3M',
        period6M: '6M',
        period1Y: '1Y',
        period3Y: '3Y',
        movingAverages: 'Moving Averages (MA20: Yellow, MA60: Blue, MA120: Pink, MA240: Teal)',
        bollingerBands: 'Bollinger Bands (Upper/Lower: Purple Dotted)',
        volume: 'Volume (Up: Red, Down: Blue)',
        rsiTitle: 'RSI (RSI: Yellow, 70: Red, 30: Blue)',

        // Conditions
        cond_golden_cross: 'Golden Cross',
        cond_rsi_oversold: 'RSI Oversold',
        cond_volume_surge: 'Volume Surge',
        cond_enhanced_ma_buy: 'Enhanced MA Buy',
        cond_enhanced_bb_rsi_buy: 'Enhanced BB+RSI Buy',
        cond_enhanced_macd_volume_buy: 'Enhanced MACD+Vol',
        cond_enhanced_momentum_buy: 'Enhanced Momentum',
        cond_death_cross: 'Death Cross',
        cond_rsi_overbought: 'RSI Overbought',
        cond_enhanced_technical_sell: 'Enhanced Tech Sell',
        cond_enhanced_bb_rsi_sell: 'Enhanced BB+RSI Sell',
    }
};
