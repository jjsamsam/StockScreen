import { useState, useEffect, useRef } from 'react'
import api from '../api'
import { createChart, ColorType, CandlestickSeries, LineSeries, HistogramSeries } from 'lightweight-charts'
import './ChartView.css'
import StockAnalysis from './StockAnalysis'

import { Language, translations } from '../translations'

interface ChartViewProps {
  symbol: string
  onClose: () => void
  language: Language
}

// 지표 가시성 상태
interface IndicatorVisibility {
  ma: boolean
  bb: boolean
  volume: boolean
  rsi: boolean
}

// 현재가 정보 인터페이스
interface QuoteData {
  price: number
  change: number
  change_percent: number
  volume: number
  name: string
}

function ChartView({ symbol, onClose, language }: ChartViewProps) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [period, setPeriod] = useState('1y')
  const [isFullScreen, setIsFullScreen] = useState(false)
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<any>(null)
  const legendRef = useRef<HTMLDivElement>(null)
  const t = translations[language];

  // 🆕 현재가 정보 상태
  const [quote, setQuote] = useState<QuoteData | null>(null)

  // 🆕 지표 토글 상태
  const [indicators, setIndicators] = useState<IndicatorVisibility>({
    ma: true,
    bb: true,
    volume: true,
    rsi: true
  })

  // 🆕 분석 패널 표시 상태
  const [showAnalysis, setShowAnalysis] = useState(false)

  // 시리즈 참조 저장
  const seriesRef = useRef<{
    ma20?: any; ma60?: any; ma120?: any; ma240?: any;
    bbUpper?: any; bbMiddle?: any; bbLower?: any;
    volume?: any; rsi?: any;
  }>({})

  // 🆕 토글 핸들러
  const toggleIndicator = (indicator: keyof IndicatorVisibility) => {
    setIndicators(prev => {
      const newState = { ...prev, [indicator]: !prev[indicator] }

      // 시리즈 가시성 즉시 업데이트
      const series = seriesRef.current
      const chart = chartRef.current

      if (chart) {
        if (indicator === 'ma') {
          series.ma20?.applyOptions({ visible: newState.ma })
          series.ma60?.applyOptions({ visible: newState.ma })
          series.ma120?.applyOptions({ visible: newState.ma })
          series.ma240?.applyOptions({ visible: newState.ma })
        } else if (indicator === 'bb') {
          series.bbUpper?.applyOptions({ visible: newState.bb })
          series.bbMiddle?.applyOptions({ visible: newState.bb })
          series.bbLower?.applyOptions({ visible: newState.bb })
        } else if (indicator === 'volume') {
          series.volume?.applyOptions({ visible: newState.volume })
          chart.priceScale('volume').applyOptions({ visible: newState.volume })
        } else if (indicator === 'rsi') {
          series.rsi?.applyOptions({ visible: newState.rsi })
          chart.priceScale('rsi').applyOptions({ visible: newState.rsi })
        }
      }

      return newState
    })
  }

  const loadChartData = async (selectedPeriod: string) => {
    setLoading(true)
    setError('')

    try {
      const response = await api.get(`/chart/${symbol}`, {
        params: { period: selectedPeriod }
      })

      if (!response.data.success) {
        setError(response.data.error || (language === 'ko' ? '데이터를 불러올 수 없습니다' : 'Could not load data'))
        setLoading(false)
        return
      }

      const data = response.data.data

      // 캔들스틱 데이터 변환
      const candlestickData = data.dates.map((date: string, index: number) => ({
        time: date,
        open: data.open[index],
        high: data.high[index],
        low: data.low[index],
        close: data.close[index],
      }))

      // 이동평균선 데이터
      const ma20Data = data.dates.map((date: string, index: number) => ({
        time: date,
        value: data.ma20[index],
      })).filter((d: any) => d.value && d.value !== 0)

      const ma60Data = data.dates.map((date: string, index: number) => ({
        time: date,
        value: data.ma60[index],
      })).filter((d: any) => d.value && d.value !== 0)

      const ma120Data = data.dates.map((date: string, index: number) => ({
        time: date,
        value: data.ma120[index],
      })).filter((d: any) => d.value && d.value !== 0)

      const ma240Data = data.dates.map((date: string, index: number) => ({
        time: date,
        value: data.ma240[index],
      })).filter((d: any) => d.value && d.value !== 0)

      // 볼린저 밴드 데이터
      const bbUpperData = data.dates.map((date: string, index: number) => ({
        time: date,
        value: data.bb_upper[index],
      })).filter((d: any) => d.value && d.value !== 0)

      const bbMiddleData = data.dates.map((date: string, index: number) => ({
        time: date,
        value: data.bb_middle[index],
      })).filter((d: any) => d.value && d.value !== 0)

      const bbLowerData = data.dates.map((date: string, index: number) => ({
        time: date,
        value: data.bb_lower[index],
      })).filter((d: any) => d.value && d.value !== 0)

      // 거래량 데이터
      const volumeData = data.dates.map((date: string, index: number) => ({
        time: date,
        value: data.volume[index],
        // 한국 표준: 상승(Red), 하락(Blue)
        color: data.close[index] >= data.open[index] ? '#ef444488' : '#2563eb88',
      }))

      // RSI 데이터
      const rsiData = data.dates.map((date: string, index: number) => ({
        time: date,
        value: data.rsi[index],
      })).filter((d: any) => d.value && d.value !== 0)

      // 차트 생성
      if (chartContainerRef.current) {
        // 기존 차트 제거
        if (chartRef.current) {
          try {
            chartRef.current.remove()
          } catch (e) {
            console.warn('Existing chart removal failed:', e)
          }
          chartRef.current = null
        }

        const chart = createChart(chartContainerRef.current, {
          layout: {
            background: { type: ColorType.Solid, color: '#0f172a' },
            textColor: '#cbd5e1',
          },
          grid: {
            vertLines: { visible: false },
            horzLines: { color: '#1e293b', visible: true },
          },
          width: chartContainerRef.current.clientWidth || 800,
          height: isFullScreen ? window.innerHeight - 250 : 600,
          localization: {
            locale: language === 'ko' ? 'ko-KR' : 'en-US',
          },
        })

        chartRef.current = chart

        // 0. 메인 차트 영역 (가격) - 아래쪽 40% 공간 확보 (거래량/RSI용)
        chart.priceScale('right').applyOptions({
          scaleMargins: { top: 0.05, bottom: 0.40 },
          visible: true,
          borderVisible: false,
        })

        // 1. 가격/이동평균선/볼린저밴드 영역
        const candlestickSeries = chart.addSeries(CandlestickSeries, {
          upColor: '#ef4444',     // 상승: 빨강
          downColor: '#2563eb',   // 하락: 파랑
          borderVisible: false,
          wickUpColor: '#ef4444',
          wickDownColor: '#2563eb',
          priceScaleId: 'right',
          priceLineVisible: false,
          lastValueVisible: false,
        })
        candlestickSeries.setData(candlestickData)

        // 볼린저 밴드
        const bbUpperSeries = chart.addSeries(LineSeries, {
          color: '#a855f7',
          lineWidth: 1,
          lineStyle: 2,
          priceScaleId: 'right',
          priceLineVisible: false,
          lastValueVisible: false,
          visible: indicators.bb,
        })
        bbUpperSeries.setData(bbUpperData)

        const bbMiddleSeries = chart.addSeries(LineSeries, {
          color: '#a855f7',
          lineWidth: 1,
          lineStyle: 1,
          priceScaleId: 'right',
          priceLineVisible: false,
          lastValueVisible: false,
          visible: indicators.bb,
        })
        bbMiddleSeries.setData(bbMiddleData)

        const bbLowerSeries = chart.addSeries(LineSeries, {
          color: '#a855f7',
          lineWidth: 1,
          lineStyle: 2,
          priceScaleId: 'right',
          priceLineVisible: false,
          lastValueVisible: false,
          visible: indicators.bb,
        })
        bbLowerSeries.setData(bbLowerData)

        // 이동평균선
        const ma20Series = chart.addSeries(LineSeries, {
          color: '#f59e0b',
          lineWidth: 2,
          priceScaleId: 'right',
          priceLineVisible: false,
          lastValueVisible: false,
          visible: indicators.ma,
        })
        ma20Series.setData(ma20Data)

        const ma60Series = chart.addSeries(LineSeries, {
          color: '#0000ff',
          lineWidth: 2,
          priceScaleId: 'right',
          priceLineVisible: false,
          lastValueVisible: false,
          visible: indicators.ma,
        })
        ma60Series.setData(ma60Data)

        const ma120Series = chart.addSeries(LineSeries, {
          color: '#ff0000',
          lineWidth: 2,
          priceScaleId: 'right',
          priceLineVisible: false,
          lastValueVisible: false,
          visible: indicators.ma,
        })
        ma120Series.setData(ma120Data)

        const ma240Series = chart.addSeries(LineSeries, {
          color: '#14b8a6',
          lineWidth: 2,
          priceScaleId: 'right',
          priceLineVisible: false,
          lastValueVisible: false,
          visible: indicators.ma,
        })
        ma240Series.setData(ma240Data)

        // 2. 거래량 영역
        const volumeSeries = chart.addSeries(HistogramSeries, {
          color: '#26a69a',
          priceFormat: { type: 'volume' },
          priceScaleId: 'volume',
          priceLineVisible: false,
          lastValueVisible: false,
          visible: indicators.volume,
        })
        volumeSeries.setData(volumeData)

        chart.priceScale('volume').applyOptions({
          scaleMargins: { top: 0.65, bottom: 0.20 },
          visible: indicators.volume,
        })

        // 3. RSI 영역
        const rsiSeries = chart.addSeries(LineSeries, {
          color: '#facc15',
          lineWidth: 2,
          priceScaleId: 'rsi',
          priceLineVisible: false,
          lastValueVisible: false,
          visible: indicators.rsi,
        })
        rsiSeries.setData(rsiData)

        chart.priceScale('rsi').applyOptions({
          scaleMargins: { top: 0.85, bottom: 0.05 },
          visible: indicators.rsi,
          borderVisible: false,
        })

        // 시리즈 참조 저장
        seriesRef.current = {
          ma20: ma20Series,
          ma60: ma60Series,
          ma120: ma120Series,
          ma240: ma240Series,
          bbUpper: bbUpperSeries,
          bbMiddle: bbMiddleSeries,
          bbLower: bbLowerSeries,
          volume: volumeSeries,
          rsi: rsiSeries,
        }

        // 다이나믹 툴팁
        const updateLegend = (param: any) => {
          if (!legendRef.current) return;

          const seriesPrices = param.seriesData || new Map();

          const getVal = (series: any) => {
            const val = seriesPrices.get(series);
            return val ? (val.value !== undefined ? val.value : val.close) : null;
          };

          const candleVal = seriesPrices.get(candlestickSeries);

          let dateStr = '';
          if (param.time) {
            dateStr = typeof param.time === 'string' ? param.time :
              `${param.time.year}-${String(param.time.month).padStart(2, '0')}-${String(param.time.day).padStart(2, '0')}`;
          }

          if (candleVal) {
            const open = candleVal.open;
            const close = candleVal.close;
            const high = candleVal.high;
            const low = candleVal.low;

            let prevClose = open;
            let change = 0;
            let changePercent = 0;

            const currentIndex = candlestickData.findIndex((d: any) => d.time === param.time);

            if (currentIndex > 0) {
              prevClose = candlestickData[currentIndex - 1].close;
              change = close - prevClose;
              changePercent = (change / prevClose) * 100;
            } else {
              change = close - open;
              changePercent = (change / open) * 100;
            }

            const color = change >= 0 ? '#ef4444' : '#2563eb';
            const sign = change > 0 ? '+' : '';

            const vol = getVal(volumeSeries);
            const rsi = getVal(rsiSeries);
            const ma20 = getVal(ma20Series);
            const ma60 = getVal(ma60Series);
            const ma120 = getVal(ma120Series);
            const ma240 = getVal(ma240Series);
            const bbUp = getVal(bbUpperSeries);
            const bbLow = getVal(bbLowerSeries);

            const volStr = vol ? (vol >= 1000000 ? `${(vol / 1000000).toFixed(1)}M` : (vol >= 1000 ? `${(vol / 1000).toFixed(1)}K` : vol)) : '-';

            legendRef.current.innerHTML = `
               <div class="legend-date">📅 ${dateStr}</div>
               
               <div class="legend-price">
                 <span class="price-value" style="color: ${color};">${close.toLocaleString()}</span>
                 <span class="price-change" style="color: ${color};">${sign}${change.toLocaleString()} (${sign}${changePercent.toFixed(2)}%)</span>
               </div>

               <div class="legend-ohlc">
                 <span>O: ${open.toLocaleString()}</span>
                 <span>H: ${high.toLocaleString()}</span>
                 <span>L: ${low.toLocaleString()}</span>
                 <span>V: ${volStr}</span>
               </div>

               ${indicators.ma ? `<div class="legend-ma">
                 <span style="color: #f59e0b;">MA20: ${ma20 ? ma20.toFixed(0) : '-'}</span>
                 <span style="color: #0000ff;">MA60: ${ma60 ? ma60.toFixed(0) : '-'}</span>
                 <span style="color: #ff0000;">MA120: ${ma120 ? ma120.toFixed(0) : '-'}</span>
                 <span style="color: #14b8a6;">MA240: ${ma240 ? ma240.toFixed(0) : '-'}</span>
               </div>` : ''}

               ${indicators.bb || indicators.rsi ? `<div class="legend-indicators">
                 ${indicators.bb ? `<span style="color: #a855f7;">BB: ${bbUp ? bbUp.toFixed(0) : '-'}~${bbLow ? bbLow.toFixed(0) : '-'}</span>` : ''}
                 ${indicators.rsi ? `<span style="color: #facc15;">RSI: ${rsi ? rsi.toFixed(1) : '-'}</span>` : ''}
               </div>` : ''}
             `;
          }
        };

        chart.subscribeCrosshairMove(updateLegend);
        chart.timeScale().fitContent()
      }

      setLoading(false)
    } catch (err: any) {
      console.error('차트 로드 실패:', err)
      setError(err.response?.data?.detail || (language === 'ko' ? '차트 데이터를 불러올 수 없습니다' : 'Could not fetch chart data'))
      setLoading(false)
    }
  }

  // Handle Resize
  useEffect(() => {
    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: isFullScreen ? window.innerHeight - 250 : 500,
        })
      }
    }

    const resizeObserver = new ResizeObserver(() => {
      handleResize()
    })

    if (chartContainerRef.current) {
      resizeObserver.observe(chartContainerRef.current)
    }

    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      resizeObserver.disconnect()
    }
  }, [isFullScreen])

  useEffect(() => {
    loadChartData(period)

    return () => {
      if (chartRef.current) {
        try {
          chartRef.current.remove()
        } catch (e) {
          console.warn('Effect cleanup chart removal failed:', e)
        }
        chartRef.current = null
      }
    }
  }, [period])

  // 🆕 현재가 정보 로드
  useEffect(() => {
    const fetchQuote = async () => {
      try {
        const response = await api.get(`/quote/${symbol}`)
        if (response.data.success) {
          setQuote(response.data.data)
        }
      } catch (err) {
        console.warn('현재가 조회 실패:', err)
      }
    }
    fetchQuote()
  }, [symbol])

  const handlePeriodChange = (newPeriod: string) => {
    if (newPeriod !== period) {
      setPeriod(newPeriod)
    }
  }

  const toggleFullScreen = () => {
    setIsFullScreen(!isFullScreen)
  }

  const getPeriodLabel = (p: string) => {
    if (p === '1mo') return t.period1M;
    if (p === '3mo') return t.period3M;
    if (p === '6mo') return t.period6M;
    if (p === '1y') return t.period1Y;
    if (p === '2y') return language === 'ko' ? '2년' : '2Y';
    if (p === '5y') return language === 'ko' ? '5년' : '5Y';
    return p;
  };

  return (
    <div className={`chart-overlay ${isFullScreen ? 'full-screen-mode' : ''}`} onClick={onClose}>
      <div className={`chart-modal ${isFullScreen ? 'is-full' : ''}`} onClick={(e) => e.stopPropagation()}>
        <div className="chart-header">
          <div className="header-title">
            <h2>📈 {symbol}</h2>
            {quote && (
              <div className="header-quote">
                <span className="quote-price">{quote.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                <span className={`quote-change ${quote.change >= 0 ? 'positive' : 'negative'}`}>
                  {quote.change >= 0 ? '+' : ''}{quote.change.toFixed(2)} ({quote.change_percent >= 0 ? '+' : ''}{quote.change_percent.toFixed(2)}%)
                </span>
                <span className="quote-vol">Vol: {quote.volume >= 1e6 ? (quote.volume / 1e6).toFixed(1) + 'M' : quote.volume >= 1e3 ? (quote.volume / 1e3).toFixed(1) + 'K' : quote.volume}</span>
              </div>
            )}
          </div>
          <div className="header-actions">
            <button className="maximize-btn" onClick={toggleFullScreen} title={isFullScreen ? (language === 'ko' ? "축소" : "Minimize") : (language === 'ko' ? "확대" : "Maximize")}>
              {isFullScreen ? '🔳' : '⬜'}
            </button>
            <button className="close-btn" onClick={onClose} title={t.close}>✕</button>
          </div>
        </div>

        {/* 기간 선택 */}
        <div className="period-selector">
          {['1mo', '3mo', '6mo', '1y', '2y', '5y'].map(p => (
            <button
              key={p}
              className={`period-btn ${period === p ? 'active' : ''}`}
              onClick={() => handlePeriodChange(p)}
            >
              {getPeriodLabel(p)}
            </button>
          ))}
        </div>

        {/* 🆕 지표 토글 버튼 */}
        <div className="indicator-toggles">
          <button
            className={`toggle-btn ${indicators.ma ? 'active' : ''}`}
            onClick={() => toggleIndicator('ma')}
          >
            📊 {language === 'ko' ? '이동평균' : 'MA'}
          </button>
          <button
            className={`toggle-btn ${indicators.bb ? 'active' : ''}`}
            onClick={() => toggleIndicator('bb')}
          >
            📈 {language === 'ko' ? '볼린저' : 'BB'}
          </button>
          <button
            className={`toggle-btn ${indicators.volume ? 'active' : ''}`}
            onClick={() => toggleIndicator('volume')}
          >
            📶 {language === 'ko' ? '거래량' : 'Vol'}
          </button>
          <button
            className={`toggle-btn ${indicators.rsi ? 'active' : ''}`}
            onClick={() => toggleIndicator('rsi')}
          >
            ⚡ RSI
          </button>

          {/* 🆕 분석 패널 토글 버튼 */}
          <button
            className={`toggle-btn analysis-toggle ${showAnalysis ? 'active' : ''}`}
            onClick={() => setShowAnalysis(!showAnalysis)}
            style={{ marginLeft: 'auto' }}
          >
            📊 {language === 'ko' ? '기술적 분석' : 'Analysis'}
          </button>
        </div>

        {error && <div className="chart-error">❌ {error}</div>}

        {/* 🆕 분석 패널 표시 */}
        {showAnalysis && (
          <div className="analysis-panel-container">
            <StockAnalysis ticker={symbol} language={language} />
          </div>
        )}

        <div className="chart-container" style={{
          visibility: loading ? 'hidden' : 'visible',
          position: 'relative',
          height: isFullScreen ? 'calc(100vh - 200px)' : 'auto'
        }}>
          {loading && <div className="chart-loading-overlay">{language === 'ko' ? '차트 로딩 중...' : 'Loading chart...'}</div>}
          <div ref={chartContainerRef} style={{
            width: '100%',
            height: isFullScreen ? '100%' : '500px',
            touchAction: 'none',
            userSelect: 'none',
            WebkitUserSelect: 'none',
            WebkitTouchCallout: 'none'
          }} />

          {!error && (
            <div
              ref={legendRef}
              className="chart-dynamic-legend"
              style={{
                position: 'absolute',
                top: '10px',
                left: '10px',
                zIndex: 20,
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                border: '1px solid #334155',
                borderRadius: '8px',
                padding: '10px',
                color: '#cbd5e1',
                pointerEvents: 'none',
                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
                backdropFilter: 'blur(4px)',
                transition: 'opacity 0.1s ease',
              }}
            >
              <div style={{ fontSize: '12px', color: '#64748b' }}>
                {language === 'ko' ? '👆 차트를 터치하여 정보 확인' : '👆 Touch chart for details'}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ChartView