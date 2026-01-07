import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { createChart, ColorType, CandlestickSeries, LineSeries, HistogramSeries } from 'lightweight-charts'
import './ChartView.css'

import { Language, translations } from '../translations'

interface ChartViewProps {
  symbol: string
  onClose: () => void
  language: Language
}

function ChartView({ symbol, onClose, language }: ChartViewProps) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [period, setPeriod] = useState('1y')
  const [isFullScreen, setIsFullScreen] = useState(false)
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<any>(null)
  const t = translations[language];

  const loadChartData = async (selectedPeriod: string) => {
    setLoading(true)
    setError('')

    try {
      const response = await axios.get(`/api/chart/${symbol}`, {
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
            horzLines: { color: '#334155' },
          },
          width: chartContainerRef.current.clientWidth || 800,
          height: isFullScreen ? window.innerHeight - 250 : 600,
        })

        chartRef.current = chart

        // 1. 가격/이동평균선/볼린저밴드 영역
        const candlestickSeries = chart.addSeries(CandlestickSeries, {
          upColor: '#ef4444',     // 상승: 빨강 (한국 표준)
          downColor: '#2563eb',   // 하락: 파랑 (한국 표준)
          borderVisible: false,
          wickUpColor: '#ef4444',
          wickDownColor: '#2563eb',
          priceScaleId: 'right',
        })
        candlestickSeries.setData(candlestickData)

        // 볼린저 밴드
        const bbUpperSeries = chart.addSeries(LineSeries, {
          color: '#a855f7',
          lineWidth: 1,
          lineStyle: 2,
          priceScaleId: 'right',
        })
        bbUpperSeries.setData(bbUpperData)

        const bbMiddleSeries = chart.addSeries(LineSeries, {
          color: '#a855f7',
          lineWidth: 1,
          lineStyle: 1,
          priceScaleId: 'right',
        })
        bbMiddleSeries.setData(bbMiddleData)

        const bbLowerSeries = chart.addSeries(LineSeries, {
          color: '#a855f7',
          lineWidth: 1,
          lineStyle: 2,
          priceScaleId: 'right',
        })
        bbLowerSeries.setData(bbLowerData)

        // 이동평균선
        const ma20Series = chart.addSeries(LineSeries, { color: '#f59e0b', lineWidth: 2, priceScaleId: 'right' })
        ma20Series.setData(ma20Data)

        const ma60Series = chart.addSeries(LineSeries, { color: '#6366f1', lineWidth: 2, priceScaleId: 'right' })
        ma60Series.setData(ma60Data)

        const ma120Series = chart.addSeries(LineSeries, { color: '#ec4899', lineWidth: 2, priceScaleId: 'right' })
        ma120Series.setData(ma120Data)

        const ma240Series = chart.addSeries(LineSeries, { color: '#14b8a6', lineWidth: 2, priceScaleId: 'right' })
        ma240Series.setData(ma240Data)

        // 2. 거래량 영역
        const volumeSeries = chart.addSeries(HistogramSeries, {
          color: '#26a69a',
          priceFormat: { type: 'volume' },
          priceScaleId: 'volume',
        })
        volumeSeries.setData(volumeData)

        chart.priceScale('volume').applyOptions({
          scaleMargins: { top: 0.75, bottom: 0.05 },
        })

        // 3. RSI 영역
        const rsiSeries = chart.addSeries(LineSeries, {
          color: '#facc15',
          lineWidth: 2,
          priceScaleId: 'rsi',
        })
        rsiSeries.setData(rsiData)

        // RSI 기준선 (70, 30)
        rsiSeries.createPriceLine({
          price: 70,
          color: '#ef4444',
          lineWidth: 1,
          lineStyle: 3,
          axisLabelVisible: true,
          title: 'Overbought',
        });
        rsiSeries.createPriceLine({
          price: 30,
          color: '#2563eb', // 과매수(하락 반전 가능성)와 대비되는 과매도(상승 반전 가능성) 색상
          lineWidth: 1,
          lineStyle: 3,
          axisLabelVisible: true,
          title: 'Oversold',
        });

        chart.priceScale('rsi').applyOptions({
          scaleMargins: { top: 0.85, bottom: 0.05 },
          visible: true,
          borderVisible: true,
        })

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
          <h2>📈 {symbol}</h2>
          <div className="header-actions">
            <button className="maximize-btn" onClick={toggleFullScreen} title={isFullScreen ? (language === 'ko' ? "축소" : "Minimize") : (language === 'ko' ? "확대" : "Maximize")}>
              {isFullScreen ? '🔳' : '⬜'}
            </button>
            <button className="close-btn" onClick={onClose} title={t.close}>✕</button>
          </div>
        </div>

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

        {error && <div className="chart-error">❌ {error}</div>}

        <div className="chart-container" style={{
          visibility: loading ? 'hidden' : 'visible',
          position: 'relative',
          height: isFullScreen ? 'calc(100vh - 200px)' : 'auto'
        }}>
          {loading && <div className="chart-loading-overlay">{language === 'ko' ? '차트 로딩 중...' : 'Loading chart...'}</div>}
          <div ref={chartContainerRef} style={{ width: '100%', height: isFullScreen ? '100%' : '500px' }} />
          {!error && (
            <div className="chart-legend">
              <span style={{ color: '#f59e0b' }}>━ MA20</span>
              <span style={{ color: '#6366f1' }}>━ MA60</span>
              <span style={{ color: '#ec4899' }}>━ MA120</span>
              <span style={{ color: '#14b8a6' }}>━ MA240</span>
              <span style={{ color: '#a855f7' }}>┉ {language === 'ko' ? '볼린저밴드' : 'BB'}</span>
              <span style={{ color: '#ef4444' }}>■ {language === 'ko' ? '거래량' : 'Volume'}</span>
              <span style={{ color: '#facc15' }}>━ RSI</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ChartView