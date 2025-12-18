import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { createChart, ColorType, CandlestickSeries, LineSeries } from 'lightweight-charts'
import './ChartView.css'

interface ChartViewProps {
  symbol: string
  onClose: () => void
}

function ChartView({ symbol, onClose }: ChartViewProps) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [period, setPeriod] = useState('1y')
  const [isFullScreen, setIsFullScreen] = useState(false)
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<any>(null)

  const loadChartData = async (selectedPeriod: string) => {
    setLoading(true)
    setError('')

    try {
      const response = await axios.get(`/api/chart/${symbol}`, {
        params: { period: selectedPeriod }
      })

      if (!response.data.success) {
        setError(response.data.error || '데이터를 불러올 수 없습니다')
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
            vertLines: { color: '#475569' },
            horzLines: { color: '#475569' },
          },
          width: chartContainerRef.current.clientWidth || 800,
          height: isFullScreen ? window.innerHeight - 250 : 500,
        })

        chartRef.current = chart

        // 캔들스틱 시리즈 (v5 API)
        const candlestickSeries = chart.addSeries(CandlestickSeries, {
          upColor: '#10b981',
          downColor: '#ef4444',
          borderVisible: false,
          wickUpColor: '#10b981',
          wickDownColor: '#ef4444',
        })
        candlestickSeries.setData(candlestickData)

        // 볼린저 밴드
        const bbUpperSeries = chart.addSeries(LineSeries, {
          color: '#8b5cf6',
          lineWidth: 1,
          lineStyle: 2,
        })
        bbUpperSeries.setData(bbUpperData)

        const bbMiddleSeries = chart.addSeries(LineSeries, {
          color: '#8b5cf6',
          lineWidth: 1,
          lineStyle: 2,
        })
        bbMiddleSeries.setData(bbMiddleData)

        const bbLowerSeries = chart.addSeries(LineSeries, {
          color: '#8b5cf6',
          lineWidth: 1,
          lineStyle: 2,
        })
        bbLowerSeries.setData(bbLowerData)

        // 이동평균선
        const ma20Series = chart.addSeries(LineSeries, {
          color: '#f59e0b',
          lineWidth: 2,
        })
        ma20Series.setData(ma20Data)

        const ma60Series = chart.addSeries(LineSeries, {
          color: '#6366f1',
          lineWidth: 2,
        })
        ma60Series.setData(ma60Data)

        const ma120Series = chart.addSeries(LineSeries, {
          color: '#ec4899',
          lineWidth: 2,
        })
        ma120Series.setData(ma120Data)

        const ma240Series = chart.addSeries(LineSeries, {
          color: '#14b8a6',
          lineWidth: 2,
        })
        ma240Series.setData(ma240Data)

        chart.timeScale().fitContent()
      }

      setLoading(false)
    } catch (err: any) {
      console.error('차트 로드 실패:', err)
      setError(err.response?.data?.detail || '차트 데이터를 불러올 수 없습니다')
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

  return (
    <div className={`chart-overlay ${isFullScreen ? 'full-screen-mode' : ''}`} onClick={onClose}>
      <div className={`chart-modal ${isFullScreen ? 'is-full' : ''}`} onClick={(e) => e.stopPropagation()}>
        <div className="chart-header">
          <h2>📈 {symbol}</h2>
          <div className="header-actions">
            <button className="maximize-btn" onClick={toggleFullScreen} title={isFullScreen ? "축소" : "확대"}>
              {isFullScreen ? '🔳' : '⬜'}
            </button>
            <button className="close-btn" onClick={onClose}>✕</button>
          </div>
        </div>

        <div className="period-selector">
          {['1mo', '3mo', '6mo', '1y', '2y', '5y'].map(p => (
            <button
              key={p}
              className={`period-btn ${period === p ? 'active' : ''}`}
              onClick={() => handlePeriodChange(p)}
            >
              {p}
            </button>
          ))}
        </div>

        {error && <div className="chart-error">❌ {error}</div>}

        <div className="chart-container" style={{
          visibility: loading ? 'hidden' : 'visible',
          position: 'relative',
          height: isFullScreen ? 'calc(100vh - 200px)' : 'auto'
        }}>
          {loading && <div className="chart-loading-overlay">차트 로딩 중...</div>}
          <div ref={chartContainerRef} style={{ width: '100%', height: isFullScreen ? '100%' : '500px' }} />
          {!error && (
            <div className="chart-legend">
              <span style={{ color: '#f59e0b' }}>━ MA20</span>
              <span style={{ color: '#6366f1' }}>━ MA60</span>
              <span style={{ color: '#ec4899' }}>━ MA120</span>
              <span style={{ color: '#14b8a6' }}>━ MA240</span>
              <span style={{ color: '#8b5cf6' }}>┉ 볼린저밴드</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ChartView