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
  const legendRef = useRef<HTMLDivElement>(null)
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
            horzLines: { color: '#1e293b', visible: true }, // 배경 그리드 복구 (아주 연하게)
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
          priceLineVisible: false, // 현재가 표시선 제거
          lastValueVisible: false, // Y축 라벨 숨김
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
        })
        bbUpperSeries.setData(bbUpperData)

        const bbMiddleSeries = chart.addSeries(LineSeries, {
          color: '#a855f7',
          lineWidth: 1,
          lineStyle: 1,
          priceScaleId: 'right',
          priceLineVisible: false,
          lastValueVisible: false,
        })
        bbMiddleSeries.setData(bbMiddleData)

        const bbLowerSeries = chart.addSeries(LineSeries, {
          color: '#a855f7',
          lineWidth: 1,
          lineStyle: 2,
          priceScaleId: 'right',
          priceLineVisible: false,
          lastValueVisible: false,
        })
        bbLowerSeries.setData(bbLowerData)

        // 이동평균선
        const ma20Series = chart.addSeries(LineSeries, {
          color: '#f59e0b',
          lineWidth: 2,
          priceScaleId: 'right',
          priceLineVisible: false,
          lastValueVisible: false,
        })
        ma20Series.setData(ma20Data)

        const ma60Series = chart.addSeries(LineSeries, {
          color: '#0000ff', // 파란색
          lineWidth: 2,
          priceScaleId: 'right',
          priceLineVisible: false,
          lastValueVisible: false,
        })
        ma60Series.setData(ma60Data)

        const ma120Series = chart.addSeries(LineSeries, {
          color: '#ff0000', // 빨간색
          lineWidth: 2,
          priceScaleId: 'right',
          priceLineVisible: false,
          lastValueVisible: false,
        })
        ma120Series.setData(ma120Data)

        const ma240Series = chart.addSeries(LineSeries, {
          color: '#14b8a6',
          lineWidth: 2,
          priceScaleId: 'right',
          priceLineVisible: false,
          lastValueVisible: false,
        })
        ma240Series.setData(ma240Data)

        // 2. 거래량 영역
        const volumeSeries = chart.addSeries(HistogramSeries, {
          color: '#26a69a',
          priceFormat: { type: 'volume' },
          priceScaleId: 'volume',
          priceLineVisible: false,
          lastValueVisible: false,
        })
        volumeSeries.setData(volumeData)

        chart.priceScale('volume').applyOptions({
          scaleMargins: { top: 0.65, bottom: 0.20 },
        })

        // 3. RSI 영역
        const rsiSeries = chart.addSeries(LineSeries, {
          color: '#facc15',
          lineWidth: 2,
          priceScaleId: 'rsi',
          priceLineVisible: false,
          lastValueVisible: false,
        })
        rsiSeries.setData(rsiData)

        // RSI 기준선 제거 (사용자 요청)

        chart.priceScale('rsi').applyOptions({
          scaleMargins: { top: 0.85, bottom: 0.05 },
          visible: true,
          borderVisible: false,
        })

        // =========================================================
        // 💫 다이나믹 툴팁 (크로스헤어 핸들러)
        // =========================================================
        const updateLegend = (param: any) => {
          if (!legendRef.current) return;




          // 데이터 가져오기
          const seriesPrices = param.seriesData || new Map();

          // Helper
          const getVal = (series: any) => {
            const val = seriesPrices.get(series);
            return val ? (val.value !== undefined ? val.value : val.close) : null;
          };

          const candleVal = seriesPrices.get(candlestickSeries);

          // 날짜 포맷팅
          let dateStr = '';
          if (param.time) {
            // param.time이 string일 수도 있고 object일 수도 있음 (lightweight-charts 버전에 따라 다름)
            dateStr = typeof param.time === 'string' ? param.time :
              `${param.time.year}-${String(param.time.month).padStart(2, '0')}-${String(param.time.day).padStart(2, '0')}`;
          }

          if (candleVal) {
            const open = candleVal.open;
            const close = candleVal.close;
            const high = candleVal.high;
            const low = candleVal.low;

            // ✅ 전일 종가 대비 등락폭 계산 (Change from Previous Close)
            // candlestickData 배열에서 현재 날짜의 인덱스를 찾고, 그 전날 데이터를 가져옴
            let prevClose = open; // 데이터가 없으면 시가를 기준으로 (당일 등락) -> 0%로 시작
            let change = 0;
            let changePercent = 0;

            // 현재 데이터의 인덱스 찾기 (시간 기준)
            const currentIndex = candlestickData.findIndex((d: any) => d.time === param.time);

            if (currentIndex > 0) {
              prevClose = candlestickData[currentIndex - 1].close;
              change = close - prevClose;
              changePercent = (change / prevClose) * 100;
            } else {
              // 첫 날인 경우: 시가 기준 등락폭 (오늘 얼마나 움직였나) or 0
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
               <div style="font-size: 14px; font-weight: bold; margin-bottom: 6px; color: #e2e8f0; border-bottom: 1px solid #334155; padding-bottom: 4px;">📅 ${dateStr}</div>
               
               <div style="display: flex; gap: 12px; align-items: baseline; margin-bottom: 8px;">
                 <span style="font-size: 20px; font-weight: bold; color: ${color};">${close.toLocaleString()}</span>
                 <span style="color: ${color}; font-size: 14px;">${sign}${change.toLocaleString()} (${sign}${changePercent.toFixed(2)}%)</span>
               </div>

               <div style="display: grid; grid-template-columns: 1fr 1fr; gap: x 16px; row-gap: 2px; font-size: 12px; color: #94a3b8; margin-bottom: 8px;">
                 <div>O: <span style="color: #cbd5e1">${open.toLocaleString()}</span></div>
                 <div>H: <span style="color: #cbd5e1">${high.toLocaleString()}</span></div>
                 <div>L: <span style="color: #cbd5e1">${low.toLocaleString()}</span></div>
                 <div>Vol: <span style="color: #cbd5e1">${volStr}</span></div>
               </div>

               <div style="margin-top: 8px; border-top: 1px dotted #475569; padding-top: 6px; display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 12px;">
                 <div style="color: #f59e0b;">MA20: ${ma20 ? ma20.toFixed(0) : '-'}</div>
                 <div style="color: #0000ff;">MA60: ${ma60 ? ma60.toFixed(0) : '-'}</div>
                 <div style="color: #ff0000;">MA120: ${ma120 ? ma120.toFixed(0) : '-'}</div>
                 <div style="color: #14b8a6;">MA240: ${ma240 ? ma240.toFixed(0) : '-'}</div>
               </div>

               <div style="margin-top: 4px; display: grid; grid-template-columns: 1fr; gap: 2px; font-size: 12px;">
                 <div style="color: #a855f7;">BB: ${bbUp ? bbUp.toFixed(0) : '-'} ~ ${bbLow ? bbLow.toFixed(0) : '-'}</div>
                 <div style="color: #facc15;">RSI: ${rsi ? rsi.toFixed(1) : '-'}</div>
               </div>
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
          <div ref={chartContainerRef} style={{
            width: '100%',
            height: isFullScreen ? '100%' : '500px',
            touchAction: 'none',        // 스크롤 방지
            userSelect: 'none',         // 텍스트 선택 방지
            WebkitUserSelect: 'none',   // iOS 텍스트 선택 방지
            WebkitTouchCallout: 'none'  // iOS 꾹 누르기 메뉴 방지
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
                backgroundColor: 'rgba(15, 23, 42, 0.9)', // 어두운 반투명 배경
                border: '1px solid #334155',
                borderRadius: '8px',
                padding: '12px',
                color: '#cbd5e1',
                pointerEvents: 'none', // 마우스 통과 (차트 조작 가능)
                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
                minWidth: '200px',
                backdropFilter: 'blur(4px)',
                transition: 'opacity 0.1s ease',
              }}
            >
              {/* 초기 안내 메시지 */}
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