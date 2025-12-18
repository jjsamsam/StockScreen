import './Header.css'

function Header() {
    return (
        <header className="header">
            <div className="header-content">
                <div className="logo">
                    <span className="logo-icon">📈</span>
                    <h1>Stock Screener</h1>
                </div>
                <div className="subtitle">
                    AI 기반 주식 스크리닝 & 예측 시스템
                </div>
            </div>
        </header>
    )
}

export default Header
