import { Language, translations } from '../translations'
import './Header.css'

interface HeaderProps {
    language: Language;
    setLanguage: (lang: Language) => void;
}

function Header({ language, setLanguage }: HeaderProps) {
    const t = translations[language];

    return (
        <header className="header">
            <div className="header-content">
                <div className="logo">
                    <span className="logo-icon">📈</span>
                    <h1>Stock Screener</h1>
                </div>
                <div className="header-right">
                    <div className="subtitle">
                        {t.headerSubtitle}
                    </div>
                    <div className="lang-selector">
                        <button
                            className={`lang-btn ${language === 'ko' ? 'active' : ''}`}
                            onClick={() => setLanguage('ko')}
                        >
                            KO
                        </button>
                        <button
                            className={`lang-btn ${language === 'en' ? 'active' : ''}`}
                            onClick={() => setLanguage('en')}
                        >
                            EN
                        </button>
                    </div>
                </div>
            </div>
        </header>
    )
}

export default Header
