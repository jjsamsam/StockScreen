import { Language, translations } from '../translations'
import BackendStatus from './BackendStatus'
import './Header.css'

interface HeaderProps {
    language: Language;
    setLanguage: (lang: Language) => void;
    isProcessing?: boolean;
}

function Header({ language, setLanguage, isProcessing }: HeaderProps) {
    const t = translations[language];

    return (
        <header className="header">
            <div className="header-content">
                <div className="logo-group">
                    <div className="logo">
                        <span className="logo-icon">📈</span>
                        <h1>Stock Screener</h1>
                    </div>
                    <BackendStatus language={language} isProcessing={isProcessing} />
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
