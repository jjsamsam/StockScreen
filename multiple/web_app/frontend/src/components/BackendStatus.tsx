import { useState, useEffect } from 'react'
import axios from 'axios'
import { Language, translations } from '../translations'
import './BackendStatus.css'

interface BackendStatusProps {
    language: Language;
    isProcessing?: boolean;
}

function BackendStatus({ language, isProcessing }: BackendStatusProps) {
    const [isOnline, setIsOnline] = useState<boolean | null>(null)
    const t = translations[language]

    const checkHealth = async () => {
        // 처리 중일 때는 연결된 것으로 간주 (백엔드가 바빠서 응답 못할 수 있음)
        if (isProcessing) {
            setIsOnline(true);
            return;
        }

        try {
            await axios.get('/api/health', { timeout: 5000 })
            setIsOnline(true)
        } catch (error) {
            setIsOnline(false)
        }
    }

    useEffect(() => {
        // Initial check
        checkHealth()

        // Check every 30 seconds
        const interval = setInterval(checkHealth, 30000)
        return () => clearInterval(interval)
    }, [isProcessing]) // isProcessing이 바뀌면 즉시 상태 반영

    return (
        <div className={`backend-status ${isOnline === true ? 'online' : isOnline === false ? 'offline' : 'checking'}`}>
            <span className="status-dot"></span>
            <span className="status-text">
                {isOnline === true ? t.backendOnline : isOnline === false ? t.backendOffline : '...'}
            </span>
        </div>
    )
}

export default BackendStatus
