import { useState, useEffect } from 'react'
import axios from 'axios'
import { Language, translations } from '../translations'
import './BackendStatus.css'

interface BackendStatusProps {
    language: Language;
}

function BackendStatus({ language }: BackendStatusProps) {
    const [isOnline, setIsOnline] = useState<boolean | null>(null)
    const t = translations[language]

    const checkHealth = async () => {
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
    }, [])

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
