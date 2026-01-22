import axios from 'axios';

// Vercel 배포 시 환경 변수에서 API URL을 가져옴
// 개발 환경에서는 프록시(/api) 사용 또는 localhost 사용
const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || '/api',
    headers: {
        'Content-Type': 'application/json',
    }
});

export default api;
