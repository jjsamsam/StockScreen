@echo off
echo ===================================================
echo Stock Screener Mobile Server
echo ===================================================
echo.
echo Starting Backend Server...
start "Stock Backend" cmd /k "cd backend && python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload"

echo Starting Frontend Server...
start "Stock Frontend" cmd /k "cd frontend && npm run dev -- --host"

echo.
echo ===================================================
echo Server is running!
echo.
echo [Action Required]
echo 1. Connect your mobile device to the same Wi-Fi.
echo 2. Open browser on mobile.
echo 3. Go to: http://192.168.50.76:3000
echo.
echo (Press any key to close this launcher, servers will keep running)
echo ===================================================
pause
