@echo off
title StockScreener Installer
echo.
echo =========================================
echo    StockScreener Installation Wizard
echo =========================================
echo.

# Set installation path
set "INSTALL_PATH=%USERPROFILE%\Desktop\StockScreener"

echo 📍 Installation location: %INSTALL_PATH%
echo.

# Create installation directory
if not exist "%INSTALL_PATH%" (
    mkdir "%INSTALL_PATH%"
    echo ✅ Created installation directory
)

# Extract using PowerShell (safer method)
echo 📦 Extracting application files...
powershell -Command "try { Expand-Archive -Path '%~dp0stockscreener_package.zip' -DestinationPath '%INSTALL_PATH%' -Force; Write-Host 'Extraction completed' } catch { Write-Host 'Extraction failed:' $_.Exception.Message }"

# Verify installation
if exist "%INSTALL_PATH%\StockScreener.exe" (
    echo ✅ StockScreener.exe found
) else (
    echo ❌ StockScreener.exe not found - installation may have failed
    goto :error
)

if exist "%INSTALL_PATH%\stock_data" (
    echo ✅ stock_data folder found
) else (
    echo ⚠️ stock_data folder not found
)

if exist "%INSTALL_PATH%\master_csv" (
    echo ✅ master_csv folder found
) else (
    echo ⚠️ master_csv folder not found
)

# Create desktop shortcut
echo 🔗 Creating desktop shortcut...
powershell -Command "try { $WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\StockScreener.lnk'); $Shortcut.TargetPath = '%INSTALL_PATH%\StockScreener.exe'; $Shortcut.WorkingDirectory = '%INSTALL_PATH%'; $Shortcut.Save(); Write-Host 'Shortcut created' } catch { Write-Host 'Shortcut creation failed' }"

echo.
echo 🎉 Installation completed successfully!
echo.
echo 📋 Installation summary:
echo    📁 Location: %INSTALL_PATH%
echo    🔗 Desktop shortcut: Created
echo    📊 CSV data: Included
echo.
echo 💡 To start StockScreener:
echo    - Double-click the desktop shortcut, or
echo    - Go to %INSTALL_PATH% and run StockScreener.exe
echo.
goto :end

:error
echo.
echo ❌ Installation failed!
echo Please check if:
echo - You have sufficient permissions
echo - The package file is not corrupted
echo - Antivirus is not blocking the installation
echo.

:end
pause
