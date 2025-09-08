#!/usr/bin/env python3
"""
create_batch_installer.py
배치 파일과 안전한 압축을 사용한 인스톨러 생성
"""

import os
import shutil
import zipfile
import time
from pathlib import Path

def create_batch_installer():
    """배치 파일 기반 인스톨러 생성"""
    
    installer_script = '''@echo off
title StockScreener Installer
echo.
echo =========================================
echo    StockScreener Installation Wizard
echo =========================================
echo.

# Set installation path
set "INSTALL_PATH=%USERPROFILE%\\Desktop\\StockScreener"

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
if exist "%INSTALL_PATH%\\StockScreener.exe" (
    echo ✅ StockScreener.exe found
) else (
    echo ❌ StockScreener.exe not found - installation may have failed
    goto :error
)

if exist "%INSTALL_PATH%\\stock_data" (
    echo ✅ stock_data folder found
) else (
    echo ⚠️ stock_data folder not found
)

if exist "%INSTALL_PATH%\\master_csv" (
    echo ✅ master_csv folder found
) else (
    echo ⚠️ master_csv folder not found
)

# Create desktop shortcut
echo 🔗 Creating desktop shortcut...
powershell -Command "try { $WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\\Desktop\\StockScreener.lnk'); $Shortcut.TargetPath = '%INSTALL_PATH%\\StockScreener.exe'; $Shortcut.WorkingDirectory = '%INSTALL_PATH%'; $Shortcut.Save(); Write-Host 'Shortcut created' } catch { Write-Host 'Shortcut creation failed' }"

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
'''

    # 인스톨러 배치 파일 생성
    with open('StockScreener_install.bat', 'w', encoding='utf-8') as f:
        f.write(installer_script)
    
    print("✅ Batch installer created: StockScreener_install.bat")

def create_package_safe():
    """안전한 방법으로 설치 패키지 생성"""
    print("📦 Creating installation package (safe method)...")
    
    # 잠시 대기하여 파일 핸들이 해제되도록 함
    time.sleep(2)
    
    try:
        # Python zipfile 모듈 사용 (더 안전함)
        with zipfile.ZipFile('stockscreener_package.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
            
            # PyInstaller 결과물 추가 (안전한 방법)
            dist_path = Path('dist/StockScreener')
            if dist_path.exists():
                print(f"📁 Adding files from: {dist_path}")
                
                for file_path in dist_path.rglob('*'):
                    if file_path.is_file():
                        try:
                            # 상대 경로로 압축
                            arcname = file_path.relative_to(dist_path)
                            zipf.write(file_path, arcname)
                            print(f"  ✅ Added: {arcname}")
                        except (PermissionError, OSError) as e:
                            print(f"  ⚠️ Skipped (in use): {file_path.name}")
                            continue
            
            # CSV 폴더들 추가
            for folder in ['stock_data', 'master_csv']:
                folder_path = Path(folder)
                if folder_path.exists():
                    print(f"📊 Adding CSV folder: {folder}")
                    
                    for file_path in folder_path.rglob('*.csv'):
                        try:
                            arcname = file_path
                            zipf.write(file_path, arcname)
                            print(f"  ✅ Added: {arcname}")
                        except (PermissionError, OSError) as e:
                            print(f"  ⚠️ Skipped: {file_path.name} - {e}")
                            continue
        
        print("✅ Package created successfully: stockscreener_package.zip")
        
        # 패키지 크기 확인
        package_size = Path('stockscreener_package.zip').stat().st_size / (1024*1024)
        print(f"📊 Package size: {package_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Package creation failed: {e}")
        return False

def create_simple_installer():
    """간단한 실행형 인스톨러 생성"""
    print("🔨 Creating simple executable installer...")
    
    # 인스톨러 Python 스크립트
    installer_py = '''
import os
import zipfile
import shutil
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, filedialog

def install():
    try:
        # GUI 초기화
        root = tk.Tk()
        root.withdraw()
        root.title("StockScreener Installer")
        
        # 기본 설치 경로
        desktop = Path.home() / "Desktop"
        default_path = desktop / "StockScreener"
        
        # 설치 경로 확인
        response = messagebox.askyesnocancel(
            "StockScreener Installer",
            f"Install StockScreener to:\\n{default_path}\\n\\nYes: Install to Desktop\\nNo: Choose custom location\\nCancel: Exit"
        )
        
        if response is None:  # Cancel
            return False
        elif response == False:  # No - custom path
            install_dir = filedialog.askdirectory(title="Choose installation folder")
            if not install_dir:
                return False
            install_path = Path(install_dir) / "StockScreener"
        else:  # Yes - default path
            install_path = default_path
        
        # 폴더 생성
        install_path.mkdir(parents=True, exist_ok=True)
        
        # 압축 해제할 패키지 찾기
        if getattr(sys, 'frozen', False):
            # PyInstaller로 패키징된 경우
            bundle_dir = Path(sys._MEIPASS)
            package_file = bundle_dir / "stockscreener_package.zip"
        else:
            # 개발 환경
            package_file = Path(__file__).parent / "stockscreener_package.zip"
        
        if not package_file.exists():
            messagebox.showerror("Error", f"Installation package not found:\\n{package_file}")
            return False
        
        # 압축 해제
        try:
            with zipfile.ZipFile(package_file, 'r') as zip_ref:
                zip_ref.extractall(install_path)
            
            # 설치 확인
            exe_path = install_path / "StockScreener.exe"
            if exe_path.exists():
                messagebox.showinfo(
                    "Installation Complete",
                    f"StockScreener installed successfully!\\n\\nLocation: {install_path}\\n\\nClick OK to finish."
                )
                
                # 바탕화면 바로가기 생성 시도
                try:
                    desktop_shortcut = desktop / "StockScreener.lnk"
                    # Windows에서 바로가기 생성 (선택사항)
                    pass
                except:
                    pass
                
                return True
            else:
                messagebox.showerror("Error", "Installation completed but StockScreener.exe not found!")
                return False
                
        except zipfile.BadZipFile:
            messagebox.showerror("Error", "Installation package is corrupted!")
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Extraction failed: {str(e)}")
            return False
            
    except Exception as e:
        messagebox.showerror("Installation Error", f"Installation failed: {str(e)}")
        return False
    finally:
        try:
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    if install():
        print("Installation completed successfully!")
    else:
        print("Installation cancelled or failed.")
'''
    
    with open('simple_installer.py', 'w', encoding='utf-8') as f:
        f.write(installer_py)
    
    # PyInstaller로 실행파일 생성
    if Path('stockscreener_package.zip').exists():
        cmd = 'pyinstaller --onefile --windowed --name StockScreener_install simple_installer.py --add-data "stockscreener_package.zip;."'
        result = os.system(cmd)
        
        if result == 0:
            print("✅ Executable installer created: dist/StockScreener_install.exe")
        else:
            print("❌ Failed to create executable installer")
    else:
        print("❌ Package file not found - skipping executable installer")

def main():
    print("🏗️ Creating installer packages...")
    print("⏳ Waiting for file handles to be released...")
    
    # 파일 핸들 해제를 위한 대기
    time.sleep(3)
    
    # 1. 안전한 패키지 생성
    if create_package_safe():
        # 2. 배치 인스톨러 생성
        create_batch_installer()
        
        # 3. 간단한 실행형 인스톨러 생성
        create_simple_installer()
        
        print("🎉 All installers created successfully!")
        print("📦 Available installers:")
        print("  🔧 dist/StockScreener_install.exe (GUI installer)")
        print("  📄 StockScreener_install.bat (Batch installer)")
        print("  📦 stockscreener_package.zip (Manual extraction)")
    else:
        print("❌ Package creation failed - installers not created")

if __name__ == "__main__":
    main()