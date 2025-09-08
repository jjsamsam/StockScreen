
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
            f"Install StockScreener to:\n{default_path}\n\nYes: Install to Desktop\nNo: Choose custom location\nCancel: Exit"
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
            messagebox.showerror("Error", f"Installation package not found:\n{package_file}")
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
                    f"StockScreener installed successfully!\n\nLocation: {install_path}\n\nClick OK to finish."
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
