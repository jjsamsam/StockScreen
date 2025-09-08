# -*- mode: python ; coding: utf-8 -*-

# Data files configuration (include local CSV files)
datas = []

# Include all CSV files from stock_data folder
import os
if os.path.exists('stock_data'):
    for file in os.listdir('stock_data'):
        if file.endswith('.csv'):
            datas.append(('stock_data/' + file, 'stock_data'))
            print(f"✅ Including: stock_data/{file}")

# Include all CSV files from master_csv folder  
if os.path.exists('master_csv'):
    for file in os.listdir('master_csv'):
        if file.endswith('.csv'):
            datas.append(('master_csv/' + file, 'master_csv'))
            print(f"✅ Including: master_csv/{file}")

print(f"📊 Total {len(datas)} data files will be included")

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,  # CSV files included here
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# Changed to onedir mode (COLLECT instead of onefile EXE)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='StockScreener',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# COLLECT creates the folder structure
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='StockScreener',
)