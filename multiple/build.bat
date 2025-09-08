echo "📦 Starting Stock Screener build..."

# Step 1: Prepare local CSV files for build
echo "1️⃣ Preparing local CSV files for build..."
echo "📂 Checking existing CSV files..."

# 로컬 폴더 존재 여부 확인
if exist "stock_data" (
    echo "✅ stock_data 폴더 발견"
    dir /b stock_data\*.csv 2>nul && echo "  📊 stock_data CSV 파일들 확인됨" || echo "  ⚠️ stock_data에 CSV 파일 없음"
) else (
    echo "❌ stock_data 폴더 없음"
    mkdir stock_data
    echo "📁 stock_data 폴더 생성됨"
)

if exist "master_csv" (
    echo "✅ master_csv 폴더 발견" 
    dir /b master_csv\*.csv 2>nul && echo "  📊 master_csv CSV 파일들 확인됨" || echo "  ⚠️ master_csv에 CSV 파일 없음"
) else (
    echo "❌ master_csv 폴더 없음"
    mkdir master_csv
    echo "📁 master_csv 폴더 생성됨"
)

echo "✅ CSV 파일 준비 완료 - 기존 로컬 파일들을 포함합니다"

# Step 2: Generate icons (optional)
echo "2️⃣ Generating icons..."
python simple_icon.py

# Step 3: PyInstaller build with CSV data included (onedir mode)
echo "3️⃣ Creating executable with CSV data..."
pyinstaller StockScreener.spec --clean

# Step 4: Completion message
echo "✅ Build completed! Check dist/StockScreener/ folder"
echo "📊 Included data:"
echo "  📂 Local stock_data folder (current stock_data folder included)"
echo "  📂 Local master_csv folder (current master_csv folder included)"
echo "🚀 Executable packaged with CSV data files!"