echo "📦 Starting Stock Screener build..."

# Step 1: Generate master CSV files
echo "1️⃣ Preparing master CSV files..."
python prepare_for_build.py

# Step 2: Generate icons (optional)
echo "2️⃣ Generating icons..."
python simple_icon.py

# Step 3: PyInstaller build
echo "3️⃣ Creating executable file..."
pyinstaller StockScreener.spec --clean

# Step 4: Completion message
echo "✅ Build completed! Check dist/StockScreener.exe"
echo "📊 Included data:"
echo "Top 100 Korean stocks"
echo "Top 100 US stocks" 
echo "Top 100 Swedish stocks"