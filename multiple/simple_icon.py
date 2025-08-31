from PIL import Image, ImageDraw

# 64x64 크기의 간단한 아이콘 생성
img = Image.new('RGB', (64, 64), color='#4CAF50')  # 초록색 배경
draw = ImageDraw.Draw(img)

# 간단한 차트 모양
draw.rectangle([10, 40, 15, 50], fill='white')
draw.rectangle([20, 30, 25, 50], fill='white')
draw.rectangle([30, 20, 35, 50], fill='white')
draw.rectangle([40, 35, 45, 50], fill='white')

# 아이콘 저장
img.save('stock_icon.ico', format='ICO')
print("아이콘 생성 완료: stock_icon.ico")