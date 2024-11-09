from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import os

def create_image_toText(text, img_width=128, img_height=32, font_size=18, font_path="font/Helvetica.ttf"):
    img = Image.new('L', (img_width, img_height), color=255)
    draw_img = ImageDraw.Draw(img)

    font = ImageFont.truetype(font_path, font_size)

    draw_img.text((5,5), text, font=font, fill=0)

    return np.array(img)


texts = [
    "200 kcal", "5 g sugar", "0.5 g salt", "3 g fat",
    "150 kcal", "10 g protein", "1 g fiber", "4 mg sodium",
    "300 kcal", "15 g carbs", "1.2 g salt", "8 g fat",
    "100 kcal", "3 g sugar", "0.3 g potassium", "2 g fat",
    "250 kcal", "8 g protein", "0.7 g salt", "6 mg iron",
    "180 kcal", "12 g fiber", "1.1 g sodium", "5 g fat",
    "220 kcal", "7 g sugar", "0.4 g calcium", "3.5 g protein",
    "270 kcal", "9 g fat", "0.6 g salt", "4.5 g carbs",
    "230 kcal", "6 g sugar", "1.3 g fiber", "7 g fat",
    "140 kcal", "4 g protein", "0.8 g sodium", "2.5 g fat",
    "160 kcal", "11 g sugar", "1.5 g iron", "6.5 g fiber",
    "210 kcal", "5.5 g carbs", "0.9 g potassium", "4.2 g fat",
    "240 kcal", "13 g fiber", "1.4 g sodium", "7.5 g protein",
    "110 kcal", "2 g sugar", "0.5 g calcium", "1 g fat",
    "280 kcal", "6 g fiber", "1 g salt", "4 g protein",
    "190 kcal", "3 g fat", "0.6 g iron", "5 g carbs",
    "130 kcal", "9 g protein", "0.7 g sodium", "3 g fiber",
    "170 kcal", "12 g sugar", "0.4 g potassium", "5.5 g fat",
    "220 kcal", "5 g carbs", "0.8 g salt", "6 g protein",
    "250 kcal", "4 g sugar", "1 g iron", "3 g fiber",
    "160 kcal", "10 g protein", "0.9 g sodium", "5 g fat",
    "290 kcal", "7 g carbs", "1.2 g calcium", "8 g fiber",
    "120 kcal", "3 g sugar", "0.6 g salt", "4.5 g fat",
    "270 kcal", "8 g fiber", "1 g iron", "6 g protein",
    "140 kcal", "2 g sugar", "0.3 g potassium", "1.5 g fat",
    "210 kcal", "9 g protein", "1.1 g salt", "3 g fiber",
    "180 kcal", "4 g carbs", "0.5 g sodium", "7 g fat",
    "230 kcal", "10 g sugar", "0.6 g calcium", "5 g fiber",
    "250 kcal", "6 g protein", "1.3 g salt", "4 g fat",
    "170 kcal", "3 g carbs", "0.8 g iron", "6.5 g fiber",
    "280 kcal", "5 g fat", "0.4 g sodium", "9 g protein",
    "150 kcal", "12 g sugar", "0.7 g calcium", "2.5 g fiber",
    "130 kcal", "7 g protein", "1 g iron", "4 g carbs",
    "200 kcal", "3 g fiber", "0.9 g sodium", "6 g fat",
    "260 kcal", "8 g sugar", "1.1 g potassium", "5.5 g protein",
    "180 kcal", "5 g carbs", "0.5 g salt", "7 g fat",
    "220 kcal", "6 g protein", "1 g fiber", "4 g sugar",
    "240 kcal", "2 g fat", "0.6 g iron", "8 g carbs",
    "200 kcal", "10 g fiber", "1.4 g calcium", "3 g protein",
    "190 kcal", "4 g sugar", "0.8 g sodium", "5 g fat",
    "270 kcal", "7 g protein", "1 g salt", "4 g fiber",
    "210 kcal", "3 g carbs", "1.1 g potassium", "6 g fat",
    "140 kcal", "5 g protein", "0.4 g iron", "9 g fiber",
    "150 kcal", "11 g sugar", "0.7 g salt", "2.5 g carbs",
    "260 kcal", "4 g fat", "1.3 g sodium", "7 g protein",
    "170 kcal", "9 g fiber", "0.5 g calcium", "5 g carbs",
    "130 kcal", "6 g protein", "0.6 g salt", "3 g fat",
    "200 kcal", "8 g sugar", "0.8 g iron", "7 g fiber",
    "110 kcal", "4 g protein", "1 g sodium", "2 g carbs",
    "240 kcal", "7 g fat", "0.3 g potassium", "5 g sugar",
    "150 kcal", "3 g fiber", "0.9 g salt", "6.5 g protein",
    "220 kcal", "10 g carbs", "1.2 g calcium", "4 g fat",
    "120 kcal", "5 g sugar", "0.5 g sodium", "8 g protein",
    "250 kcal", "4 g fat", "1 g iron", "7.5 g fiber",
    "170 kcal", "2 g protein", "0.6 g salt", "5 g carbs",
    "140 kcal", "6 g sugar", "0.8 g potassium", "3.5 g fat",
    "210 kcal", "9 g fiber", "1 g calcium", "6 g protein",
    "130 kcal", "5 g carbs", "0.4 g sodium", "7 g fat",
    "190 kcal", "3 g protein", "1.1 g salt", "8 g fiber",
    "270 kcal", "10 g fat", "0.7 g iron", "4 g sugar",
    "150 kcal", "8 g fiber", "0.5 g salt", "6 g protein",
    "280 kcal", "4 g sugar", "1 g calcium", "7 g carbs",
    "160 kcal", "5 g fat", "0.9 g potassium", "3 g fiber",
    "240 kcal", "6 g protein", "1.2 g salt", "4 g sugar",
    "180 kcal", "7 g carbs", "0.6 g sodium", "9 g fat",
    "220 kcal", "2 g sugar", "1.3 g iron", "5.5 g fiber",
    "200 kcal", "9 g protein", "0.7 g potassium", "3 g fat",
    "110 kcal", "6 g carbs", "0.4 g salt", "8 g fiber",
    "130 kcal", "8 g protein", "1 g sodium", "5 g fat",
    "250 kcal", "3 g fiber", "1.5 g calcium", "6 g sugar",
    "170 kcal", "7 g fat", "0.6 g salt", "4 g carbs",
    "210 kcal", "10 g protein", "0.8 g potassium", "2 g fiber",
    "190 kcal", "4 g sugar", "1 g iron", "7 g protein",
    "140 kcal", "6 g fiber", "0.5 g sodium", "5.5 g fat",
    "120 kcal", "9 g protein", "1.1 g salt", "3 g carbs",
    "280 kcal", "2 g fat", "0.7 g calcium", "8 g fiber",
    "240 kcal", "5 g sugar", "0.4 g salt", "6 g protein",
    "150 kcal", "10 g fiber", "1 g iron", "4 g fat",
    "260 kcal", "3 g protein", "0.9 g potassium", "7.5 g carbs",
    "200 kcal", "8 g fat", "1.2 g salt", "5 g sugar",
    "180 kcal", "4 g carbs", "0.6 g sodium", "9 g protein",
    "170 kcal", "2 g fiber", "1.3 g iron", "6 g fat",
    "140 kcal", "7 g protein", "0.5 g calcium", "3 g sugar",
    "250 kcal", "6 g fiber", "0.8 g potassium", "5 g fat",
    "160 kcal", "3 g fat", "1 g salt", "9 g carbs"
]



os.makedirs("data_sintetik", exist_ok=True)
for i, text in enumerate(texts):
    img_array = create_image_toText(text)
    img = Image.fromarray(img_array)
    img.save(f"data_sintetik/{text.replace(' ', '-')}.png")


data = {'image_path' : [f"data_sintetik/{text.replace(' ', '-')}.png" for text in texts], 'label': texts}

df = pd.DataFrame(data)
df.to_csv("labels.csv", index=False)