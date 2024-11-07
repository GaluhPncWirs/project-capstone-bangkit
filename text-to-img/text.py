from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import os

def create_image_toText(text, img_width=128, img_height=32, font_size=16, font_path="font/Helvetica.ttf"):
    img = Image.new('L', (img_width, img_height), color=255)
    draw_img = ImageDraw.Draw(img)

    font = ImageFont.truetype(font_path, font_size)

    draw_img.text((5,5), text, font=font, fill=0)

    return np.array(img)


texts = ['200 kcal', '5 g sugar', '0.5 g salt', '3 g fat']


os.makedirs("data_sintetik", exist_ok=True)
for i, text in enumerate(texts):
    img_array = create_image_toText(text)
    img = Image.fromarray(img_array)
    img.save(f"data_sintetik/{text.replace(' ', '_')}.png")


data = {'image_path' : [f"data_sintetik/{text.replace(' ', '_')}.png" for text in texts], 'label': texts}

df = pd.DataFrame(data)
df.to_csv("labels.csv", index=False)