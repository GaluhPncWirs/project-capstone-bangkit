from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import os

def create_image_toText(text, img_width=250, img_height=35, font_size=18, font_path="font/Helvetica.ttf"):
    img = Image.new('L', (img_width, img_height), color=255)
    draw_img = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    draw_img.text((5, 5), text, font=font, fill=0)
    return np.array(img)

informasi_nilai_gizi = pd.read_csv('data_ing.csv')
nama_kandungan = informasi_nilai_gizi['nama kandungan'].tolist()
jumlah_kandungan = informasi_nilai_gizi['jumlah kandungan'].tolist()
print(jumlah_kandungan)

# texts = [f"{nama} {jumlah}" for nama, jumlah in zip(nama_kandungan, jumlah_kandungan)]

# output_folder = "data_for_train_model"
# os.makedirs(output_folder, exist_ok=True)

# image_paths = []
# for i, text in enumerate(texts):
#     try:
#         img_array = create_image_toText(text)
#         img = Image.fromarray(img_array)
#         image_path = os.path.join(output_folder, f"text_{i}.png")
#         img.save(image_path)
#         image_paths.append(image_path)
#     except Exception as e:
#         print(f"Error processing text '{text}': {e}")

# data = {'image_path': image_paths, 'label': texts}
# df = pd.DataFrame(data)
# df.to_csv("data_labels.csv", index=False)

# print("Proses selesai! Data gambar dan label berhasil disimpan")

