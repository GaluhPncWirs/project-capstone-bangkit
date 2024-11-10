import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

payload = {
    "gizi": "41",
    "parameter": "",
    "nilai": "",
    "satuan": "g",
    "satuan_all": "g",
    "rdJenisProduk": "Minuman",
    "kelompokPangan": "",
    "rdJenisMerk": "Semua",
    "rdSertifikasi": "BPOM",
    "parameter_bdd": "",
    "bdd": ""
}

session = requests.Session()

count = 0
pages = 1
max_link_per_page = 20
all_data = []
all_nama_data = []
produk = []

def loadingAnimation():
    for i in range(8):
        print("Sedang Beralih Ke Page Selanjutnya" + "." * i, end="\r")
        time.sleep(0.8)

while True:
    base_url = f"https://nilaigizi.com/pencarian/pencarian_adv/{pages}/asc"
    url = base_url.format(page=pages)
    response = session.post(url, data=payload)

    if response.status_code != 200:
        print(f"Gagal mengambil halaman {pages}, kode status: {response.status_code}")
        break

    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all('a', href=True)

    if not links:
        print(f"Tidak ada link yang ditemukan di halaman {pages}, menghentikan.")
        break

    for link in links[:max_link_per_page]:
        href = link['href']
        if not href.startswith('http'):
            href = requests.compat.urljoin(url, href)

        page_response = session.get(href)
        if page_response.status_code == 200:
            print(f"data berhasil di scrapping ke-{count + 1}")
            page_soup = BeautifulSoup(page_response.text, 'html.parser')
            data = []
            nama_data = []

            nameProduct = page_soup.find('b').text.strip() if page_soup.find('b') else "Nama Produk Tidak Ditemukan"
            produk.append(nameProduct)
            informasi_gizi = page_soup.find("table")

            if informasi_gizi:
                for row in informasi_gizi.find_all('tr'):
                    find_TR = row.find_all('td')
                    if find_TR:
                        data_gizi = [gizi.text.strip() for gizi in find_TR if gizi.text.strip()][1:-1]
                        data_nama_gizi = [nama_gizi.text.strip() for nama_gizi in find_TR if nama_gizi.text.strip()][:1]

                        if not all_nama_data:  # Hanya menambahkan kolom sekali
                            all_nama_data = ["Nama Produk"] + data_nama_gizi

                        data.append(data_gizi)

                all_data.extend(data)
        else:
            print("Gagal Mengambil Halaman")

        count += 1
        time.sleep(0.8)

    if pages == 1:
        print("proses scrapping data selesai")
        break

    pages += 1
print(f"total data yang di proses ada - {count}")

if len(all_data) > 0:
    df = pd.DataFrame(all_data, columns=all_nama_data)
    df.to_csv('data_nilai_gizi.csv', index=False)
    print("Data berhasil disimpan ke 'data_nilai_gizi.csv'")
else:
    print("Tidak ada data yang ditemukan.")
