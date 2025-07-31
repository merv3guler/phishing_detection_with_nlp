import pandas as pd
from bs4 import BeautifulSoup

# --- BeautifulSoup ile HTML İçeriği Kontrol Fonksiyonu ---
def html_iceriyor_mu_bs(metin: str) -> bool:
    """
    Verilen metnin HTML etiketleri içerip içermediğini BeautifulSoup kullanarak kontrol eder.
    Parametreler:
        metin (str): Kontrol edilecek metin.
    Dönüş:
        bool: HTML etiketi içeriyorsa True, içermiyorsa False.
    """
    # Gelen metnin string olup olmadığını ve boş olup olmadığını kontrol et
    if not isinstance(metin, str) or not metin.strip():
        return False # String değilse veya boşsa HTML içermiyor kabul et
    soup = BeautifulSoup(metin, "html.parser")
    # Herhangi bir HTML etiketi bulursa (örn: soup.find() ilk etiketi bulur)
    if soup.find(): # soup.find() bir etiket bulursa Tag nesnesi, bulamazsa None döner
        return True  # Etiket bulunduysa
    return False # Etiket bulunamadıysa

# --- Veri Setini Yükleme ---
df = pd.read_csv('data/CEAS_08.csv')

# --- KONTROL EDİLECEK SÜTUNUN ADINI BELİRTİN ---
metin_sutunu_adi = 'body' # E-posta metinlerini içeren sütununuzun adı

print(f"'{metin_sutunu_adi}' sütununda HTML etiketi taraması başlatılıyor...")
html_bulundu_flag = False
bulunan_index = -1
bulunan_metin_ornegi = ""

try:
    for index, row in df.iterrows():
        hucre_metni = row[metin_sutunu_adi]
        # NaN, None veya diğer non-string değerleri kontrol et
        if not isinstance(hucre_metni, str):
            # String olmayan değerleri atla
            continue
        if html_iceriyor_mu_bs(hucre_metni):
            html_bulundu_flag = True
            bulunan_index = index
            bulunan_metin_ornegi = str(hucre_metni)[:100] # Hata mesajı için metinden bir örnek al
            # Hata bulunduğunda döngüden çık ve hatayı fırlat
            raise ValueError(f"HATA: HTML etiketi bulundu! DataFrame indeksi: {bulunan_index}, Sütun: '{metin_sutunu_adi}'.\nEtiket içeren metin örneği: '{bulunan_metin_ornegi}...'")
    # Eğer döngü hatasız tamamlanırsa (hiç HTML etiketi bulunmazsa)
    print(f"Tarama tamamlandı. '{metin_sutunu_adi}' sütununda herhangi bir HTML etiketi bulunamadı. ✅")
except ValueError as e:
    print(e) # Yakalanan ValueError'ı yazdır
except Exception as e:
    print(f"Beklenmedik bir hata oluştu: {e}")