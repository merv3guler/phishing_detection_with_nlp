# Phishing E-posta Tespiti: NLP ve Makine Öğrenmesi Tabanlı Analiz

Bu proje, Doğal Dil İşleme (DDİ) ve Makine Öğrenmesi (MÖ) tekniklerini kullanarak phishing (oltalama) e-postalarını tespit etmeyi amaçlamaktadır. Proje, e-posta içeriklerini analiz ederek e-postaları "phishing" veya "meşru" olarak sınıflandırır.

## Kurulum ve Hazırlık

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1.  **Projeyi Klonlayın (veya İndirin):**
    Eğer bir Git deposu olarak mevcutsa:
    ```bash
    git clone [https://github.com/merv3guler/phishing_detection_with_nlp.git](https://github.com/merv3guler/phishing_detection_with_nlp.git)
    cd phishing_detection_with_nlp
    ```
    Değilse, proje dosyalarını bir dizine kopyalayın.

2.  **Python Ortamı Oluşturun (Tavsiye Edilir):**
    Proje **Python 3.12.2** ile test edilmiştir. Diğer Python 3.x versiyonlarıyla da uyumlu olması beklenir ancak 3.12.2 önerilir.
    Proje bağımlılıklarını sistem genelindeki Python kurulumunuzdan izole etmek için bir sanal ortam oluşturmanız önerilir.
    ```bash
    python -m venv venv
    ```
    Sanal ortamı aktifleştirin:
    * Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    * macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Bağımlılıkları Yükleyin:**
    Projenin çalışması için gerekli olan kütüphaneleri `requirements.txt` dosyasından yükleyin.
    ```bash
    pip install -r requirements.txt
    ```
    **Not:** `requirements.txt` dosyasında listelenen bazı kütüphaneler NLTK veri paketleri (`punkt`, `stopwords`, `wordnet`, `averaged_perceptron_tagger`) için çeşitli indirmeler gerektirebilir. `preprocessing.py` içerisinde bu indirmeleri otomatikleştiren kodlar bulunmaktadır. İlk çalıştırmada internet bağlantısı gerekebilir.

4.  **Model Dosyalarını İndirin/Hazırlayın (Gerekirse):**
    Eğitilmiş modeller (`.joblib` dosyaları) ve vektörleştiriciler `models/` dizininde bulunmalıdır. Eğer bu dosyalar projeyle birlikte gelmediyse veya kendi modellerinizi eğitecekseniz `model_training.py` betiğini çalıştırmanız gerekebilir.
    `models/` klasörü yoksa oluşturun.

## Kullanım

Proje, `app.py` betiği aracılığıyla tek bir e-postanın phishing olup olmadığını tahmin etmek için kullanılabilir. Betik, `.eml` veya `.msg` formatındaki e-posta dosyalarını veya doğrudan metin girdisini kabul eder.

**Temel Kullanım Komutları:**

1.  **.eml veya .msg Dosyasından Tahmin Yapma:**
    ```bash
    python app.py --file "path/to/your/email.eml"
    ```
    veya
    ```bash
    python app.py --file "path/to/your/email.msg"
    ```

2.  **Doğrudan Metin Girdisi ile Tahmin Yapma:**
    ```bash
    python app.py --text "Bu bir örnek e-posta metnidir..."
    ```

3.  **Ayrıntılı Çıktı (Verbose Mode):**
    İşleme adımları ve çıkarılan metin hakkında daha fazla bilgi almak için `--verbose` bayrağını kullanın:
    ```bash
    python app.py --file "path/to/your/email.eml" --verbose
    ```
    veya
    ```bash
    python app.py --text "Bu bir örnek e-posta metnidir..." --verbose
    ```

**Çıktı Örneği:**

```
--- Tahmin Sonucu (path/to/your/email.eml dosyasından) ---
Durum: Phishing (Model Tahmini: 1)
```

Veya :

```
--- Tahmin Sonucu (path/to/your/email.eml dosyasından) ---
Durum: Meşru (Model Tahmini: 0)
```

## Model Eğitimi ve Değerlendirme

Modelleri yeniden eğitmek, farklı konfigürasyonları denemek veya performanslarını değerlendirmek için `model_training.py` betiği kullanılır.

```bash
python model_training.py
```

Bu betik:
1.  `data/CEAS_08.csv` veri setini yükler.
2.  Veriyi ön işler (`preprocessing.py` kullanarak) ve `results/CEAS_08_preprocessed.parquet` olarak kaydeder.
3.  Farklı öznitelik çıkarma yöntemleri (TF-IDF, Word Embeddings) ve makine öğrenmesi algoritmaları (Linear SVM, Random Forest, Naive Bayes) kullanarak çeşitli modelleri eğitir ve değerlendirir.
4.  Eğitim ve test metriklerini (Accuracy, Precision, Recall, F1-Score, ROC AUC) hesaplar.
5.  En iyi performans gösteren modelleri ve ilgili vektörleştiricileri/ölçekleyicileri `models/` dizinine kaydeder.
6.  Karşılaştırmalı performans sonuçlarını `results/model_performance_comparison.csv` dosyasına yazar.
7.  Karışıklık matrislerini `results/confusion_matrices/` dizinine kaydeder.
8.  ROC eğrilerini `results/roc_curves/` dizinine kaydeder.
9.  İstatistiksel anlamlılık testleri için McNemar testini uygular ve sonuçları `results/mcnemar_test_karsilastirma_sonuclari.csv` dosyasına yazar.

## Veri Seti

Bu projede kullanılan ana veri seti **CEAS_08**'dir. Bu veri seti, çeşitli e-posta kaynaklarından derlenmiş ve özellikle phishing/meşru e-posta sınıflandırması için hazırlanmıştır. Veri seti, e-posta gövdesi, etiketleri ve URL'ler gibi çeşitli alanları içerir.

## Yöntemler ve Teknikler

Bu projede kullanılan temel yöntemler ve teknikler şunlardır:

* **Veri Ön İşleme:**
    * HTML etiketlerinin temizlenmesi
    * URL, e-posta adresi ve özel karakterlerin işlenmesi/kaldırılması
    * Küçük harfe dönüştürme
    * Noktalama işaretlerinin kaldırılması
    * Sayıların işlenmesi
    * Stopword (etkisiz kelimeler) çıkarımı
    * Tokenizasyon (kelimelere ayırma)
    * Lemmatizasyon (kelime köklerini bulma)
    * Dilbilgisi ve yazım denetimi

* **Öznitelik Çıkarımı (Feature Extraction):**
    * **TF-IDF (Term Frequency-Inverse Document Frequency):** Metin verilerini sayısal vektörlere dönüştürmek için kullanılır. Kelimelerin belge içindeki ve tüm koleksiyondaki önemini tartar.
    * **Word Embeddings (Kelime Gömme - Word2Vec):** Kelimeleri yoğun vektör temsillerine dönüştürerek anlamsal ilişkilerini yakalar. Bu projede `gensim` kütüphanesi ile Word2Vec modeli kullanılmıştır.
    * **Ek Metinsel ve Yapısal Öznitelikler:**
        * Kelime sayısı, karakter sayısı, cümle sayısı
        * Ortalama kelime uzunluğu
        * URL, e-posta adresi, telefon numarası varlığı/sayısı
        * Büyük harf oranı
        * Noktalama işareti sayısı
        * Okunabilirlik skorları (örneğin Flesch-Kincaid)
        * Dilbilgisi ve yazım hatası sayısı

* **Makine Öğrenmesi Modelleri:**
    * **Linear Support Vector Machine (Linear SVM):** Özellikle yüksek boyutlu seyrek verilerde (TF-IDF gibi) etkili olan bir sınıflandırma algoritmasıdır.
    * **Random Forest:** Birden fazla karar ağacını birleştirerek daha güçlü ve stabil tahminler yapan bir topluluk öğrenme yöntemidir.
    * **Naive Bayes (MultinomialNB ve GaussianNB):** Olasılık temelli, basit ve hızlı sınıflandırıcılardır. MultinomialNB genellikle metin sınıflandırmada TF-IDF gibi özelliklerle, GaussianNB ise sürekli verilerle (kelime gömme ortalamaları gibi) kullanılır.

* **Model Değerlendirme:**
    * **Metrikler:** Doğruluk (Accuracy), Kesinlik (Precision), Duyarlılık (Recall), F1-Skoru, ROC AUC (Receiver Operating Characteristic - Area Under Curve).
    * **Karışıklık Matrisi (Confusion Matrix):** Modelin doğru ve yanlış tahminlerini sınıflara göre gösterir.
    * **ROC Eğrisi:** Sınıflandırıcının performansını farklı eşik değerlerinde görselleştirir.
    * **McNemar Testi:** İki farklı modelin tahmin hataları arasında istatistiksel olarak anlamlı bir fark olup olmadığını belirlemek için kullanılır.

## Uyumluluk

* **Python Sürümü:** Proje, **Python 3.12.2** sürümü ile test edilmiş ve bu sürümle uyumlu olduğu doğrulanmıştır. Diğer Python 3.x sürümleriyle de çalışması beklenmekle birlikte, en iyi sonuçlar için belirtilen sürümün kullanılması tavsiye edilir.
* **İşletim Sistemi:** Betikler platform bağımsız olacak şekilde geliştirilmiştir ve Windows, macOS ve Linux işletim sistemlerinde çalıştırılabilir olmalıdır.

## Projeye Omuz Vermek mi? Harika Fikir!

Bu projeye katkıda bulunmak isterseniz ne mutlu bana! Beğendiyseniz GitHub'da bir yıldızınızı ⭐ alırım!

Aklınıza takılan bir hata, parlak bir özellik fikri ya da bir iyileştirme öneriniz varsa, hiç çekinmeden bir `Issue` açabilir veya doğrudan `Pull Request` gönderebilirsiniz. Sürece dahil olmak isterseniz izlemeniz gereken yol oldukça basit:

1.  **Fork'la:** Önce bu repoyu kendi hesabınıza bir 'Fork'layın.
2.  **Branch'ini Oluştur:** `feature/harika-fikrim` gibi havalı bir isimle kendi branch'inizi oluşturun.
3.  **Değişikliğini Yap & Commit'le:** Sihrinizi konuşturun ve kodunuzu anlamlı bir mesajla commit'leyin. (`git commit -m 'Şu harika özelliği ekledim'`)
4.  **Push'la:** Kendi branch'inizi `origin`'e push'layın. (`git push origin feature/harika-fikrim`)
5.  **Pull Request Aç:** Son olarak, değişikliklerinizi anlatan bir Pull Request açın, gerisini beraber hallederiz!

## Lisans

MIT License

## İletişim

Merve Güler - [mervecap@icloud.com](mailto://mervecap@icloud.com)