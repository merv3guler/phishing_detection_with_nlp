import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from textstat import flesch_kincaid_grade, gunning_fog # Okunabilirlik metrikleri için
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # Duygu analizi için

# Gerekli NLTK veri paketlerini indir (yüklü değil ise)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt')
try:
    WordNetLemmatizer().lemmatize("tests")
except LookupError:
    nltk.download('wordnet')
try:
    nltk.pos_tag(["test"])
except LookupError:
    # NLTK'nın standart averaged_perceptron_tagger yerine, spesifik çalışma ortamımda averaged_perceptron_tagger_eng paket kimliğinin POS etiketleyici için işlevsel olduğu tespit edilmiştir.
    #nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')

class Preprocessor:
    def __init__(self, language='english'):
        """
        Ön işleme (preprocessing) sınıfını başlatır.
        Bu metod, phishing tespiti uygulaması için metin ön işleme işlemlerinde gerekli bileşenleri hazırlar.
        Dil ayarlarını yapar, ilgili dilin stopword listesini yükler ve kelime köklerine indirgeme (lemmatization) için WordNetLemmatizer örneği oluşturur.
        Ayrıca duygu analizi için bir analizör da başlatılır.
        Parametreler:
            language (str): Stopword ve lemmatization işlemlerinde kullanılacak dil. Varsayılan: 'english'.
        Özellikler:
            language (str): Ön işlemede kullanılan dil kodu.
            stop_words (set): Seçilen dile göre stopword kümesi.
            lemmatizer (WordNetLemmatizer): Kelimeleri kök haline getiren örnek.
            sentiment_analyzer (SentimentIntensityAnalyzer): Duygu analizi için örnek.
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer() # Duygu analizi için

    def clean_text(self, text):
        """
        Girdi metnini phishing tespiti için temizler ve standartlaştırır.
        Bu metod aşağıdaki işlemleri uygular:
        1. String olmayan girdileri string'e çevirir (bytes, liste, dict, obje vs.).
        2. HTML etiketlerini kaldırır.
        3. URL'leri <URL> etiketiyle değiştirir.
        4. E-posta adreslerini <EMAIL> etiketiyle değiştirir.
        5. Para birimlerini/miktarlarını <MONEY> etiketiyle değiştirir.
        6. IP adreslerini <IPADDRESS> etiketiyle değiştirir.
        7. Metni küçük harfe çevirir.
        8. Sayıları <NUMBER> etiketiyle değiştirir.
        9. Noktalama ve özel karakterleri kaldırır (etiketler hariç).
        10. Fazla boşlukları temizler.
        Parametreler:
            text (Any): Girdi metni (string veya string'e çevrilebilen herhangi bir tip).
        Dönüş:
            str: Temizlenmiş ve standartlaştırılmış metin.
        """
        if not isinstance(text, str): # Eğer text string değilse
            # Diğer tipleri string'e çevirme
            if isinstance(text, bytes): # Bytes tipinde ise
                text = text.decode('utf-8', errors='ignore')
            elif isinstance(text, (list, tuple)): # Liste veya demet ise
                text = ', '.join(map(str, text))
            elif isinstance(text, dict): # Sözlükler için
                text = ', '.join(map(str, text.values()))
            elif hasattr(text, '__dict__'): # objeler için
                text = ', '.join(map(str, text.__dict__.values()))
            else: # Diğer tipler için varsayılan string dönüşümü
                text = str(text)

        # 1. HTML etiketlerini kaldır
        text = re.sub(r'<[^>]+>', ' ', text)

        # 2. URL'leri <URL> etiketiyle değiştir
        text = re.sub(r'http\S+|www\S+|https\S+', '<URL>', text, flags=re.MULTILINE)

        # 3. E-posta adreslerini <EMAIL> etiketiyle değiştir
        text = re.sub(r'\S*@\S*\s?', '<EMAIL>', text)

        # 4. Para birimlerini/miktarlarını <MONEY> etiketiyle değiştir
        text = re.sub(r'\$\d+(?:\.\d+)?|\d+\s*(?:usd|eur|gbp|dollars|pounds|euros)', '<MONEY>', text, flags=re.IGNORECASE)

        # 5. IP adreslerini <IPADDRESS> etiketiyle değiştir
        text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IPADDRESS>', text)

        # 6. Küçük harfe çevir
        text = text.lower()

        # 7. Sayıları kaldır veya <NUMBER> etiketiyle değiştir
        text = re.sub(r'\d+', '<NUMBER>', text) # Etiketle
        # text = re.sub(r'\d+', ' ', text) # Kaldır (Boşlukla değiştirerek kelimelerin birleşmesini önle)

        # 8. Noktalama işaretlerini ve özel karakterleri kaldır (kelime olmayanları boşlukla değiştir)
        text = re.sub(r'[^\w\s<>@-]', ' ', text) # Etiketleri koru (<URL>, <EMAIL> vs.)
        text = re.sub(r'\s+', ' ', text).strip() # Fazla boşlukları temizle

        return text

    def get_pos_tags(self, text_tokens):
        """
        Verilen token listesi için sözcük türü (POS) etiketlerini döndürür.
        Parametreler:
            text_tokens (List[str]): POS etiketi üretilecek token listesi.
        Dönüş:
            List[str]: Girdi listesindeki her token için POS etiketi.
        """
        return [tag for word, tag in pos_tag(text_tokens)]

    def get_readability_scores(self, text):
        """
        Verilen metin için okunabilirlik skorlarını hesaplar.
        Bu fonksiyon iki okunabilirlik metriği hesaplar:
            - Flesch-Kincaid seviye skoru
            - Gunning Fog indeksi
        Not:
            Çok kısa veya aşırı işlenmiş metinlerde bu skorlar güvenilir olmayabilir. Ham metin üzerinde analiz önerilir.
        Parametreler:
            text (str): Okunabilirlik skorları hesaplanacak metin.
        Dönüş:
            dict: Aşağıdaki anahtarları içeren bir sözlük:
                - 'flesch_kincaid_grade' (float): Flesch-Kincaid seviye skoru, hata olursa -1.
                - 'gunning_fog' (float): Gunning Fog indeksi, hata olursa -1.
        """
        # Bu fonksiyonların çok kısa veya işlenmiş metinlerde iyi çalışmayabileceğine dikkat edin.
        # Ham metin üzerinde kullanmak daha iyi bir fikirdir.
        try:
            fk_grade = flesch_kincaid_grade(text)
            gf_index = gunning_fog(text)
            return {'flesch_kincaid_grade': fk_grade, 'gunning_fog': gf_index}
        except: # Hataları yakala, özellikle çok kısa metinler için
            return {'flesch_kincaid_grade': -1, 'gunning_fog': -1}

    def get_sentiment_scores(self, text):
        """
        Verilen metin için duygu analiz skorlarını döndürür.

        Parametreler:
            text (str): Analiz edilecek metin.
        Dönüş:
            dict: Negatif, nötr, pozitif ve bileşik (compound) gibi duygu skorlarını içeren sözlük.
        """        
        return self.sentiment_analyzer.polarity_scores(text)

    def get_wordnet_pos(self, tag):
        """
        Verilen POS etiketi için WordNet lemmatizer'ın beklediği etikete dönüştürme işlemini yapar.

        Bu metod, genel bir POS etiketini (genellikle bir POS etiketleyiciden alınan) WordNet lemmatizer'ın beklediği formata eşler:
            - 'J' ile başlıyorsa sıfat ('a')
            - 'V' ile başlıyorsa fiil ('v')
            - 'N' ile başlıyorsa isim ('n')
            - 'R' ile başlıyorsa zarf ('r')
        Diğer tüm durumlarda varsayılan olarak 'n' döner.

        Parametreler:
            tag (str): Dönüştürülecek POS etiketi.

        Dönüş:
            str: WordNet POS etiketi ('a', 'v', 'n', 'r').
        """
        if tag.startswith('J'):
            return 'a'
        elif tag.startswith('V'):
            return 'v'
        elif tag.startswith('N'):
            return 'n'
        elif tag.startswith('R'):
            return 'r'
        else:
            return 'n'

    def preprocess(self, text, include_pos=False, for_vectorization=False):
        """
        Girdi metni üzerinde temizlik, tokenizasyon ve kök bulma (lemmatization) işlemlerini uygular.
        Parametreler:
            text (str): Ön işleme tabi tutulacak metin.
            include_pos (bool): Çıktıda sözcük türü (POS) etiketleri de bulunsun mu?
            for_vectorization (bool): Vektörleştirme için mi kullanılacak?
        Dönüş:
            dict: Ön işlenmiş metin, token listesi, temizlenmiş metin, okunabilirlik ve duygu skorları gibi bilgiler.
        """
        # Metni temizle
        cleaned_text = self.clean_text(text)

        # Tokenize et
        words = word_tokenize(cleaned_text)

        # Lemmatization ve stopword temizleme
        pos_tags = pos_tag(words) # POS etiketlerini al
        processed_words = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag))
            for word, tag in pos_tags
            if word not in self.stop_words and len(word) > 1
        ]

        # Okunabilirlik skorlarını hesapla
        readability = self.get_readability_scores(cleaned_text)

        # Duygu analizi ve for_vectorization için birleştirme
        combined_text = ' '.join(processed_words)
        
        # Duygu skorlarını hesapla
        sentiment = self.get_sentiment_scores(combined_text)

        # if include_pos:
            # POS etiketlerini özellik olarak nasıl kullanacağınıza karar vermelisiniz.
            # Örn: ' '.join(pos_tags) veya frekansları vs.
            # combined_text += " <POS> " + ' '.join(pos_tags)

        return {
            "text": combined_text if for_vectorization else processed_words,
            "cleaned_text": cleaned_text,
            "tokens": processed_words,
            "pos_tags": self.get_pos_tags(processed_words) if include_pos else None,
            "readability": readability,
            "sentiment": sentiment
        }