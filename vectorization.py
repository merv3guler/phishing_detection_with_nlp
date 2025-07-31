# vectorization.py
import os
import re
import joblib
import tldextract
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText
from spellchecker import SpellChecker
from scipy.sparse import hstack

# --- Anahtar Kelime Listeleri ---
# Aşağıdaki anahtar kelime listeleri, e-posta metinlerinde phishing tespiti için çeşitli psikolojik ve içeriksel ipuçlarını yakalamak amacıyla kullanılır.
# Her bir liste, ilgili temadaki (aciliyet, tehdit, ödül, otorite, kişisel bilgi, URL) anahtar kelimeleri içerir.
URGENCY_KEYWORDS = [
    # Zaman Kısıtlaması ve Son Tarihler
    "now",
    "today",
    "immediately",
    "instant",
    "urgent",
    "hurry",
    "quick",
    "fast",
    "expedite",
    "soon",
    "before it's too late",
    "final",
    "last chance",
    "last minute",
    "deadline",
    "expires",
    "ending soon",
    "limited time",
    "running out",
    "don't delay",
    "act now",
    "act fast",
    "closing",
    "valid until",
    "only hours left",
    "today only",
    "one day only",
    "by midnight",
    "by end of day",
    "this week only",
    "ends tonight",
    "offer ends",
    "final call",
    "countdown",
    "approaching deadline",
    "don't wait",
    "without delay",
    "promptly",
    "asap", # as soon as possible
    "at once",

    # Sınırlı Stok ve Erişim
    "limited",
    "only",
    "few",
    "scarce",
    "exclusive",
    "rare",
    "while supplies last",
    "limited stock",
    "limited availability",
    "limited edition",
    "selling fast",
    "almost gone",
    "only few left",
    "get it before it's gone",
    "stocks running low",
    "once in a lifetime",
    "not available long",
    "subject to availability",
    "first come first served",
    "inventory low",
    "disappearing",

    # Acil Eylem ve Önem
    "important",
    "critical",
    "alert",
    "warning",
    "action required",
    "immediate action",
    "respond now",
    "verify",
    "confirm",
    "update",
    "secure",
    "protect",
    "don't miss out",
    "take action",
    "do it now",
    "click here",
    "download now",
    "register now",
    "login",
    "access",
    "essential",
    "vital",
    "mandatory",
    "must",
    "required",
    "need to",
    "priority",
    "sign up",
    "subscribe",
    "claim your",
    "get your",
    "do not ignore",
    "attention",
    "notice",

    # Olumsuz Sonuçlar ve Tehditler
    "avoid",
    "risk",
    "expired",
    "suspended",
    "deactivated",
    "failure",
    "problem",
    "issue",
    "miss out",
    "lose",
    "forfeit",
    "consequence",
    "unauthorized",
    "security alert",
    "breach",
    "penalty",
    "danger",
    "threat",
    "vulnerable",
    "compromised",
    "locked",
    "disabled",
    "payment overdue",
    "account will be closed",
    "stop",
    "prevent",

    # Yoğunluk ve Hız
    "flash", # örn., flash sale
    "instantaneous",
    "rapid",
    "swift",
    "speed",
    "blitz",
    "rush",
    "breaking", # örn., breaking news
    "hot", # örn., hot offer
    "special", # örn., special offer

    # Yeni ve Yenilikçi
    "secret",
    "insider",
    "revealed",
    "unlock",
    "bonus",
    "free", # Genellikle sınırlı süreli teklifler ile birleştirilir
    "guaranteed", # Sınırlı bir teklife yanıt verme baskısı ekleyebilir
    "proven",
    "new",
    "introducing",
    "be the first",
    "early bird"
]
THREAT_KEYWORDS = [
    # Güvenlik & Hesap İhlali
    "hack",
    "hacked",
    "compromised",
    "breach",
    "security breach",
    "data breach",
    "unauthorized access",
    "suspicious activity",
    "malware",
    "virus",
    "spyware",
    "ransomware",
    "trojan",
    "keylogger",
    "phishing",
    "identity theft",
    "fraud",
    "scam",
    "exploit",
    "vulnerability",
    "vulnerable",
    "attack",
    "cyberattack",
    "threat detected",
    "account locked",
    "account suspended",
    "password compromised",
    "login attempt",
    "unusual sign-in",
    "security alert",
    "warning",
    "alert",
    "immediate action required", # Acil eylem gerektirir
    "risk of data loss",
    "exposed",
    "leaked",
    "credentials",
    "verification failed",
    "authentication error",

    # Yasal & Yönetsel Tehditler
    "legal action",
    "lawsuit",
    "court",
    "police",
    "FBI",
    "arrest",
    "warrant",
    "subpoena",
    "investigation",
    "illegal",
    "unlawful",
    "penalty",
    "fine",
    "sanction",
    "government",
    "federal",
    "compliance",
    "non-compliance",
    "violation",
    "infringement",
    "prosecution",

    # Finansal Tehditler
    "payment overdue",
    "outstanding balance",
    "debt",
    "collection",
    "financial loss",
    "funds",
    "bank account",
    "credit card",
    "block transaction",
    "freeze account",
    "unpaid",
    "invoice", # Genellikle ödeme talebi ile ilişkilidir
    "fee",
    "charge",

    # Şantaj & Zorbalık
    "demand",
    "ultimatum",
    "consequences",
    "failure to comply",
    "you must",
    "required",
    "compelled",
    "expose", # Tehdit veya şantaj bağlamında kullanılır
    "reveal",
    "publish",
    "blackmail",
    "extortion",
    "monitor",
    "surveillance",
    "tracked",
    "beware",
    "danger",
    "dangerous",
    "harm",
    "serious",
    "critical", # Genellikle acil durum veya tehdit bağlamında kullanılır
    "imminent",
    "stop",
    "cease and desist", # Yasal bir tehdit olarak kullanılır
    "notification of", # Genellikle olumsuz bir olay/tehditten önce gelir

    # Genel Olumsuz Sonuçlar & Uyarılar
    "problem",
    "issue",
    "error",
    "failure",
    "loss",
    "damage",
    "terminate",
    "termination",
    "deactivate",
    "closure",
    "disruption",
    "restricted",
    " flagged",
    "violated terms",
    "against policy"
]
REWARD_KEYWORDS = [
    # Ödül & Kazanç
    "prize",
    "winner",
    "won",
    "congratulations",
    "claim your prize",
    "you have won",
    "selected",
    "chosen",
    "reward",
    "rewards program",
    "points",
    "bonus",
    "gift",
    "free",
    "free gift",
    "free trial",
    "complimentary",
    "giveaway",
    "sweepstakes",
    "lottery",
    "jackpot",
    "cash prize",
    "voucher",
    "coupon",
    "discount",
    "special offer",
    "exclusive offer",
    "deal",
    "promotion",
    "promo code",
    "save",
    "savings",
    "earn",
    "get paid",
    "cash back",
    "refund",
    "credit",
    "bonus points",
    "extra",
    "benefit",
    "opportunity",
    "chance to win",
    "lucky",
    "fortune",
    "windfall",
    "grant",
    "scholarship", # Belirli dolandırıcılık türlerinde bir tuzak olabilir
    "inheritance", # Klasik dolandırıcılık tuzağı
    "compensation",
    "reimbursement",
    "exclusive access",
    "upgrade",
    "free upgrade",
    "no cost",
    "zero cost",
    "on the house",
    "gift card",
    "e-gift",
    "special promotion",
    "member benefits",
    "loyalty reward",
    "VIP",
    "premium",
    "limited-time offer", # Genellikle acil durumlarla örtüşse de, genellikle ödüller için kullanılır
    "instant win",
    "guaranteed prize",
    "unlock", # örn., ödülünüzü açın
    "redeem",
    "eligible",
    "entitled",
    "celebrate",
    "thank you gift",
    "appreciation",
    "token of gratitude",
    "win big"
]
AUTHORITY_KEYWORDS = [
    # Otorite & Resmi Kurumlar
    "admin",
    "administrator",
    "official",
    "support",
    "security department",
    "IT department",
    "human resources",
    "billing department",
    "customer service",
    "bank",
    "federal",
    "government",
    "police",
    "IRS",
    "CEO",
    "manager",
    "director",
    "executive",
    "compliance",
    "verification team",
    "account services",
    "legal department",
    "attorney",
    "lawyer",
    "court",
    "notice",
    "alert",
    "important notification",
    "system administrator",
    "webmaster",
    "postmaster",
    "helpdesk",
    "service provider",
]
PERSONAL_INFO_KEYWORDS = [
    # Kişisel Bilgi Talepleri
    "password",
    "username",
    "login",
    "account number",
    "credit card",
    "social security",
    "SSN",
    "bank details",
    "pin code",
    "security question",
    "mother's maiden name",
    "date of birth",
    "full name",
    "address",
    "phone number",
    "verify your account",
    "confirm your identity",
    "update your details",
    "secure your account",
    "click here to login",
    "access your account"
]
URL_PHISHING_KEYWORDS = [
    # URL'de Sıkça Kullanılan Phishing Anahtar Kelimeleri
    "login",
    "verify",
    "account",
    "secure",
    "update",
    "confirm",
    "password",
    "signin",
    "banking",
    "ebay",
    "paypal",
    "appleid",
    "support",
    "service",
    "webscr",
    "cmd",
    "token",
    "recovery",
    "reset",
    "activity",
    "admin",
    "administrator",
    "alert",
    "authentication",
    "authorize",
    "backup",
    "bill",
    "card",
    "center",
    "client",
    "credentials",
    "customer",
    "details",
    "device",
    "document",
    "download",
    "form",
    "fraud",
    "help",
    "identity",
    "information",
    "invoice",
    "issue",
    "key",
    "limited",
    "logon",
    "mail",
    "manage",
    "message",
    "notification",
    "online",
    "order",
    "payment",
    "portal",
    "profile",
    "protection",
    "recent",
    "refund",
    "register",
    "report",
    "restore",
    "review",
    "safe",
    "security",
    "serve",
    "session",
    "settings",
    "statement",
    "submit",
    "suspended",
    "transaction",
    "unlock",
    "unusual",
    "user",
    "validate",
    "verification",
    "warning",
    "wallet",
    "weblogin",
    "webmail",
    "amazon",
    "aol",
    "alibaba",
    "bankofamerica",
    "barclays",
    "bitcoin",
    "blockchain",
    "chase",
    "citibank",
    "coinbase",
    "docusign",
    "dropbox",
    "facebook",
    "fedex",
    "gmail",
    "godaddy",
    "google",
    "hsbc",
    "icloud",
    "instagram",
    "irs",
    "linkedin",
    "lloyds",
    "microsoft",
    "myaccount",
    "navyfederal",
    "netflix",
    "office365",
    "outlook",
    "santander",
    "skype",
    "smtp",
    "sparkasse",
    "spotify",
    "steam",
    "suntrust",
    "tdbank",
    "twitter",
    "ups",
    "usps",
    "wellsfargo",
    "whatsapp",
    "windows",
    "yahoo",
    "youtube",
    "zoom",
    ".exe",
    ".js",
    ".php",
    ".asp",
    ".html",
    ".htm",
    "cgi-bin",
    "redirect",
    "wp-admin",
    "wp-login",
    "cpanel",
    "plesk",
    "adminpanel",
    "userlogin",
    "data",
    "file",
    "script",
    "exploit",
    "payload",
    " phishing",
    "suspicious",
    "vulnerability",
    "free",
    "gift",
    "prize",
    "winner",
    "congratulations",
    "urgent",
    "immediate",
    "actionrequired",
    "important",
    "critical",
    "invoice-",
    "payment-",
    "secure-",
    "login-",
    "verify-",
    "confirm-"
]

# Yazım denetleyicisi (performans için bir kere oluşturulur)
spell = SpellChecker()

def count_spelling_errors(text):
    """
    Verilen metindeki yazım hatası (imla) sayısını ve oranını hesaplar.

    Parametreler:
        text (str): Analiz edilecek metin.
    Dönüş:
        tuple: (yazım hatası sayısı, hata oranı)
    """
    if not text or not isinstance(text, str):
        return 0, 0.0
    
    # Metni kelimelere ayır (sadece harf içeren kelimeler)
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0, 0.0
        
    misspelled = spell.unknown(words)
    num_misspelled = len(misspelled)
    error_ratio = num_misspelled / len(words)
    return num_misspelled, error_ratio

def calculate_all_caps_ratio(text):
    """
    Metindeki tümü büyük harf olan kelimelerin oranını hesaplar.
    Parametreler:
        text (str): Analiz edilecek metin.
    Dönüş:
        float: Büyük harfli kelime oranı.
    """
    if not text or not isinstance(text, str):
        return 0.0
    words = re.findall(r'\b[A-Za-z]{2,}\b', text) # En az 2 harfli kelimeler
    if not words:
        return 0.0
    all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 1) # Tek harfli büyük harfleri (A, I) sayma
    return all_caps_words / len(words)

def calculate_special_char_ratio(text):
    """
    Metindeki özel karakterlerin (harf ve rakam olmayan) toplam karaktere oranını hesaplar.
    Parametreler:
        text (str): Analiz edilecek metin.
    Dönüş:
        float: Özel karakter oranı.
    """
    if not text or not isinstance(text, str):
        return 0.0
    body_length = len(text)
    if body_length == 0:
        return 0.0
    special_chars = sum(1 for char in text if not char.isalnum() and not char.isspace())
    return special_chars / body_length

def create_tfidf_features(corpus_processed_strings, vectorizer_path=None, save_path=None, **tfidf_params):
    """
    Verilen metinlerden TF-IDF öznitelik matrisini oluşturur veya kaydedilmiş bir vektörleştiriciyi yükler.
    Parametreler:
        corpus_processed_strings (list of str): İşlenmiş metinlerin listesi.
        vectorizer_path (str, optional): Var olan bir TF-IDF vektörleştiricisinin yolu.
        save_path (str, optional): Eğitilen vektörleştiricinin kaydedileceği yol.
        **tfidf_params: TF-IDF vektörleştirici için ek parametreler.
    Dönüş:
        features: TF-IDF öznitelik matrisi
        vectorizer: TF-IDF vektörleştirici nesnesi
    """
    if vectorizer_path and os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
        features = vectorizer.transform(corpus_processed_strings)
        print(f"TF-IDF vektörleştiricisi şuradan yüklendi: {vectorizer_path}")
    else:
        if not tfidf_params: # Varsayılan parametreler
            tfidf_params = {'max_df': 0.95, 'min_df': 5, 'ngram_range': (1, 2)}
        vectorizer = TfidfVectorizer(**tfidf_params)
        features = vectorizer.fit_transform(corpus_processed_strings)
        print("Yeni TF-IDF vektörleştiricisi eğitildi.")
        if save_path:
            joblib.dump(vectorizer, save_path)
            print(f"TF-IDF vektörleştiricisi şuraya kaydedildi: {save_path}")
    return features, vectorizer

def extract_additional_features(email_bodies_raw_series):
    """
    E-posta gövdelerinden ek özellikler (URL, anahtar kelime skorları, yazım/dilbilgisi hataları vb.) çıkarır.
    Parametreler:
        email_bodies_raw_series (pd.Series): E-posta gövdelerinin ham metinleri.
    Dönüş:
        np.ndarray: Ek özelliklerin bulunduğu matris.
    """
    features_list = []
    for body in email_bodies_raw_series:
        if not isinstance(body, str): # NaN veya başka bir tip gelirse
            body = ""

        # URL ile ilgili özellikler
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', body)
        num_urls = len(urls)
        num_https_urls = sum(1 for url in urls if url.startswith('https'))
        uses_ip_address_in_url_flag = 0 # Sadece flag olarak kullanıyoruz, bu yüzden _flag ekledim.
        # Yeni URL öznitelikleri için başlangıç değerleri
        max_subdomain_parts = 0
        any_url_contains_phishing_keywords = 0

        for url_text in urls:
            # IP adresi kullanımı kontrolü (her bir URL için)
            if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url_text):
                uses_ip_address_in_url_flag = 1 # Herhangi bir URL IP adresi içeriyorsa 1 yap

            # Alt alan adı sayısı tespiti
            try:
                ext = tldextract.extract(url_text)
                # ext.subdomain boş string ise 0 parça, aksi takdirde nokta ile ayrılmış parça sayısı
                current_subdomain_count = len(ext.subdomain.split('.')) if ext.subdomain else 0
                if current_subdomain_count > max_subdomain_parts:
                    max_subdomain_parts = current_subdomain_count
            except Exception: # tldextract ile ilgili bir sorun olursa diye
                pass # Şimdilik hata durumunda bu URL'i atla

            # URL metninde phishing anahtar kelimesi arama
            if any_url_contains_phishing_keywords == 0: # Zaten bulunduysa tekrar arama
                url_lower = url_text.lower()
                for keyword in URL_PHISHING_KEYWORDS:
                    if keyword in url_lower:
                        any_url_contains_phishing_keywords = 1
                        break # Bu URL için anahtar kelime bulundu, diğerlerine bakmaya gerek yok

        # Dilsel ve Psikolojik İpuçları
        body_lower = body.lower()
        urgency_score = sum(1 for keyword in URGENCY_KEYWORDS if keyword in body_lower)
        threat_score = sum(1 for keyword in THREAT_KEYWORDS if keyword in body_lower)
        reward_score = sum(1 for keyword in REWARD_KEYWORDS if keyword in body_lower)
        authority_score = sum(1 for keyword in AUTHORITY_KEYWORDS if keyword in body_lower)
        personal_info_score = sum(1 for keyword in PERSONAL_INFO_KEYWORDS if keyword in body_lower)
        contains_personal_info_request = int(personal_info_score > 0)
        # Yazım denetimi
        num_spelling_errors, spelling_error_ratio = count_spelling_errors(body)
        # Tüm büyük harf oranı
        all_caps_ratio = calculate_all_caps_ratio(body)
        # Özel karakter oranı
        special_char_ratio = calculate_special_char_ratio(body)

        # Gövde uzunluğu
        body_length = len(body)
        # Sayıların oranı
        num_digits = sum(c.isdigit() for c in body)
        digit_ratio = num_digits / (body_length + 1e-6) # Sıfıra bölme hatasını önle

        features_list.append({
            'num_urls': num_urls,
            'num_https_urls': num_https_urls,
            'uses_ip_address_in_url': uses_ip_address_in_url_flag,
            'max_subdomain_parts': max_subdomain_parts,
            'any_url_contains_phishing_keywords': any_url_contains_phishing_keywords,
            'urgency_score': urgency_score,
            'threat_score': threat_score,
            'reward_score': reward_score,
            'authority_score': authority_score,
            'personal_info_score': personal_info_score,
            'contains_personal_info_request': contains_personal_info_request,
            'num_spelling_errors': num_spelling_errors,
            'spelling_error_ratio': spelling_error_ratio,
            'all_caps_ratio': all_caps_ratio,
            'special_char_ratio': special_char_ratio,
            'body_length': body_length,
            'digit_ratio': digit_ratio
        })
    df_additional_features = pd.DataFrame(features_list)
    # Eksik değerleri (eğer oluşursa, string olmayan body'lerden kaynaklanabilir) 0 ile doldur
    return df_additional_features.fillna(0).values # NumPy array olarak döndür

def create_word_embedding_features(corpus_tokens,
                                   embedding_model_path=None,
                                   save_trained_model_path=None,
                                   model_type='word2vec',
                                   mode='avg',
                                   corpus_processed_strings_for_tfidf=None,
                                   tfidf_vectorizer_path=None,
                                   save_tfidf_vectorizer_path=None,
                                   **embedding_params):
    """
    Verilen token dizileriyle Word Embedding (Word2Vec/FastText) modelini yükler veya eğitir ve belge vektörlerini oluşturur.
    Ayrıca, istenirse TF-IDF ağırlıklı ortalama ile belge vektörleri döndürebilir.
    Parametreler:
        corpus_tokens (list of list of str): Tokenize edilmiş dokümanlar.
        embedding_model_path (str, optional): Var olan embedding modelinin yolu.
        save_trained_model_path (str, optional): Eğitilen modelin kaydedileceği yol.
        model_type (str): 'word2vec' veya 'fasttext'.
        mode (str): 'avg', 'sum' veya 'tfidf-avg'.
        corpus_processed_strings_for_tfidf (list of str, optional): TF-IDF için işlenmiş metinler.
        tfidf_vectorizer_path (str, optional): Var olan TF-IDF vektörleştirici yolu.
        save_tfidf_vectorizer_path (str, optional): Eğitilen TF-IDF vektörleştiricinin kaydedileceği yol.
        **embedding_params: Embedding modeli için ek parametreler.
    Dönüş:
        doc_vectors: Belge vektörleri (np.ndarray)
        embedding_model: Eğitimli embedding modeli
        tfidf_vec_for_return: TF-IDF vektörleştirici (veya None)
    """
    embedding_model = None
    tfidf_vec_for_return = None

    if embedding_model_path and os.path.exists(embedding_model_path):
        print(f"{model_type} modeli {embedding_model_path} adresinden yükleniyor...")
        if model_type.lower() == 'word2vec':
            embedding_model = Word2Vec.load(embedding_model_path)
        elif model_type.lower() == 'fasttext':
            embedding_model = FastText.load(embedding_model_path)
        else:
            raise ValueError("Geçersiz model_type. 'word2vec' veya 'fasttext' olmalı.")
        print("Model başarıyla yüklendi.")
    else:
        print(f"{embedding_model_path if embedding_model_path else 'Path belirtilmedi'}, model bulunamadı. Yeni {model_type} modeli eğitiliyor...")
        params = {
            'vector_size': 100, 'window': 5, 'min_count': 2,
            'workers': os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1,
        }
        params.update(embedding_params)

        if model_type.lower() == 'word2vec':
            embedding_model = Word2Vec(sentences=corpus_tokens, **params)
        elif model_type.lower() == 'fasttext':
            embedding_model = FastText(sentences=corpus_tokens, **params)
        else:
            raise ValueError("Geçersiz model_type. 'word2vec' veya 'fasttext' olmalı.")
        print("Yeni model başarıyla eğitildi.")

        if save_trained_model_path:
            embedding_model.save(save_trained_model_path)
            print(f"Eğitilen {model_type} modeli {save_trained_model_path} adresine kaydedildi.")

    vocab = embedding_model.wv.key_to_index
    embedding_vector_size = embedding_model.vector_size
    doc_vectors = np.zeros((len(corpus_tokens), embedding_vector_size))

    if mode == 'tfidf-avg':
        tfidf_vectorizer = None
        if tfidf_vectorizer_path and os.path.exists(tfidf_vectorizer_path):
            print(f"TF-IDF vektörleştirici {tfidf_vectorizer_path} adresinden yükleniyor...")
            tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
            print("TF-IDF vektörleştirici yüklendi.")
        else:
            print("Yeni TF-IDF vektörleştirici eğitiliyor...")
            if corpus_processed_strings_for_tfidf is None:
                print("corpus_processed_strings_for_tfidf belirtilmedi, corpus_tokens'tan stringler oluşturuluyor...")
                corpus_processed_strings_for_tfidf = [" ".join(tokens) for tokens in corpus_tokens]
            
            tfidf_params_default = {'max_df': 0.95, 'min_df': 1, 'ngram_range': (1,1)}
            tfidf_vectorizer = TfidfVectorizer(**tfidf_params_default) # embedding_params ile karışmaması için farklı isim
            tfidf_vectorizer.fit(corpus_processed_strings_for_tfidf)
            print("Yeni TF-IDF vektörleştirici eğitildi.")
            if save_tfidf_vectorizer_path:
                joblib.dump(tfidf_vectorizer, save_tfidf_vectorizer_path)
                print(f"TF-IDF vektörleştirici {save_tfidf_vectorizer_path} adresine kaydedildi.")
        
        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        temp_corpus_strings = [" ".join(doc_tokens) for doc_tokens in corpus_tokens]
        doc_tfidf_matrix = tfidf_vectorizer.transform(temp_corpus_strings)
        tfidf_vec_for_return = tfidf_vectorizer

    for i, doc_tokens in enumerate(corpus_tokens):
        word_vectors = []
        weights = []
        current_doc_tfidf_scores = {}
        if mode == 'tfidf-avg':
            feature_indices = doc_tfidf_matrix[i].indices
            tfidf_scores = doc_tfidf_matrix[i].data
            for idx, score in zip(feature_indices, tfidf_scores):
                current_doc_tfidf_scores[tfidf_feature_names[idx]] = score

        for token in doc_tokens:
            if token in vocab: # veya embedding_model.wv.has_index_for(token)
                word_vectors.append(embedding_model.wv[token])
                if mode == 'tfidf-avg':
                    weights.append(current_doc_tfidf_scores.get(token, 0))

        if not word_vectors:
            doc_vectors[i] = np.zeros(embedding_vector_size)
            continue

        word_vectors_np = np.array(word_vectors)

        if mode == 'avg':
            doc_vectors[i] = np.mean(word_vectors_np, axis=0)
        elif mode == 'sum':
            doc_vectors[i] = np.sum(word_vectors_np, axis=0)
        elif mode == 'tfidf-avg':
            if not weights or sum(weights) == 0:
                doc_vectors[i] = np.mean(word_vectors_np, axis=0)
            else:
                weights_np = np.array(weights).reshape(-1, 1)
                weighted_sum_vectors = np.sum(word_vectors_np * weights_np, axis=0)
                sum_of_weights = np.sum(weights_np)
                doc_vectors[i] = weighted_sum_vectors / sum_of_weights if sum_of_weights != 0 else np.zeros(embedding_vector_size)
        else:
            raise ValueError(f"Geçersiz mode: {mode}. 'avg', 'sum', veya 'tfidf-avg' olmalı.")
            
    return doc_vectors, embedding_model, tfidf_vec_for_return

def combine_feature_sets(base_features_sparse, additional_features_dense):
    """
    Temel (ör. TF-IDF) ve ek (ör. istatistiksel) özellik matrislerini birleştirir.
    Parametreler:
        base_features_sparse: Temel özellikler (sparse matris).
        additional_features_dense: Ek özellikler (numpy array).
    Dönüş:
        csr_matrix: Birleştirilmiş özellik matrisi.
    """
    if additional_features_dense.shape[0] != base_features_sparse.shape[0]:
        raise ValueError(f"Satır sayıları uyuşmuyor: Temel Özellikler={base_features_sparse.shape[0]}, Ek Özellikler={additional_features_dense.shape[0]}")
    # additional_features_dense zaten numpy array ise csr_matrix'e çevirmeye gerek yok, hstack halleder.
    return hstack([base_features_sparse, additional_features_dense]).tocsr()