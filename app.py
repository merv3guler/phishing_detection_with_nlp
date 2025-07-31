import os
import joblib
import pandas as pd
import numpy as np
import argparse
import sys

# .eml dosyalarını işlemek için
import email
from email import policy
from email.parser import BytesParser

# .msg dosyalarını işlemek için (üçüncü taraf kütüphane)
try:
    import extract_msg
except ImportError:
    extract_msg = None  # Kütüphanenin varlığını kontrol edebilmek için

# Özel ön işleme modülü
from preprocessing import Preprocessor

# Özel vektörleştirme fonksiyonları ve NLTK/language_tool kurulumu
from vectorization import (
    extract_additional_features,
    combine_feature_sets,
)

# --- Genel Değişkenler ---
model = None
tfidf_vectorizer = None
additional_features_scaler = None
preprocessor_instance = None
VERBOSE_MODE = False # --verbose bayrağı ile kontrol edilecek

# --- Yapılandırma ---
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "linearsvm_tfidf_gelişmiş.joblib")
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_gelismis_vectorizer.joblib")
ADDITIONAL_FEATURES_SCALER_PATH = os.path.join(MODEL_DIR, "minmax_scaler_tfidf_gelismis_additional.joblib")

def v_print(message):
    """
    Sadece verbose mod aktifken mesaj yazdırır.
    Parametreler:
        message (str): Yazdırılacak mesaj.
    """
    if VERBOSE_MODE:
        print(message)

def load_artifacts():
    """
    Önceden eğitilmiş modeli, vektörleştiricileri ve ön işleyici örneğini yükler.
    """
    global model, tfidf_vectorizer, additional_features_scaler, preprocessor_instance
    v_print("Model ve yardımcı araçlar yükleniyor...")
    try:
        if not os.path.exists(MODEL_DIR):
            print(f"Hata: Model dizini '{MODEL_DIR}' bulunamadı.", file=sys.stderr)
            raise FileNotFoundError(f"Model dizini '{MODEL_DIR}' bulunamadı.")

        if not os.path.exists(MODEL_PATH):
            print(f"Hata: Model dosyası '{MODEL_PATH}' bulunamadı.", file=sys.stderr)
            raise FileNotFoundError(f"Model dosyası '{MODEL_PATH}' bulunamadı.")
        model = joblib.load(MODEL_PATH)
        v_print(f"Model şuradan yüklendi: {MODEL_PATH}")

        if not os.path.exists(TFIDF_VECTORIZER_PATH):
            print(f"Hata: TF-IDF vektörleştirici dosyası '{TFIDF_VECTORIZER_PATH}' bulunamadı.", file=sys.stderr)
            raise FileNotFoundError(f"TF-IDF vektörleştirici dosyası '{TFIDF_VECTORIZER_PATH}' bulunamadı.")
        tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        v_print(f"TF-IDF vektörleştirici şuradan yüklendi: {TFIDF_VECTORIZER_PATH}")

        if not os.path.exists(ADDITIONAL_FEATURES_SCALER_PATH):
            print(f"Hata: Ek özellikler için ölçekleyici dosyası '{ADDITIONAL_FEATURES_SCALER_PATH}' bulunamadı.", file=sys.stderr)
            print("Bu dosya, TFIDF_Gelişmiş hattının ek özellikleri için MinMaxScaler içermelidir.", file=sys.stderr)
            raise FileNotFoundError(f"Ek özellikler için ölçekleyici dosyası '{ADDITIONAL_FEATURES_SCALER_PATH}' bulunamadı.")
        additional_features_scaler = joblib.load(ADDITIONAL_FEATURES_SCALER_PATH)
        v_print(f"Ek özellikler için ölçekleyici şuradan yüklendi: {ADDITIONAL_FEATURES_SCALER_PATH}")

        preprocessor_instance = Preprocessor(language='english')
        v_print("Ön işleyici başlatıldı.")
        # Aşağıdaki satır geliştirici notudur, vectorization.py import edildiğinde bu araçlar zaten başlatılır.
        # v_print("vectorization.py'nin genel araçları (grammar_tool, spell) başlatılmış olmalıdır.")
        v_print("Model ve yardımcı araçlar başarıyla yüklendi.")

    except FileNotFoundError as fnf_error:
        # Bu hatalar zaten stderr'e yazdırıldığı için tekrar print etmeye gerek yok.
        # print(f"Dosya Bulunamadı Hatası (artifact loading): {fnf_error}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Model ve yardımcı araçlar yüklenirken beklenmedik bir hata oluştu: {e}", file=sys.stderr)
        sys.exit(1)

def get_email_body_from_eml(file_path):
    """
    Verilen .eml dosyasından e-posta gövdesini çıkarır.
    Parametreler:
        file_path (str): .eml dosyasının yolu.
    Dönüş:
        str: E-posta gövdesi metni.
    """
    try:
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)

        body_text_plain = ""
        body_text_html = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                charset = part.get_content_charset() or 'utf-8'

                if "attachment" not in content_disposition.lower() and part.get_payload(decode=True):
                    payload_decoded = None
                    try:
                        payload_decoded = part.get_payload(decode=True).decode(charset, errors='replace')
                    except (UnicodeDecodeError, AttributeError, LookupError):
                        payload_decoded = part.get_payload(decode=True).decode('latin-1', errors='replace')

                    if content_type == "text/plain" and not body_text_plain:
                        body_text_plain = payload_decoded
                    elif content_type == "text/html" and not body_text_html:
                        body_text_html = payload_decoded
        else:
            charset = msg.get_content_charset() or 'utf-8'
            payload_decoded = None
            try:
                payload_decoded = msg.get_payload(decode=True).decode(charset, errors='replace')
            except (UnicodeDecodeError, AttributeError, LookupError):
                payload_decoded = msg.get_payload(decode=True).decode('latin-1', errors='replace')

            if msg.get_content_type() == "text/plain":
                body_text_plain = payload_decoded
            elif msg.get_content_type() == "text/html":
                body_text_html = payload_decoded

        extracted_body = body_text_plain if body_text_plain else body_text_html
        if not extracted_body:
            raw_payload = msg.get_payload()
            if isinstance(raw_payload, str):
                extracted_body = raw_payload
        if not extracted_body:
            print(f"Uyarı: '{file_path}' (.eml) dosyasından e-posta gövdesi çıkarılamadı. "
                  f"Dosya yapısı beklenenden farklı olabilir veya gövde boş olabilir.", file=sys.stderr)
        return extracted_body if extracted_body else ""
    except Exception as e:
        print(f"Hata: '{file_path}' (.eml) dosyası işlenirken bir sorun oluştu: {e}", file=sys.stderr)
        return ""

def get_email_body_from_msg(file_path):
    """
    Verilen .msg dosyasından e-posta gövdesini çıkarır.
    Parametreler:
        file_path (str): .msg dosyasının yolu.
    Dönüş:
        str: E-posta gövdesi metni.
    """
    if extract_msg is None:
        print(f"Hata: .msg dosyalarını işlemek için 'extract_msg' kütüphanesi gerekli. "
              f"Lütfen 'pip install extract_msg' komutu ile kurun.", file=sys.stderr)
        return None
    try:
        msg_obj = extract_msg.Message(file_path)
        body_content = msg_obj.body
        if not body_content and msg_obj.html_body:
            v_print(f"Bilgi: '{file_path}' (.msg) için düz metin gövde bulunamadı, HTML gövde kullanılıyor.")
            body_content = msg_obj.html_body
        if not body_content:
            print(f"Uyarı: '{file_path}' (.msg) dosyasından e-posta gövdesi çıkarılamadı. "
                  f"Dosya boş olabilir veya desteklenmeyen bir yapıya sahip olabilir.", file=sys.stderr)
        return body_content if body_content else ""
    except Exception as e:
        print(f"Hata: '{file_path}' (.msg) dosyası işlenirken bir sorun oluştu: {e}", file=sys.stderr)
        return ""

def predict_email_phishing(email_text_raw: str):
    """
    Yüklenen modeli ve TFIDF_Gelişmiş özelliklerini kullanarak bir e-posta metninin phishing olup olmadığını tahmin eder.
    Parametreler:
        email_text_raw (str): Ham e-posta metni.
    Dönüş:
        tuple: (tahmin sonucu (0/1), işlenmiş metin parçası)
    """
    if not all([model, tfidf_vectorizer, additional_features_scaler, preprocessor_instance]):
        # Bu hata load_artifacts içinde zaten ele alınmış olmalı, ancak bir güvence olarak kalabilir.
        print("Hata: Model ve yardımcı araçlar yüklenmemiş.", file=sys.stderr)
        raise Exception("Model ve yardımcı araçlar yüklenmemiş.")

    if pd.isna(email_text_raw): # Girdi None veya NaN ise boş stringe çevir
        email_text_raw = ""

    processed_output = preprocessor_instance.preprocess(email_text_raw, for_vectorization=False, include_pos=False)
    processed_body_str_for_tfidf = " ".join(processed_output['tokens'])
    email_body_series = pd.Series([email_text_raw])
    additional_feats_from_func = extract_additional_features(email_body_series)
    readability_fk = processed_output['readability']['flesch_kincaid_grade']
    readability_gf = processed_output['readability']['gunning_fog']
    sentiment_compound = processed_output['sentiment']['compound']
    readability_sentiment_feats = np.array([[readability_fk, readability_gf, sentiment_compound]])
    all_additional_feats_unscaled = np.hstack((additional_feats_from_func, readability_sentiment_feats))
    all_additional_feats_scaled = additional_features_scaler.transform(all_additional_feats_unscaled)
    tfidf_features_sparse = tfidf_vectorizer.transform([processed_body_str_for_tfidf])
    final_features_for_model = combine_feature_sets(tfidf_features_sparse, all_additional_feats_scaled)
    prediction = model.predict(final_features_for_model)
    return int(prediction[0]), processed_body_str_for_tfidf

def main():
    """
    Komut satırından argümanları alır, e-posta içeriğini okur ve phishing tahmini yapar.
    """
    global VERBOSE_MODE # GlobalVERBOSE_MODE değişkenini main içinde ayarlayacağız

    parser = argparse.ArgumentParser(
        description="Bir e-postanın phishing olup olmadığını tahmin eder.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        type=str,
        help="Analiz edilecek e-posta dosyasının yolu.\n"
             "Desteklenen formatlar:\n"
             "  .txt    (Düz metin e-posta gövdesi)\n"
             "  .eml    (Standart e-posta formatı)\n"
             "  .msg    (Microsoft Outlook e-posta formatı,\n"
             "           'extract_msg' kütüphanesi gereklidir)"
    )
    group.add_argument("--text", type=str, help="Analiz edilecek e-posta metni (doğrudan giriş).")
    parser.add_argument("--verbose", action="store_true", help="İşlem adımları ve çıkarılan metin parçacığı gibi daha fazla detay yazdırır.")

    args = parser.parse_args()
    VERBOSE_MODE = args.verbose # --verbose bayrağına göre global değişkeni ayarla

    try:
        load_artifacts()
    except Exception:
        # load_artifacts zaten hatayı yazdırıp sys.exit() ile çıkıyor.
        # Bu yüzden burada ek bir işlem yapmaya gerek yok.
        sys.exit(1) # Yine de burada bir çıkış olması iyi bir pratik

    email_content = ""
    source_description = ""

    if args.file:
        file_path = args.file
        file_ext = os.path.splitext(file_path)[1].lower()
        source_description = f"'{file_path}' dosyasından"
        v_print(f"{source_description} e-posta içeriği işleniyor...")

        if not os.path.exists(file_path):
            print(f"Hata: '{file_path}' dosyası bulunamadı.", file=sys.stderr)
            sys.exit(1)

        if file_ext == ".eml":
            email_content = get_email_body_from_eml(file_path)
        elif file_ext == ".msg":
            email_content = get_email_body_from_msg(file_path)
            if email_content is None: # extract_msg kütüphanesi eksikse
                sys.exit(1)
        elif file_ext == ".txt" or file_ext == "":
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    email_content = f.read()
                v_print(f"{source_description} (düz metin) içerik okundu.")
            except Exception as e:
                print(f"Hata: '{file_path}' (düz metin) dosyası okunurken bir sorun oluştu: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # Bilinmeyen uzantılar için son kullanıcıya her zaman bir uyarı ver
            print(f"Uyarı: '{file_path}' için dosya uzantısı ('{file_ext}') tanınmıyor. "
                  f"Düz metin olarak okunmaya çalışılacak.", file=sys.stdout)
            v_print(f"En iyi sonuçlar için .txt, .eml veya .msg uzantılı dosyaları kullanın.")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    email_content = f.read()
                v_print(f"{source_description} (düz metin varsayılarak) içerik okundu.")
            except Exception as e:
                print(f"Hata: '{file_path}' dosyası (düz metin varsayılarak) okunurken bir sorun oluştu: {e}", file=sys.stderr)
                sys.exit(1)
    elif args.text:
        email_content = args.text
        source_description = "doğrudan metin girdisinden"
        v_print(f"{source_description} e-posta içeriği alındı.")

    if not email_content and args.file:
        # get_email_body_from_eml/msg zaten uyarı veriyor, burada genel bir uyarı daha olabilir
        print(f"Uyarı: {source_description} analiz edilecek e-posta içeriği alınamadı. "
              "Lütfen dosyanın var olduğundan, doğru formatta olduğundan ve boş olmadığından emin olun.", file=sys.stderr)
        sys.exit(1)
    elif not email_content and args.text: # --text ile boş metin verilirse
        print("Hata: Sağlanan e-posta metni boş olamaz.", file=sys.stderr)
        sys.exit(1)

    try:
        prediction_result, processed_snippet = predict_email_phishing(email_content)
        status_message = "Phishing" if prediction_result == 1 else "Meşru (Legitimate)"

        # Son kullanıcıya gösterilecek ana çıktı
        print(f"\n--- Tahmin Sonucu ({source_description}) ---")
        print(f"Durum: {status_message} (Model Tahmini: {prediction_result})")

        if args.verbose:
            print(f"\nÇıkarılan/İşlenen Metin Parçacığı (İlk 200 karakter):\n{processed_snippet[:200]}...")

    except Exception as e:
        print(f"\nTahmin sırasında bir hata oluştu: {e}", file=sys.stderr)
        if args.verbose: # Verbose modda hatanın detayını göster
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()