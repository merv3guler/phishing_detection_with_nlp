# Gereken kütüphaneler
import os
import sys
import joblib
import traceback
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from statsmodels.stats.contingency_tables import mcnemar

# Yerel modüllerden importlar
from preprocessing import Preprocessor
from vectorization import create_tfidf_features, extract_additional_features, combine_feature_sets, create_word_embedding_features

# Sabitler ve Yollar
DATA_PATH = "data/CEAS_08.csv"
RESULTS_DIR = "results"
MODELS_DIR = "models"
PREPROCESSED_DATA_PATH = os.path.join(RESULTS_DIR, "CEAS_08_preprocessed.parquet")
PERFORMANCE_FILE = os.path.join(RESULTS_DIR, "model_performance_comparison.csv")
CONFUSION_MATRIX_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
ROC_CURVES_DIR = os.path.join(RESULTS_DIR, "roc_curves")

# Dizinleri oluştur
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)
os.makedirs(ROC_CURVES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def apply_preprocessing(text_data, preprocessor_instance):
    if pd.isna(text_data):
        processed_output = preprocessor_instance.preprocess("", for_vectorization=False, include_pos=False)
    else:
        processed_output = preprocessor_instance.preprocess(text_data, for_vectorization=False, include_pos=False)

    cleaned_tokens = [
        str(token) for token in processed_output['tokens']
        if token is not None and str(token).strip() != ""
    ]

    return pd.Series({
        'processed_body_str': " ".join(cleaned_tokens),
        'processed_body_tokens': cleaned_tokens,
        'cleaned_text': processed_output['cleaned_text'],
        'readability_fk': processed_output['readability']['flesch_kincaid_grade'],
        'readability_gf': processed_output['readability']['gunning_fog'],
        'sentiment_compound': processed_output['sentiment']['compound']
    })

def plot_confusion_matrix(cm, classes, model_name, feature_set_name, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Karışıklık Matrisi\nModel: {model_name}\nÖznitelik Seti: {feature_set_name}')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve_and_save(y_test, y_pred_proba, model_name, feature_set_name, save_path):
    if y_pred_proba is None:
        print(f"    Uyarı: ROC Eğrisi çizilemedi ({model_name}, {feature_set_name}) - Olasılık skorları yok.")
        return

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
    roc_auc_value = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC eğrisi (AUC = {roc_auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı (FPR)')
    plt.ylabel('Doğru Pozitif Oranı (TPR)')
    plt.title(f'ROC Eğrisi\nModel: {model_name}\nÖznitelik Seti: {feature_set_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    try:
        plt.savefig(save_path)
    except Exception as e_save_roc:
        print(f"    HATA: ROC Eğrisi kaydedilemedi {save_path}: {e_save_roc}")
    plt.close()

def get_combined_additional_features(current_df: pd.DataFrame) -> np.ndarray:
    additional_feats_from_func = extract_additional_features(current_df['body'])
    readability_sentiment_cols = ['readability_fk', 'readability_gf', 'sentiment_compound']
    if not all(col in current_df.columns for col in readability_sentiment_cols):
        eksik_sutunlar = [col for col in readability_sentiment_cols if col not in current_df.columns]
        raise KeyError(f"DataFrame'de gerekli okunabilirlik/duygu sütunları eksik: {eksik_sutunlar}")
    readability_sentiment_feats = current_df[readability_sentiment_cols].values
    if additional_feats_from_func.shape[0] != readability_sentiment_feats.shape[0]:
        raise ValueError(
            f"Ek özellikler ve okunabilirlik/duygu skorları arasında satır sayısı uyuşmazlığı: "
            f"extract_additional_features = {additional_feats_from_func.shape[0]}, "
            f"readability_sentiment_feats = {readability_sentiment_feats.shape[0]}"
        )
    all_additional_feats_unscaled = np.hstack((additional_feats_from_func, readability_sentiment_feats))
    return all_additional_feats_unscaled

def tfidf_gelismis_pipeline(current_df):
    tfidf_features, tfidf_vec = create_tfidf_features(current_df['processed_body_str'].tolist(),
                                                      save_path=os.path.join(MODELS_DIR, "tfidf_gelismis_vectorizer.joblib"))
    all_additional_feats_unscaled = get_combined_additional_features(current_df)
    scaler_additional = MinMaxScaler()
    all_additional_feats_scaled = scaler_additional.fit_transform(all_additional_feats_unscaled)
    return combine_feature_sets(tfidf_features, all_additional_feats_scaled), {
        "tfidf_vectorizer": tfidf_vec,
        "additional_features_shape": all_additional_feats_scaled.shape[1],
        "type": "sparse",
        "scaler_additional_features": scaler_additional
    }

def wordembedding_gelismis_pipeline(current_df):
    embedding_features_dense, embedding_model, _ = create_word_embedding_features(
        corpus_tokens=current_df['processed_body_tokens'].tolist(),
        model_type='fasttext',
        save_trained_model_path=os.path.join(MODELS_DIR, "word_embedding_gelismis_model.joblib"),
        vector_size=150,
        epochs=20,
        min_count=1
    )
    all_additional_feats_unscaled = get_combined_additional_features(current_df)
    if embedding_features_dense.shape[0] != all_additional_feats_unscaled.shape[0]:
        raise ValueError(f"Satır sayıları uyuşmuyor: Embedding Feat.={embedding_features_dense.shape[0]}, Tüm Ek Feat.={all_additional_feats_unscaled.shape[0]}")
    combined_dense_features = np.hstack((embedding_features_dense, all_additional_feats_unscaled))
    meta_data = {"embedding_model": embedding_model,
                   "additional_features_shape": all_additional_feats_unscaled.shape[1],
                   "type": "dense"}
    return combined_dense_features, meta_data

def perform_mcnemar_test(y_true, y_pred_model1, y_pred_model2, model1_name, model2_name, feature_set_name1, feature_set_name2=""):
    n10 = np.sum((y_pred_model1 == y_true) & (y_pred_model2 != y_true))
    n01 = np.sum((y_pred_model1 != y_true) & (y_pred_model2 == y_true))
    n11 = np.sum((y_pred_model1 == y_true) & (y_pred_model2 == y_true))
    n00 = np.sum((y_pred_model1 != y_true) & (y_pred_model2 != y_true))
    table = [[n11, n10], [n01, n00]]
    result_dict = {
        "Model1": model1_name, "FS1": feature_set_name1, "Model2": model2_name,
        "FS2": feature_set_name2 if feature_set_name2 and feature_set_name1 != feature_set_name2 else feature_set_name1,
        "N10_M1_Doğru_M2_Yanlış": n10, "N01_M1_Yanlış_M2_Doğru": n01,
        "Statistic": None, "P_Value": None, "Yorum": ""
    }
    comparison_str = f"{model1_name} ({feature_set_name1})"
    if feature_set_name2 and feature_set_name1 != feature_set_name2:
        comparison_str += f" vs. {model2_name} ({feature_set_name2})"
    else:
        comparison_str += f" vs. {model2_name} (aynı özellik seti: {feature_set_name1})"
    print(f"\n  McNemar Testi Sonucu ({comparison_str}):")
    print(f"    Model 1 Doğru / Model 2 Yanlış (b): {n10}")
    print(f"    Model 1 Yanlış / Model 2 Doğru (c): {n01}")
    try:
        result_mcnemar = mcnemar(table, exact=False, correction=True)
        p_value = result_mcnemar.pvalue
        statistic = result_mcnemar.statistic
        result_dict["Statistic"] = statistic
        result_dict["P_Value"] = p_value
        print(f"    İstatistik: {statistic:.4f}, p-değeri: {p_value:.4f}")
        comment = "Performanslar arasında istatistiksel olarak ANLAMLI bir fark vardır (p < 0.05)." if p_value < 0.05 else "Performanslar arasında istatistiksel olarak anlamlı bir fark YOKTUR (p >= 0.05)."
        print(f"    Yorum: {comment}")
        result_dict["Yorum"] = comment
    except ValueError as ve:
        comment = f"Hesaplanamadı: {ve}. Hata paternleri aynı olabilir (b+c=0)."
        print(f"    {comment}")
        result_dict["Yorum"] = comment
    return result_dict

def run_mcnemar_if_predictions_exist(all_predictions, key1, key2, model1_print_name, model2_print_name, fs_name1_print, fs_name2_print=""):
    if key1 in all_predictions and key2 in all_predictions:
        y_pred1, y_test_shared = all_predictions[key1]
        y_pred2, _ = all_predictions[key2]
        return perform_mcnemar_test(y_test_shared, y_pred1, y_pred2, model1_print_name, model2_print_name, fs_name1_print, fs_name2_print)
    else:
        missing_keys_list = [str(k) for k in [key1, key2] if k not in all_predictions]
        return None
    
def run_experiment():
    df = None
    if os.path.exists(PREPROCESSED_DATA_PATH):
        print(f"Önceden işlenmiş veri yükleniyor: {PREPROCESSED_DATA_PATH}")
        try:
            df = pd.read_parquet(PREPROCESSED_DATA_PATH)            
            if 'label' in df.columns:
                df['label'] = df['label'].astype(int)
            else:
                print(f"HATA: Yüklenen ön işlenmiş veride 'label' sütunu bulunamadı: {PREPROCESSED_DATA_PATH}")
                sys.exit("Program 'label' sütunu eksikliği nedeniyle sonlandırıldı.")

            if 'processed_body_tokens' in df.columns and not df.empty:
                df['processed_body_tokens'] = df['processed_body_tokens'].apply(
                    lambda x: [str(token) for token in list(x)] if isinstance(x, np.ndarray) else 
                              ([str(token) for token in x] if isinstance(x, list) else x)
                )
                first_valid_tokens_series = df['processed_body_tokens'].dropna()
                if not first_valid_tokens_series.empty:
                    first_valid_tokens_value = first_valid_tokens_series.iloc[0]
                    if isinstance(first_valid_tokens_value, str): # Eğer hala string ise (beklenmedik durum)
                        try:
                            df['processed_body_tokens'] = df['processed_body_tokens'].apply(
                                lambda x: literal_eval(x) if isinstance(x, str) else x
                            )
                        except Exception as e_literal:
                            print(f"HATA: literal_eval sırasında hata (ikinci deneme): {e_literal}")
            else:
                sys.exit("Program 'processed_body_tokens' sütunu eksikliği nedeniyle sonlandırıldı.")
        except Exception as e:
            print(f"Önceden işlenmiş veri ({PREPROCESSED_DATA_PATH}) yüklenirken hata: {e}.")
            traceback.print_exc()
            df = None
    else:
        df = None

    if df is None:
        try:
            df_original = pd.read_csv(DATA_PATH, usecols=['body', 'label'])
            df_original.dropna(subset=['body', 'label'], inplace=True)
            df_original['label'] = df_original['label'].astype(int)
        except FileNotFoundError:
            print(f"HATA: Orijinal veri seti bulunamadı: {DATA_PATH}")
            sys.exit("Program orijinal veri dosyası eksikliği nedeniyle sonlandırıldı.")
        except Exception as e:
            print(f"Orijinal veri yüklenirken hata: {e}")
            sys.exit("Program orijinal veri yükleme hatası nedeniyle sonlandırıldı.")

        if df_original.empty:
            print("Orijinal veri seti boş. İşlem durduruldu.")
            sys.exit("Program boş orijinal veri seti nedeniyle sonlandırıldı.")
        try:
            preprocessor_instance = Preprocessor()
        except Exception as e:
            print(f"Ön işleme nesnesi oluşturulurken hata: {e}")
            sys.exit("Program ön işleme nesnesi oluşturma hatası nedeniyle sonlandırıldı.")
        try:
            processed_data_df = df_original['body'].apply(lambda x: apply_preprocessing(x, preprocessor_instance))
            df = pd.concat([df_original.reset_index(drop=True), processed_data_df.reset_index(drop=True)], axis=1)
        except Exception as e:
            print(f"Ön işleme sırasında genel bir hata oluştu: {e}")
            traceback.print_exc()
            sys.exit("Program ön işleme hatası nedeniyle sonlandırıldı.")
        try:
            df.to_parquet(PREPROCESSED_DATA_PATH, index=False)
        except ImportError:
            print("HATA: Parquet formatında kaydetmek için 'pyarrow' veya 'fastparquet' kütüphanesi gereklidir.")
            raise Exception("Parquet kaydı için gerekli kütüphane eksik.")
        except Exception as e:
            print(f"Ön işlenmiş veri kaydedilirken hata: {e}")

    if df is None or df.empty:
        print("Veri işlenemedi veya yüklenemedi. Deney durduruldu.")
        sys.exit("Program veri yokluğu nedeniyle sonlandırıldı.")

    try:
        corpus_tokens_for_we_temel = df['processed_body_tokens'].tolist()
    except KeyError:
        print("HATA: DataFrame'de 'processed_body_tokens' sütunu bulunamadı. WordEmbedding_Temel oluşturulamıyor.")
        sys.exit("Program 'processed_body_tokens' eksikliği nedeniyle sonlandırıldı.")
    except Exception as e_corpus_prep:
        print(f"HATA: WordEmbedding_Temel için corpus_tokens hazırlanırken bir sorun oluştu: {e_corpus_prep}")
        traceback.print_exc()
        sys.exit("Program corpus_tokens hazırlama hatası nedeniyle sonlandırıldı.")

    y = df['label'].values

    feature_extraction_pipelines = {
        "TFIDF_Temel": lambda: create_tfidf_features(df['processed_body_str'].tolist(),
                                                 save_path=os.path.join(MODELS_DIR, "tfidf_temel_vectorizer.joblib")),
        "WordEmbedding_Temel": lambda: (
            (features_tuple := create_word_embedding_features(
                corpus_tokens_for_we_temel, 
                model_type='word2vec',
                save_trained_model_path=os.path.join(MODELS_DIR, "word_embedding_temel_model.joblib")
            )),
            features_tuple[0], 
            {"embedding_model": features_tuple[1], "type": "dense"} 
        )[1:], 
        "TFIDF_Gelişmiş": lambda: tfidf_gelismis_pipeline(df),
        "WordEmbedding_Gelişmiş": lambda: wordembedding_gelismis_pipeline(df)
    }
    models_to_train = {
        "NaiveBayes": MultinomialNB(),
        "LinearSVM": LinearSVC(dual="auto", C=0.1, max_iter=6000, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }
    models_to_train_dense = {
        "NaiveBayes_Gaussian": GaussianNB(),
        "LinearSVM": LinearSVC(dual="auto", C=0.1, max_iter=6000, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }

    all_predictions = {}
    all_results = []

    for fs_name, fs_func in feature_extraction_pipelines.items():
        print(f"\n--- Öznitelik Seti: {fs_name} ---")
        try:
            X_features, vectorizers_or_meta = fs_func() 
            if fs_name == "TFIDF_Gelişmiş" and "scaler_additional_features" in vectorizers_or_meta:
                scaler_to_save = vectorizers_or_meta["scaler_additional_features"]
                scaler_filename = os.path.join(MODELS_DIR, "minmax_scaler_tfidf_gelismis_additional.joblib")
                try:
                    joblib.dump(scaler_to_save, scaler_filename)
                except Exception as e_save_scaler:
                    print(f"    HATA: TFIDF_Gelişmiş için ek özellikler MinMaxScaler kaydedilemedi {scaler_filename}: {e_save_scaler}")
            is_sparse = hasattr(X_features, 'tocsr') 
        except Exception as e:
            print(f"Hata: {fs_name} için öznitelik çıkarımı başarısız: {e}")
            traceback.print_exc()
            continue

        X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.25, random_state=42, stratify=y)

        scaler = None 
        if isinstance(vectorizers_or_meta, dict) and vectorizers_or_meta.get("type") == "dense":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train) 
            X_test = scaler.transform(X_test)     
            vectorizers_or_meta['scaler'] = scaler 
            scaler_filename = os.path.join(MODELS_DIR, f"scaler_{fs_name.lower().replace(' ', '_')}.joblib")
            try:
                joblib.dump(scaler, scaler_filename)
            except Exception as e_save_scaler:
                print(f"    HATA: Scaler kaydedilemedi {scaler_filename}: {e_save_scaler}")

        current_models_dict = models_to_train
        if not is_sparse and "NaiveBayes" in current_models_dict : 
            current_models_dict = {k:v for k,v in models_to_train_dense.items()}

        for model_name, model_instance in current_models_dict.items():
            if model_name == "NaiveBayes" and not is_sparse:
                continue
            print(f"  Model Eğitiliyor: {model_name}")
            try:
                model_instance.fit(X_train, y_train)
            except Exception as e:
                print(f"    HATA: {model_name} modeli {fs_name} ile eğitilirken sorun: {e}")
                traceback.print_exc() 
                continue

            y_pred = model_instance.predict(X_test)
            all_predictions[(fs_name, model_name)] = (y_pred, y_test) 

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            y_pred_proba_for_roc = None
            roc_auc = None
            try:
                if hasattr(model_instance, "predict_proba"):
                    y_pred_proba_for_roc = model_instance.predict_proba(X_test)[:, 1]
                elif hasattr(model_instance, "decision_function"):
                    y_pred_proba_for_roc = model_instance.decision_function(X_test)
                
                if y_pred_proba_for_roc is not None:
                    roc_auc = roc_auc_score(y_test, y_pred_proba_for_roc)
            except Exception as e_roc:
                print(f"    Uyarı: ROC AUC hesaplanırken hata ({model_name}, {fs_name}): {e_roc}")

            print(f"    {model_name} - {fs_name}: Acc={accuracy:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, AUC={roc_auc if roc_auc is not None else 'N/A'}")
            all_results.append({
                "FeatureSet": fs_name, "Model": model_name, "Accuracy": accuracy,
                "Precision": precision, "Recall": recall, "F1_Score": f1, "ROC_AUC": roc_auc
            })
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            cm_save_path = os.path.join(CONFUSION_MATRIX_DIR, f"{model_name}_{fs_name}_cm.png")
            plot_confusion_matrix(cm, classes=["Meşru (0)", "Phishing (1)"], model_name=model_name, feature_set_name=fs_name, save_path=cm_save_path)
            if y_pred_proba_for_roc is not None:
                roc_curve_save_path = os.path.join(ROC_CURVES_DIR, f"{model_name}_{fs_name}_roc.png")
                plot_roc_curve_and_save(y_test, y_pred_proba_for_roc, model_name, fs_name, roc_curve_save_path)
            else:
                pass
            model_filename = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}_{fs_name.lower().replace(' ', '_')}.joblib")
            try:
                joblib.dump(model_instance, model_filename)
            except Exception as e_save:
                print(f"    HATA: Model kaydedilemedi {model_filename}: {e_save}")

    if all_results:
        df_performance = pd.DataFrame(all_results)
        df_performance.to_csv(PERFORMANCE_FILE, index=False)
        print(f"\nPerformans sonuçları kaydedildi: {PERFORMANCE_FILE}")
        print("\n--- Performans Özeti ---")
        print(df_performance.sort_values(by="F1_Score", ascending=False))
        print("\n\n--- McNemar İstatistiksel Anlamlılık Testleri ---")
        mcnemar_results_list = []
        print("\n--- Öncelik 1: Gelişmiş Öznitelik Setinin Etkisi (Temel vs. Gelişmiş) ---")
        print("\n  1.1. TF-IDF Tabanlı (Temel vs. Gelişmiş):")
        for model_actual_name in ["LinearSVM", "RandomForest", "NaiveBayes"]:
            mcnemar_res = run_mcnemar_if_predictions_exist(all_predictions, ("TFIDF_Temel", model_actual_name), ("TFIDF_Gelişmiş", model_actual_name), model_actual_name, model_actual_name, "TFIDF_Temel", "TFIDF_Gelişmiş")
            if mcnemar_res: mcnemar_results_list.append(mcnemar_res)
        print("\n  1.2. Word Embedding Tabanlı (Temel-Word2Vec vs. Gelişmiş-FastText):")
        we_model_map = {"LinearSVM": ("LinearSVM", "LinearSVM"), "RandomForest": ("RandomForest", "RandomForest"), "NaiveBayes": ("NaiveBayes_Gaussian", "NaiveBayes_Gaussian")}
        for model_print_name, (model_temel_actual, model_gelismis_actual) in we_model_map.items():
            mcnemar_res = run_mcnemar_if_predictions_exist(all_predictions, ("WordEmbedding_Temel", model_temel_actual), ("WordEmbedding_Gelişmiş", model_gelismis_actual), f"{model_print_name} (W2V)", f"{model_print_name} (FT)", "WordEmbedding_Temel", "WordEmbedding_Gelişmiş")
            if mcnemar_res: mcnemar_results_list.append(mcnemar_res)
        print("\n--- Öncelik 2: Gelişmiş Setler Üzerinde Algoritma Karşılaştırmaları ---")
        print("\n  2.1. TFIDF_Gelişmiş Üzerinde Algoritmalar:")
        fs_tfidf_g = "TFIDF_Gelişmiş"
        models_on_tfidf_g = ["LinearSVM", "RandomForest", "NaiveBayes"]
        for i in range(len(models_on_tfidf_g)):
            for j in range(i + 1, len(models_on_tfidf_g)):
                mcnemar_res = run_mcnemar_if_predictions_exist(all_predictions, (fs_tfidf_g, models_on_tfidf_g[i]), (fs_tfidf_g, models_on_tfidf_g[j]), models_on_tfidf_g[i], models_on_tfidf_g[j], fs_tfidf_g)
                if mcnemar_res: mcnemar_results_list.append(mcnemar_res)
        print("\n  2.2. WordEmbedding_Gelişmiş Üzerinde Algoritmalar:")
        fs_we_g = "WordEmbedding_Gelişmiş"
        models_on_we_g = ["LinearSVM", "RandomForest", "NaiveBayes_Gaussian"] 
        for i in range(len(models_on_we_g)):
            for j in range(i + 1, len(models_on_we_g)):
                mcnemar_res = run_mcnemar_if_predictions_exist(all_predictions, (fs_we_g, models_on_we_g[i]), (fs_we_g, models_on_we_g[j]), models_on_we_g[i], models_on_we_g[j], fs_we_g)
                if mcnemar_res: mcnemar_results_list.append(mcnemar_res)
        print("\n--- Öncelik 3: En İyi TF-IDF (Gelişmiş) vs. En İyi Word Embedding (Gelişmiş) ---")
        if not df_performance.empty:
            best_tfidf_row = df_performance[df_performance["FeatureSet"] == "TFIDF_Gelişmiş"].nlargest(1, "F1_Score")
            best_we_row = df_performance[df_performance["FeatureSet"] == "WordEmbedding_Gelişmiş"].nlargest(1, "F1_Score")
            if not best_tfidf_row.empty and not best_we_row.empty:
                best_tfidf_model_name, best_we_model_name = best_tfidf_row["Model"].iloc[0], best_we_row["Model"].iloc[0]
                print(f"  Karşılaştırılıyor: En İyi TFIDF_Gelişmiş ({best_tfidf_model_name}) vs. En İyi WordEmbedding_Gelişmiş ({best_we_model_name})")
                mcnemar_res = run_mcnemar_if_predictions_exist(all_predictions, ("TFIDF_Gelişmiş", best_tfidf_model_name), ("WordEmbedding_Gelişmiş", best_we_model_name), best_tfidf_model_name, best_we_model_name, "TFIDF_Gelişmiş", "WordEmbedding_Gelişmiş")
                if mcnemar_res: mcnemar_results_list.append(mcnemar_res)
            else: print("    Uyarı: Öncelik 3 için en iyi TF-IDF veya Word Embedding konfigürasyonları bulunamadı.")
        else: print("    Uyarı: Öncelik 3 için performans verisi (df_performance) boş.")
        if mcnemar_results_list:
            df_mcnemar_results = pd.DataFrame(mcnemar_results_list)
            MCNEMAR_RESULTS_FILE = os.path.join(RESULTS_DIR, "mcnemar_test_karsilastirma_sonuclari.csv")
            df_mcnemar_results.to_csv(MCNEMAR_RESULTS_FILE, index=False)
            print(f"\nMcNemar testi sonuçları kaydedildi: {MCNEMAR_RESULTS_FILE}")
            print("\n--- McNemar Testi Özeti ---")
            print(df_mcnemar_results)
        else: print("\nHiç McNemar testi sonucu üretilmedi.")
    else:
        print("Hiçbir model başarıyla test edilemedi.")

if __name__ == "__main__":
    run_experiment()