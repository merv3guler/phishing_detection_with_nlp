import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def safe_str(val, maxlen=30):
    """
    Hücredeki değeri güvenli ve okunabilir bir şekilde stringe çevirir.
    Uzun metinleri kısaltır, özel karakterleri temizler.
    Parametreler:
        val: Herhangi bir değer.
        maxlen: Maksimum karakter uzunluğu.
    Dönüş:
        str: Temizlenmiş ve kısaltılmış string.
    """
    s = str(val)
    # Süslü parantez, ters slash, dolar, yüzde, alt çizgi, kare, &, ^, ~, yeni satır, tab vb. karakterleri kaldır
    s = re.sub(r'[\{\}\\\$%\_\#\&\^\~\n\r\t]', ' ', s)
    # Çoklu boşlukları teke indir
    s = re.sub(r'\s+', ' ', s)
    return s if len(s) <= maxlen else s[:maxlen] + "..."

# Veri setini yükle
df = pd.read_csv('data/CEAS_08.csv')

# Sonuçlar klasörü yoksa oluştur
os.makedirs('results', exist_ok=True)

# Rastgele örneklem seç
sample_df = df.sample(n=5, random_state=123)
table_data = [[safe_str(cell) for cell in row] for row in sample_df.values]

# Rastgele seçilen 5 satırın body, label ve urls sütunlarını Excel dosyasına aktar ve kaydet
sample_df[['body', 'label', 'urls']].to_excel('results/sample_rows.xlsx', index=False)

# Rastgele seçilen 5 satırı görselleştir
col_maxlens = [max([len(str(row[i])) for row in table_data] + [len(str(df.columns[i]))]) for i in range(len(df.columns))]
total_len = sum(col_maxlens)
col_widths = [max(0.08, l/total_len) for l in col_maxlens]  # minimum genişlik sınırı
fig, ax = plt.subplots(figsize=(min(1.5*len(df.columns), 18), 2.5))
ax.axis('off')
tbl = ax.table(cellText=table_data, colLabels=list(df.columns), loc='center', cellLoc='center', bbox=[0,0,1,1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.5)
# Sütun genişliklerini ayarla (i, 0) header ve (i, j) hücreler için
for i in range(len(df.columns)):
    for j in range(len(table_data)+1):  # +1 header için
        cell = tbl[(j, i)]
        cell.set_width(col_widths[i])
for key, cell in tbl.get_celld().items():
    cell.set_linewidth(0.5)
    cell.set_height(0.15)
#plt.title('Rastgele 5 Satır Örneği', y=1.12)
plt.tight_layout()
plt.savefig('results/sample_rows_table.png', dpi=200, bbox_inches='tight')
#plt.show() # Tabloyu görsel olarak göstermek için satırın başındaki # sembolünü kaldırabilirsiniz

# Veri setinin şeklini ve sütun isimlerini görselleştir
fig, ax = plt.subplots(figsize=(max(6, len(df.columns)*0.7), 2.5))
ax.axis('off')
rows, cols = df.shape
# Satır ve sütun bilgisini tek satırda, özel karakter veya kaçış karakteri olmadan yaz
shape_text = f"Satır sayısı: {rows} | Sütun sayısı: {cols}"
ax.text(0.5, 0.8, shape_text, fontsize=14, ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='gray', boxstyle='round,pad=0.7'))
# Sütun isimlerini alta ekle
colnames = ', '.join([str(c) for c in df.columns])
ax.text(0.5, 0.35, f"Sütunlar: {colnames}", fontsize=10, ha='center', va='center', wrap=True, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
#plt.title('Veri Setinin Şekli ve Sütunlar', y=1.15)
plt.tight_layout()
plt.savefig('results/dataset_shape_columns.png', dpi=200, bbox_inches='tight')
#plt.show() # Veri setinin şekli ve sütunlarını görsel olarak göstermek için satırın başındaki # sembolünü kaldırabilirsiniz

# Etiketlerin dağılımını bar grafiği olarak görselleştir (sayı ve yüzde ile birlikte)
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='label', hue='label', data=df, palette='pastel', legend=False)
#plt.title('Phishing ve Gerçek E-postaların Dağılımı')
plt.xlabel('Etiket')
plt.ylabel('Sayı')
max_count = max([p.get_height() for p in ax.patches])
for p in ax.patches:
    count = int(p.get_height())
    percentage = 100 * count / len(df)
    ax.annotate(f'{count}\n({percentage:.2f}%)', (p.get_x() + p.get_width() / 2., count), ha='center', va='bottom', fontsize=11)
ax.set_ylim(0, max_count * 1.20)
plt.tight_layout()
plt.savefig('results/label_distribution.png', dpi=200, bbox_inches='tight')
#plt.show() # Etiket özet tablosunu görsel olarak göstermek için satırın başındaki # sembolünü kaldırabilirsiniz

# Etiket özet tablosu oluştur
label_summary = df['label'].value_counts().rename_axis('Etiket').reset_index(name='Sayı')
label_summary['Yüzde'] = 100 * label_summary['Sayı'] / len(df)
label_summary = label_summary.set_index('Etiket')
label_summary['Yüzde'] = label_summary['Yüzde'].map('{:.2f}%'.format)

# Tüm sütunlar için eksik değer sayısı ve oranını bar grafiği ile göster
missing_counts = df.isnull().sum()
missing_percent = 100 * missing_counts / len(df)

fig, ax = plt.subplots(figsize=(max(6, len(df.columns)*1.2), 4))
bars = ax.bar(df.columns, missing_counts, color=sns.color_palette('pastel', len(df.columns)))
for i, (count, percent) in enumerate(zip(missing_counts, missing_percent)):
    ax.text(i, count + 0.1, f'{count} ({percent:.2f}%)', ha='center', va='bottom', fontsize=11)
ax.set_ylabel('Eksik Değer Sayısı')
#ax.set_title('Veri Setindeki Eksik Değerler')
plt.ylim(0, max(missing_counts)*1.2 + 1)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('results/missing_values.png', dpi=200, bbox_inches='tight')
#plt.show() # Eksik değer özet tablosunu görsel olarak göstermek için satırın başındaki # sembolünü kaldırabilirsiniz
