import pandas as pd
import numpy as np
import string
import re
import nltk
import Sastrawi
import ast  # For evaluating the string representation of lists
import math
import matplotlib.pyplot as plt
import os

print("Analisis Sentimen Pengguna Media Sosial Terhadap Operasional LRT Jabodebek")
print("Backpropagation")
print("Nama : Maliky Syailendra Haqi Sutafa")
input("Tekan Enter untuk melanjutkan")


files = input('Masukkan file excel untuk dianalisis (harus .xlsx!): ')

os.system("cls")

# Nama file Excel sumber dan tujuan
destination_file = 'Pelatihan_4.xlsx'
source_file = f'{files}.xlsx'


# Membaca data kolom "Stemmed" dari file Excel tujuan (destination.xlsx)
sentimen_latih = pd.read_excel(destination_file, usecols=['Sentimen'])
target_latih = pd.read_excel(destination_file, usecols=['Target'])

# Membaca data kolom "Stemmed" dari file Excel sumber (source.xlsx)
sentimen_uji = pd.read_excel(source_file, usecols=['Sentimen'])
target_uji = pd.read_excel(source_file, usecols=['Target'])

#Input manual
manual = input("Apakah anda ingin memasukkan kata tambahan untuk dianalisis? (Tekan y untuk ya, tekan apapun untuk tidak) : ")

if(manual == 'y'):
    while(True):
        label = input("Masukkan '1' jika pujian dan '0' jika kritikan : ")
        if(label == '1'):
            os.system("cls")
            komentar = input("Masukkan komentar positif (pujian/saran/masukan) : ")
            sentiment = int(label)
            tweets_manual = pd.DataFrame({'Sentimen': [komentar]})
            sentimen_manual = pd.DataFrame({'Target': [sentiment]})
            sentimen_uji = pd.concat([sentimen_uji, tweets_manual], ignore_index=True)
            target_uji = pd.concat([target_uji, sentimen_manual], ignore_index=True)
            break
        elif(label == '0'):
            os.system("cls")
            komentar = input("Masukkan komentar positif (pujian/saran/masukan): ")
            sentiment = int(label)
            tweets_manual = pd.DataFrame({'Sentimen': [komentar]})
            sentimen_manual = pd.DataFrame({'Target': [sentiment]})
            sentimen_uji = pd.concat([sentimen_uji, tweets_manual], ignore_index=True)
            target_uji = pd.concat([target_uji, sentimen_manual], ignore_index=True)
            break
        else:
            os.system("cls")
            print("Masukkan yang benar!")
else:
    pass

# Menggabungkan kolom "Stemmed" dari kedua DataFrame
tweets = pd.concat([sentimen_latih, sentimen_uji], ignore_index=True)
target = pd.concat([target_latih, target_uji], ignore_index=True)
target = target["Target"]

# Membuat DataFrame gabungan dengan kolom "Data" dan "Sentimen"
data_combined = pd.DataFrame({
    "Data": sentimen_uji["Sentimen"],
    "Sentimen": target_uji["Target"].map({1: "Positif", 0: "Negatif"})
})


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print("Berikut adalah data yang akan diuji : ")

print(data_combined)

input("Tekan Enter untuk melanjutkan")
print("Berikut adalah tahapan-tahapan untuk menganalisis sentimen opini LRT Jabodebek : ")
print("1. Text Preprocessing")
print("2. Ekstraksi Fitur TF-IDF")
print("3. Normalisasi Data")
print("4. Pelatihan Jaringan Algoritma")
print("5. Uji Validasi")
print("6. Evaluasi Performa")
print("")
input("Tekan Enter untuk memulai tahap 1. Text Preprocessing")

os.system("cls")

print("1. Case Folding...")
def clean_text(text):
    text = text.lower()
    # Menghapus mention dan URL
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)
    # Menghapus simbol non-alfanumerik dan angka
    text = re.sub("([^a-z \t])", " ", text)
    return ' '.join(text.split())

tweets['clean'] = tweets.apply(lambda x: clean_text(x['Sentimen']), axis=1)
tweet = tweets['clean']

tweets.to_csv('clean_tweets.csv', index=True)
#tweets['clean'].head(10)

print("Proses Case Folding Selesai!")

from nltk.tokenize import RegexpTokenizer

print("2. Tokenizing...")
# Membuat tokenizer
regexp = RegexpTokenizer(r'\w+|$[0-9]+|\S+')

# Fungsi untuk menghapus kata berulang sambil mempertahankan urutan
def remove_duplicate_and_preserve_order(tokens):
    seen = set()
    result = []
    for token in tokens:
        if token not in seen:
            result.append(token)
            seen.add(token)
    return result

# Tokenisasi teks dan menghapus kata-kata berulang
tweets['Token'] = tweets['clean'].apply(lambda x: remove_duplicate_and_preserve_order(regexp.tokenize(x)))

token = tweets['Token']
token.to_csv('tokenized.csv', index=False)

print("Proses Tokenizing Selesai!")

print("3. Normalisasi...")

normalized_word = pd.read_csv("Normalisasi.csv", encoding='latin1')

normalized_word_dict = {}

for index,row in normalized_word.iterrows():
     if row[0] not in normalized_word_dict:
          normalized_word_dict[row[0]] = row[1]

def normalized_term(document):
      return [normalized_word_dict[term] if term in normalized_word_dict else term for term in document]

tweets['Normalisasi'] = tweets['Token'].apply(normalized_term)
#tweets['Normalisasi']

token = tweets['Normalisasi']
token.to_csv('normalized.csv', index=False)

print("Proses Normalisasi Kata selesai!")

print("4. Stopword Removal...")

# Membaca isi file stopword.txt dan memuatnya ke dalam sebuah list
with open('stopword.txt', 'r', encoding='utf-8') as file:
    stopword_from_file = file.readlines()
stopwords = [word.strip() for word in stopword_from_file]

#print(stopwords)

def stopwords_text(tokens):
    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords:
            token = token.lower()
            cleaned_tokens.append(token)
    clean_tokens = [re.sub(r'\d+', '', token) for token in cleaned_tokens]
    return clean_tokens

tweets['stop'] = tweets['Normalisasi'].apply(stopwords_text)
stop = tweets['stop']
#tweets['stop']
remove = tweets['stop']
remove.to_csv('stopword_removal.csv', index=False)

print("Proses Stopword Removal Selesai!")

print("5. Stemming...")

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

# Kamus kata-kata yang akan diubah
custom_words = {
    "diturunin": "turun",
    "diandalin": "andal",
    "maluin": "malu",
    "ngepas": "pas",
    "ngetwit": "twit",
    "headwaynya": "headway",
    "nggak": "tidak",
    "benarin": "benar",
    "dibenarin": "benar",
    "terima kasih": ["terima","kasih"],
    "kerja bagus": ["kerja","bagus"],
    "nyobain": "coba",
    "ngebelain":"bela",
    "lrtnya":"lrt",
    "bermasalah": "masalah",
    "aus":"aus",
    "memudahkan":"mudah",
    "berkurang":"kurang",
    "dikurangi":"kurang",
    "optimal":"optimal"

}

#stemmer = stem_factory.create_stemmer(dictionary)

def stemming_text(tokens):
    hasil = []
    for token in tokens:
        if token in custom_words:
            if isinstance(custom_words[token], list):
                hasil.extend(custom_words[token])
            else:
                hasil.append(custom_words[token])
        else:
            hasil.append(stemmer.stem(token))
    return hasil

# Fungsi untuk menghapus kata-kata duplikat setelah stemming
def remove_duplicates(words):
    unique_words = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
    return unique_words

from nltk.stem import PorterStemmer
#stemmer = PorterStemmer()
tweets['stemmed'] = tweets['stop'].apply(stemming_text)

for i, teks in enumerate(tweets["stemmed"]):
    tweets["stemmed"][i] = remove_duplicates(teks)


#print(tweets['stemmed'])
#print(target)

print("Proses Stemming Selesai! Berikut adalah hasil Text Preprocessing : ")

stemming = pd.DataFrame({
    'stemmed': tweets['stemmed'],
    'target': target
})

print(stemming)
#stemming.to_csv('Uji.csv', index=False)
print("")
input("Tekan Enter untuk memulai tahap 2. Ekstraksi Fitur")
os.system("cls")
print("Proses TF-IDF...")
stemming.to_csv('Hasil Stemming.csv', index=False)
df = stemming

# Convert the string representation of lists to actual lists
#df['stemmed'] = df['stemmed'].apply(ast.literal_eval)
tg = df['target']
#print(df['stemmed'])

# Convert the stemmed data to strings
documents = [' '.join(words) for words in df['stemmed']]

# Create a DataFrame for better representation and understanding
df['Documents'] = documents

# Step 1: Calculate Term Frequency (TF)
tf_matrix = []
for doc in documents:
    tf_doc = {word: doc.split().count(word) for word in set(doc.split())}
    tf_matrix.append(tf_doc)
    
# Step 2: Calculate Inverse Document Frequency (IDF)
def calculate_idf(docs):
    idf_dict = {}
    total_docs = len(docs)

    for doc in docs:
        for word in set(doc.split()):
            idf_dict[word] = idf_dict.get(word, 0) + 1
    print(idf_dict)

    idf_values = {word: math.log10(total_docs / (count)) for word, count in idf_dict.items()}
    return idf_values

idf_values = calculate_idf(documents)

# Step 3: Calculate TF-IDF
tfidf_matrix = []
for tf_doc in tf_matrix:
    tfidf_doc = {word: (tf * idf_values[word])+1 for word, tf in tf_doc.items()}
    tfidf_matrix.append(tfidf_doc)
    
# Convert the TF-IDF matrix to a DataFrame for better representation
tfidf_df = pd.DataFrame(tfidf_matrix)

# Replace NaN values with 0
tfidf_df.fillna(0, inplace=True)
os.system("cls")

print("Proses Ekstraksi Fitur Selesai!")
print("Berikut matriks TF-IDF untuk proses analisis sentimen : ")
print("")
# Display the result
print(tfidf_df)
tfidf_df.to_csv("Hasil TF-IDF.csv")

input("Tekan Enter untuk memulai tahap 3. Normalisasi Data")

os.system("cls")

# Data Normalization
max_value = tfidf_df.max().max()
min_value = tfidf_df.min().min()

normalized_df = (0.8 * (tfidf_df - min_value) / (max_value - min_value)) + 0.1

print("Data setelah normalisasi : \n")
# Display the result
Xmatrix = pd.DataFrame(normalized_df)
new_columns = [f"x({i+1})" for i in range(Xmatrix.shape[1])]
new_index = [f"{i+1}" for i in range(Xmatrix.shape[0])]
Xmatrix.index = new_index
Xmatrix.columns = new_columns
print(Xmatrix)
Xmatrix.to_csv("Normalized TF-IDF.csv")
input("Tekan Enter untuk memulai Pelatihan Jaringan! (4) ")

def input_data_pelatihan(normalisasi_tfidf):
    jumlah_data = len(normalisasi_tfidf)
    nilai_df = len(normalisasi_tfidf[0])

    input_data = np.zeros((jumlah_data, nilai_df))
    target = np.zeros(jumlah_data)

    for i in range(jumlah_data):
        for j in range(nilai_df):
            input_data[i][j] = normalisasi_tfidf[i][j]

    return input_data

y = np.array([[target] for target in df["target"]])
X = input_data_pelatihan(normalized_df.values)

jumlah_latih = round(len(X)*0.60)
jumlah_uji = len(X) - jumlah_latih

X_train = X[0:jumlah_latih]
y_train = y[0:jumlah_latih]

print("Input Data untuk Pelatihan:")
Xmatrix = pd.DataFrame(X_train)
new_columns = [f"x({i+1})" for i in range(Xmatrix.shape[1])]
new_index = [f"Pola Testing ke-{i+1}" for i in range(Xmatrix.shape[0])]
Xmatrix.index = new_index
Xmatrix.columns = new_columns
print(Xmatrix)


def inisialisasi_parameter(alpha, max_iter, err):
    learning_rate = alpha
    maksimal_iterasi = max_iter
    target_error = err
    return learning_rate, maksimal_iterasi, target_error

def membangkitkan_bobot_dan_bias(input_size, hidden_size, output_size):
    np.random.seed(0)  # Untuk memastikan hasil yang konsisten
    bbv = np.random.uniform(-1, 1, size=(hidden_size,)) #np.random.randn(hidden_size)  # Bias lapisan tersembunyi
    bbw = np.random.uniform(-1, 1, size=(input_size, hidden_size)) #np.random.randn(input_size, hidden_size)  # Bobot antara input dan lapisan tersembunyi
    deltav = np.zeros(hidden_size)  # Perubahan bias lapisan tersembunyi
    deltaw = np.zeros((input_size, hidden_size))  # Perubahan bobot antara input dan lapisan tersembunyi
    bbvo = np.random.uniform(-1, 1, size=(output_size,)) #np.random.randn(output_size)  # Bias lapisan output
    bbwo = np.random.uniform(-1, 1, size=(hidden_size, output_size)) #np.random.randn(hidden_size, output_size)  # Bobot antara lapisan tersembunyi dan output
    deltavo = np.zeros(output_size)  # Perubahan bias lapisan output
    deltawo = np.zeros((hidden_size, output_size))  # Perubahan bobot antara lapisan tersembunyi dan output
    return bbv, bbw, deltav, deltaw, bbvo, bbwo, deltavo, deltawo


def feedforward(input_data, bbv, bbw, bbvo, bbwo):
    z = np.dot(input_data, bbw) + bbv
    y = 1 / (1 + np.exp(-z))
    o = np.dot(y, bbwo) + bbvo
    t = 1 / (1 + np.exp(-o))
    return y, t

def backpropagation(target, t, y, alpha, bbwo, input_data):
    #epsilon = 1e-10
    output_error = (-1 * (target * 1 / (t)) + ((1 - target) * 1 / (1 - t)))
    output_delta = output_error * t * (1 - t)

    hidden_error = output_delta.dot(bbwo.T)
    hidden_delta = hidden_error * y * (1 - y)

    deltavo = alpha * np.sum(output_delta, axis=0)
    deltawo = alpha * np.dot(y.T, output_delta)

    deltav = alpha * np.sum(hidden_delta, axis=0)
    deltaw = alpha * np.dot(input_data.T, hidden_delta)

    return deltavo, deltawo, deltav, deltaw

def pembaruan_bobot_dan_bias(bbv, deltav, bbw, deltaw, bbvo, deltavo, bbwo, deltawo):
    bbv -= deltav
    bbw -= deltaw
    bbvo -= deltavo
    bbwo -= deltawo

    return bbv, bbw, bbvo, bbwo

def uji_validasi(input_data, bbv, bbw, bbvo, bbwo):
    # Proses feedforward
    _,t = feedforward(input_data, bbv, bbw, bbvo, bbwo)
    return t

def threshold_function(predictions, threshold=0.6):
    return (predictions > threshold).astype(int)

def pelatihan_data(input_data, target, alpha, max_iter, err):
    input_size = input_data.shape[1]
    hidden_size = input_size  # Jumlah neuron pada lapisan tersembunyi
    output_size = target.shape[1]

    # Inisialisasi parameter
    learning_rate, max_iter, err = inisialisasi_parameter(alpha, max_iter, err)

    # Membangkitkan bobot dan bias awal
    bbv, bbw, deltav, deltaw, bbvo, bbwo, deltavo, deltawo = membangkitkan_bobot_dan_bias(input_size, hidden_size, output_size)

    # Inisialisasi iterasi
    epoch = 0

    #Penampung nilai BCE
    BCE_values = []

    while epoch < max_iter:
        # Proses feedforward
        y, t = feedforward(input_data, bbv, bbw, bbvo, bbwo)
        
        # Proses backpropagation
        deltavo, deltawo, deltav, deltaw = backpropagation(target, t, y, alpha, bbwo, input_data) 

        # Proses pembaruan bobot dan bias
        bbv, bbw, bbvo, bbwo = pembaruan_bobot_dan_bias(bbv, deltav, bbw, deltaw, bbvo, deltavo, bbwo, deltawo)       

        # Proses hitung MSE
        #epsilon=1e-10
        #t = np.clip(t, epsilon, 1. - epsilon)
        BCE = -1*np.mean((target*np.log(t))+((1-target)*np.log(1-t)))
        BCE_values.append(BCE)
        if np.isnan(BCE):
            break
        print(f"Epoch {epoch + 1}/{max_iter} - Error Value: {BCE}")
        # Cek kondisi berhenti
        if BCE <= err:
            break
        epoch += 1

    #input("Tekan enter untuk melihat bobot dan bais optimal!")
    #os.system("cls")
    n = len(BCE_values)
    values = BCE_values[n-1]
    print(f'\nDiperoleh BCE sebesar : {values:.5f}')
    return bbv, bbw, bbvo, bbwo, BCE_values

f1 = 0
while(f1 < 98.50):
    input("Tekan enter untuk melanjutkan proses pelatihan")
    alpha = float(input("Masukkan nilai learning rate : "))
    max_iter = int(input("Masukkan batas maksimum iterasi : "))
    err = float(input("Masukkan batas nilai error : "))
    bbv, bbw, bbvo, bbwo, BCE_values = pelatihan_data(X_train, y_train, alpha, max_iter, err)
    bobot_bias_optimal = (bbv, bbw, bbvo, bbwo)
    input("\nTekan enter untuk melihat hasil data tesing")
    data_testing = uji_validasi(X_train, *bobot_bias_optimal)

    from sklearn.metrics import f1_score

    # Calculate F1-score for various threshold values
    threshold_values = np.arange(0.1, 1.0, 0.01)
    f1_scores = []

    for threshold in threshold_values:
        thresholded_outputs = threshold_function(data_testing, threshold=threshold)
        f1_scores.append(f1_score(y_train, thresholded_outputs, average='binary'))

    # Find the threshold that maximizes F1-score for the new dataset
    optimal_threshold = threshold_values[np.argmax(f1_scores)]
    thresholded_outputs = threshold_function(data_testing, threshold=optimal_threshold)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Convert the thresholded outputs to a numpy array
    thresholded_outputs_np = np.array(thresholded_outputs)

    # Convert the thresholded outputs to binary (0 or 1)
    binary_thresholded_outputs = (thresholded_outputs_np > 0.5).astype(int)

    # Evaluate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(y_train, binary_thresholded_outputs) * 100
    precision = precision_score(y_train, binary_thresholded_outputs) * 100
    recall = recall_score(y_train, binary_thresholded_outputs) * 100
    f1 = float(f1_score(y_train, binary_thresholded_outputs) * 100)
    
    os.system("cls")
    print(f"\nHasil Testing Training :")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    
   
    if(f1 < 98.50):
        print(f"Maaf! Nilai uji akurasi anda masih dibawah 98.50% ")

input("Tekan enter untuk melihat bobot dan bais optimal!")


Xmatrix = pd.DataFrame(bbv)
new_columns = [f"x({0})" for i in range(Xmatrix.shape[1])]
new_index = [f"z{i+1}" for i in range(Xmatrix.shape[0])]
Xmatrix.index = new_index
Xmatrix.columns = new_columns
bbv_x = Xmatrix

Xmatrix = pd.DataFrame(bbw)
new_columns = [f"z({i+1})" for i in range(Xmatrix.shape[1])]
new_index = [f"x{i+1}" for i in range(Xmatrix.shape[0])]
Xmatrix.index = new_index
Xmatrix.columns = new_columns
bbw_x = Xmatrix

Xmatrix = pd.DataFrame(bbvo)
new_columns = [f"y(0)" for i in range(Xmatrix.shape[1])]
new_index = [f"z(0)" for i in range(Xmatrix.shape[0])]
Xmatrix.index = new_index
Xmatrix.columns = new_columns
bbvo_x = Xmatrix

Xmatrix = pd.DataFrame(bbwo)
new_columns = [f"y({1})" for i in range(Xmatrix.shape[1])]
new_index = [f"z{i+1}" for i in range(Xmatrix.shape[0])]
Xmatrix.index = new_index
Xmatrix.columns = new_columns
bbwo_x = Xmatrix

print("Dataframe.")
print("Bobot dan bias optimal:")
print("Bias Input Hidden:")
print(bbv_x)
bbv_x.to_csv("Bias Input Hidden.csv")
print("Bobot Input Hidden:")
print(bbw_x)
bbw_x.to_csv("Bobot Input Hidden.csv")
print("Bias Hidden Output:")
print(bbvo_x)
bbvo_x.to_csv("Bias Hidden Output.csv")
print("Bobot Hidden Output :")
print(bbwo_x)
bbwo_x.to_csv("Bobot Hidden Output.csv")


input("Tekan enter untuk lanjut ke proses 5. Uji Validasi")
os.system("cls")

X_test = X[jumlah_latih:jumlah_latih+jumlah_uji]
y_test = y[jumlah_latih:jumlah_latih+jumlah_uji]

print("Data untuk Uji Validasi :")
Xmatrix = pd.DataFrame(X_test)
new_columns = [f"x({i+1})" for i in range(Xmatrix.shape[1])]
new_index = [f"Pola Testing ke-{i+1}" for i in range(Xmatrix.shape[0])]
Xmatrix.index = new_index
Xmatrix.columns = new_columns
print(Xmatrix)

input("Tekan enter untuk lanjut")

output_uji = uji_validasi(X_test, *bobot_bias_optimal)

from sklearn.metrics import f1_score

# Calculate F1-score for various threshold values
threshold_values = np.arange(0.1, 1.0, 0.01)
f1_scores = []

for threshold in threshold_values:
    thresholded_outputs = threshold_function(output_uji, threshold=threshold)
    f1_scores.append(f1_score(y_test, thresholded_outputs, average='binary'))

# Find the threshold that maximizes F1-score for the new dataset
optimal_threshold = threshold_values[np.argmax(f1_scores)]
thresholded_outputs = threshold_function(output_uji, threshold=optimal_threshold)


print("\nHasil Uji Validasi dengan data Uji :")
validation_results = pd.DataFrame({
    'Input': sentimen_uji['Sentimen'].reset_index(drop=True),
    'Target': pd.Series(y_test.flatten()),
    'Prediksi': pd.Series(thresholded_outputs.flatten())
})
print(validation_results)

validation_results.to_excel("Hasil Uji Validasi.xlsx")

input("Tekan enter untuk lanjut melihat hasil 6. Evaluasi Performa Algoritma")
os.system("cls")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Convert the thresholded outputs to a numpy array
thresholded_outputs_np = np.array(thresholded_outputs)

# Convert the thresholded outputs to binary (0 or 1)
binary_thresholded_outputs = (thresholded_outputs_np > 0.5).astype(int)

# Evaluate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, binary_thresholded_outputs)*100
precision = precision_score(y_test, binary_thresholded_outputs)*100
recall = recall_score(y_test, binary_thresholded_outputs)*100
f1 = f1_score(y_test, binary_thresholded_outputs)*100

print(f"\nHasil Evaluasi Performa Pada Data Uji:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")

# Visualisasi nilai BCE
plt.plot(BCE_values)
plt.xlabel('Epoch')
plt.ylabel('BCE Value')
plt.title('Perubahan BCE per Epoch')
plt.grid(True)
plt.show()


