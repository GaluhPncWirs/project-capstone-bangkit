import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, Dense, Bidirectional, LSTM, StringLookup, Dropout
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd


img_height = 32 
img_width = 128
vocab = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 
    'kcal', 'g', 'mg', 'sugar', 'protein', 'fiber', 'sodium', 'carbs', 
    'salt', 'fat', 'iron', 'potassium', 'calcium'
]
data_ing = pd.read_csv("labels.csv")
vocab_size = len(vocab)
image_paths = data_ing['image_path'].to_list()
labels = data_ing['label'].to_list()
batch_size = 8

class OCRDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=32, img_height=32, img_width=128, vocab=None):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.vocab = vocab
        self.char_to_num = StringLookup(vocabulary=list(vocab), mask_token=None)

    def __len__(self):
        return (len(self.image_paths) + self.batch_size - 1) // self.batch_size
        # return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.image_paths))
        # batch_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        # batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        if start_idx >= len(self.image_paths):
            raise ValueError("Batch kosong terdeteksi pada index:", idx)
        batch_paths = self.image_paths[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]

        # print(f"Batch Paths: {batch_paths}, Batch Labels: {batch_labels}")

        batch_images = []
        batch_labels_encoded = []
        
        for path, label in zip(batch_paths, batch_labels):
            img = preprosesing_image(path, self.img_height, self.img_width)
            batch_images.append(img)
            label_encoded = self.char_to_num(tf.strings.unicode_split(label, 'UTF-8')).numpy()
            batch_labels_encoded.append(label_encoded)

        batch_images = np.array(batch_images)
        batch_labels = tf.keras.preprocessing.sequence.pad_sequences(batch_labels_encoded, padding="post")

        return batch_images, batch_labels

class CTCCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, vocab):
        super().__init__()
        self.validation_data = validation_data
        self.vocab = vocab

    def decode_batch_predictions(self, preds):
        input_len = np.ones(preds.shape[0]) * preds.shape[1]
        results, _ = tf.keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)
        output_text = []
        for result in results[0]:
            text = ''.join([self.vocab[int(i)] for i in result if int(i) < len(self.vocab)])
            output_text.append(text)
        return output_text

    def on_epoch_end(self, epoch, logs=None):
        batch_images, batch_labels = next(iter(self.validation_data))
        preds = self.model.predict(batch_images)
        pred_texts = self.decode_batch_predictions(preds)
        accuracy = sum([pred_text == true_text for pred_text, true_text in zip(pred_texts, labels)]) / len(labels)
        print(f" - val_accuracy: {accuracy:.4f}")

    # def on_epoch_end(self, epoch, logs=None):
    #     # Mengambil batch validasi pertama untuk evaluasi
    #     batch_images, batch_labels = next(iter(self.validation_data))
        
    #     # Prediksi hasil dari model
    #     preds = self.model.predict(batch_images)
    #     pred_texts = self.decode_batch_predictions(preds)
        
    #     # Konversi batch_labels menjadi teks untuk perbandingan
    #     true_texts = []
    #     for label in batch_labels:
    #         text = ''.join([self.vocab[int(i)] for i in label if int(i) < len(self.vocab) and i != 0])
    #         true_texts.append(text)
        
    #     # Hitung akurasi
    #     correct = sum([pred == true for pred, true in zip(pred_texts, true_texts)])
    #     accuracy = correct / len(true_texts)
    #     print(f" - val_accuracy: {accuracy:.4f}")


def preprosesing_image(img_path, img_height=32, img_width=128):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img / 255.0
    return img


def build_crnn_model(vocab_size, img_width=128, img_height=32):
    input_img = tf.keras.layers.Input(shape=(img_height, img_width, 1), name="image", dtype='float32')
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    new_shape = ((img_width // 8), (img_height // 8) * 256)
    x = Reshape(target_shape=new_shape)(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Dense(vocab_size + 1, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    model = tf.keras.Model(inputs=input_img, outputs=x, name="CRNN_OCR")
    return model

def ctc_loss(y_true, y_pred):
    input_length = tf.fill([tf.shape(y_pred)[0], 1], tf.shape(y_pred)[1])
    label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=1, keepdims=True)
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

model = build_crnn_model(vocab_size)
# model.summary()
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=ctc_loss)

train_data = OCRDataGenerator(image_paths[:80], labels[:80], batch_size=batch_size, vocab=vocab)
validation_data = OCRDataGenerator(image_paths[80:], labels[80:], batch_size=batch_size, vocab=vocab)
accuracy_callback = CTCCallback(validation_data=validation_data, vocab=vocab)
# images, labels = train_data[0]
# print("Images shape:", images.shape)
# print("Labels shape:", labels.shape)
# callbacks = [
#     EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
#     ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3)
# ]

model.fit(
    train_data,
    validation_data=validation_data,
    epochs=30,
    callbacks=[accuracy_callback]
)


# def clean_predic(prediction):
#     tokens = prediction.split('calcium')
#     clean_text = 'calcium'.join(dict.fromkeys(tokens))
#     return clean_text

# def predict_text(model, image_paths, vocab):
#     prediksi = []
#     for imagePath in image_paths:
#         img = preprosesing_image(imagePath, img_height, img_width)
#         img = tf.expand_dims(img, axis=0)
#         pred = model.predict(img)

#         input_len = np.ones((1,)) * pred.shape[1]
#         decoded, _ = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
#         raw_pred = [vocab[i] for i in decoded[0][0].numpy() if i < len(vocab)]
#         pred_text = []
#         prev_char = None
#         for char in raw_pred:
#             if char != prev_char:
#                 pred_text.append(char)
#                 prev_char = char
#         prediksi.append(''.join(pred_text))
#     return prediksi

# # Contoh prediksi
# predicted_text = predict_text(model, image_paths, vocab)
# print("untuk prediksi text ", predicted_text)

# def predict_text_with_top5(model, image_paths, vocab):
#     predictions = []
#     for image_path in image_paths:
#         img = preprosesing_image(image_path, img_height, img_width)
#         img = tf.expand_dims(img, axis=0)
#         pred = model.predict(img)
#         print("Prediksi bentuk:", pred.shape)
        
#         for timestep in range(pred.shape[1]):
#             top5_indices = np.argsort(pred[0, timestep, :])[-5:]
#             top5_probs = pred[0, timestep, top5_indices]
#             print(f"Timestep {timestep}: Top 5 index {top5_indices}, Probabilitas: {top5_probs}")
        
#         input_len = np.ones((1,)) * pred.shape[1]
#         decoded, _ = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
#         pred_text = ''.join([vocab[i] for i in decoded[0][0].numpy() if i < len(vocab)])
#         predictions.append(pred_text)
    
#     return predictions

# # Coba gunakan fungsi ini untuk meninjau hasil prediksi top-5
# predicted_texts = predict_text_with_top5(model, image_paths, vocab)
# print("untuk prediksi top 5 ", predicted_texts)

# def calculate_accuracy(model, data_generator):
#     correct = 0
#     total = 0

#     # Iterasi batch pada data generator
#     for batch_images, batch_labels in data_generator:
#         # Lewati batch yang kosong
#         if len(batch_images) == 0 or len(batch_labels) == 0:
#             print("Batch kosong terdeteksi. Melompati batch ini.")
#             continue

#         # Dapatkan prediksi model
#         y_pred = model.predict(batch_images)
        
#         # Decode prediksi dengan CTC
#         input_length = np.ones(y_pred.shape[0]) * y_pred.shape[1]
#         decoded_predictions = tf.keras.backend.ctc_decode(y_pred, input_length=input_length, greedy=True)[0][0]

#         # Bandingkan prediksi dengan label asli
#         for i in range(len(batch_labels)):
#             if i >= decoded_predictions.shape[0]:
#                 continue  # Lewati jika indeks prediksi di luar jangkauan

#             # Konversi indeks prediksi dan label asli ke teks
#             pred_text = ''.join([vocab[index] for index in decoded_predictions[i].numpy() if index != -1])
#             true_text = ''.join([vocab[index] for index in batch_labels[i] if index != 0])

#             if pred_text == true_text:
#                 correct += 1
#             total += 1

#     # Menghindari pembagian dengan nol
#     accuracy = correct / total if total > 0 else 0
#     return accuracy

# # Panggil fungsi calculate_accuracy
# accuracy = calculate_accuracy(model, train_data)
# print(f"Accuracy: {accuracy * 100:.2f}%")




# class CustomDataset(tf.keras.utils.Sequence):
#     def __init__(self, data, labels, batch_size=32, **kwargs):
#         super().__init__(**kwargs)
#         self.data = data
#         self.labels = labels
#         self.batch_size = batch_size

#     def __len__(self):
#         return len(self.data) // self.batch_size

#     def __getitem__(self, idx):
#         batch_x = self.data[idx * self.batch_size: (idx + 1) * self.batch_size]
#         batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
#         return batch_x, batch_y

# class PredictionLogger(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         sample_image, sample_label = next(iter(dataset))
#         prediction = model.predict(sample_image)
#         print(f"Predicted: {prediction}, True Label: {sample_label}")

# # 1. Fungsi untuk memuat dan memproses gambar
# def process_image(image_path):
#     image = tf.io.read_file(image_path)               # Membaca gambar dari file
#     image = tf.image.decode_png(image, channels=1)    # Mendekode gambar PNG
#     image = tf.image.resize(image, [img_height, img_width])  # Ubah ukuran gambar
#     image = tf.cast(image, tf.float32) / 255.0        # Normalisasi ke rentang [0, 1]
#     return image

# # 2. Membuat StringLookup untuk konversi label ke numerik
# string_lookup = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token='')
# max_label_length = 10

# # 3. Fungsi untuk memproses label menjadi token numerik
# def process_label(label):
#     chars = tf.strings.bytes_split(label)                 # Memisahkan setiap karakter
#     label_ids = string_lookup(chars)                     # Konversi karakter ke ID numerik
#     # label_ids = tf.pad(label_ids, [[0, max_label_length - tf.shape(label_ids)[0]]], constant_values=0)
#     return label_ids

# # 4. Fungsi untuk menggabungkan proses gambar dan label
# def load_and_preprocess(image_path, label):
#     image = process_image(image_path)
#     label = process_label(label)
#     return image, label

# # 5. Membuat dataset dari paths dan labels
# dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
# dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# def pad_labels(image, label):
#     label = tf.cond(
#         tf.shape(label)[0] < max_label_length,
#         lambda: tf.pad(label, [[0, max_label_length - tf.shape(label)[0]]]),
#         lambda: label[:max_label_length] 
#     )
#     return image, label

# dataset = dataset.map(pad_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# # Melatih model
# model.fit(dataset, epochs=10)