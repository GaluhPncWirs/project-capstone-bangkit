import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, Dense, Bidirectional, LSTM, Dropout, StringLookup
from tensorflow.keras.utils import Sequence
import numpy as np

# Parameter
img_height, img_width = 32, 128
vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '.', 'g', 'salt', 'fat', 'sugar', 'kcal']
vocab_size = len(vocab)
image_paths = ["data_sintetik/0.5_g_salt.png", "data_sintetik/3_g_fat.png", "data_sintetik/5_g_sugar.png", "data_sintetik/200_kcal.png"]
labels = ['0.5 g salt', '3 g fat', '5 g sugar', '200 kcal']
batch_size = 1

# Data Generator
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

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.image_paths))
        batch_paths = self.image_paths[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
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

# Fungsi Preprocessing Image
def preprosesing_image(img_path, img_height=32, img_width=128):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.image.convert_image_dtype(img, tf.float32) / 255.0
    return img

# Build Model
def build_crnn_model(vocab_size, img_width=128, img_height=32):
    input_img = tf.keras.layers.Input(shape=(img_height, img_width, 1), name="image", dtype='float32')
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    new_shape = ((img_width // 8), (img_height // 8) * 128)
    x = Reshape(target_shape=new_shape)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dense(64, activation="relu")(x)  # Layer tambahan
    
    x = Dropout(0.5)(x)
    x = Dense(vocab_size + 1, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=input_img, outputs=x, name="CRNN_OCR")
    return model


# CTC Loss
def ctc_loss(y_true, y_pred):
    input_length = tf.fill([tf.shape(y_pred)[0], 1], tf.shape(y_pred)[1])
    label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=1, keepdims=True)
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


# Model Initialization
model = build_crnn_model(vocab_size)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=ctc_loss)

# Data Preparation
train_data = OCRDataGenerator(image_paths, labels, batch_size=batch_size, vocab=vocab)
model.fit(train_data, epochs=100)

# Prediction Function
def predict_text_batch(model, image_paths, vocab):
    predictions = []
    for image_path in image_paths:
        img = preprosesing_image(image_path, img_height, img_width)
        img = tf.expand_dims(img, axis=0)
        pred = model.predict(img)

        # Hitung input_length dinamis berdasarkan output model
        input_len = np.ones((1,)) * pred.shape[1]
        
        # Decode dengan CTC decode
        decoded, _ = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)

        # Konversi hasil decode ke teks
        pred_text = ''.join([vocab[i] for i in decoded[0][0].numpy() if i < len(vocab)])

        # Tambahkan hasil prediksi ke daftar
        predictions.append(pred_text)
        
        # Debugging untuk melihat hasil tiap prediksi
        print(f"Image Path: {image_path}, Predicted Shape: {pred.shape}, Predicted Text: {pred_text}")
    
    return predictions

# Memanggil fungsi prediksi batch
predicted_texts = predict_text_batch(model, image_paths, vocab)
print("Hasil Prediksi:", predicted_texts)


