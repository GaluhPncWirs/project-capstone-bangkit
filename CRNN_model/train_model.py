import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, Dense, Bidirectional, LSTM, StringLookup, Dropout
import pandas as pd
import matplotlib.pyplot as plt

img_height = 35 
img_width = 250
vocab = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '(', ')', ' ', 'g', 'kkal', 'mcg', 'mg', 'ml', 'Lemak tidak jenuh tunggal', 'Tembaga', 'Seng', 'Vitamin B6', 'Natrium', 'Magnesium', 'Vitamin C', 'Asam linoleat', 'Vitamin B1', 'Kalsium', 'Kromium', 'Energi dari lemak', 'Jumlah Per Sajian', 'Vitamin B12', 'Energi', 'Asam a-linolenat', 'Vitamin A', 'Vitamin K', 'Vitamin B2', 'Kolin', 'Vitamin D', 'Vitamin B5', 'Karbohidrat total', 'Lemak trans', 'Folat', 'Biotin', 'Serat pangan', 'Lemak jenuh', 'Iodium', 'Kolesterol', 'Gula', 'Mangan', 'Selenium', 'Protein', 'Lemak total', 'Kalium', 'Vitamin E', 'Vitamin B3', 'Fosfor', 'Besi', 'Lemak tidak jenuh ganda'
]
data_ing = pd.read_csv("data_labels.csv")
vocab_size = len(vocab)
image_paths = data_ing['image_path'].to_list()
labels = data_ing['label'].to_list()
batch_size = 32

train_size = int(len(image_paths) * 0.8)

train_image_paths = image_paths[:train_size]
train_labels = labels[:train_size]
val_image_paths = image_paths[train_size:]
val_labels = labels[train_size:]


def preprocess_image_label(image_paths, labels, img_height, img_width, vocab):
    char_to_num = StringLookup(vocabulary=list(vocab), mask_token=None)
    images = []
    labels_encoded = []
    
    for path, label in zip(image_paths, labels):
        try:
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=1)
            img = tf.image.resize(img, [img_height, img_width])
            img = tf.image.convert_image_dtype(img, tf.float32) / 255.0
            images.append(img)
            label_encoded = char_to_num(tf.strings.unicode_split(label, 'UTF-8'))
            labels_encoded.append(label_encoded)
        except Exception as e:
            print(f"Error memproses gambar di path {path}: {e}")
    
    images = tf.convert_to_tensor(images)
    labels_encoded = tf.keras.preprocessing.sequence.pad_sequences(
        [label.numpy() for label in labels_encoded], padding="post"
    )
    labels_encoded = tf.convert_to_tensor(labels_encoded)
    return images, labels_encoded

def build_crnn_model(vocab_size, img_width=250, img_height=35):
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
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
)

train_images, train_labels = preprocess_image_label(
    train_image_paths, train_labels, img_height, img_width, vocab
)
val_images, val_labels = preprocess_image_label(
    val_image_paths, val_labels, img_height, img_width, vocab
)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=ctc_loss)


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
)


plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()