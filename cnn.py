import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Bidirectional, LSTM, StringLookup, Dropout, BatchNormalization
from tensorflow.keras.models import Model
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
batch_size = 128

train_size = int(len(image_paths) * 0.8)

train_image_paths = image_paths[:train_size]
train_labels = labels[:train_size]
val_image_paths = image_paths[train_size:]
val_labels = labels[train_size:]


def preprocess_data(image_paths, labels, img_height, img_width, vocab):
    char_to_num = StringLookup(vocabulary=list(vocab), mask_token=None)
    images, label_sequences = [], []

    for path, label in zip(image_paths, labels):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [img_height, img_width])
        img = tf.image.convert_image_dtype(img, tf.float32)
        images.append(img)

        label_sequence = char_to_num(tf.strings.unicode_split(label, 'UTF-8'))
        label_sequences.append(label_sequence)

    padded_labels = tf.keras.preprocessing.sequence.pad_sequences(
        [label.numpy() for label in label_sequences], padding='post', value=-1
    )
    return tf.convert_to_tensor(images), tf.convert_to_tensor(padded_labels)

def build_crnn_model(vocab_size, img_width=250, img_height=35):
    input_img = Input(shape=(img_height, img_width, 1), name='image', dtype='float32')

    # CNN layers
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Reshape for RNN layers
    x = Reshape(target_shape=((img_width // 8), (img_height // 8) * 256))(x)

    # RNN layers
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)

    # Fully connected layer
    x = Dense(vocab_size + 1, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=x, name='CRNN')
    return model

def ctc_loss(y_true, y_pred):
    input_length = tf.fill([tf.shape(y_pred)[0], 1], tf.shape(y_pred)[1])
    label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=1, keepdims=True)
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

model = build_crnn_model(len(vocab))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=ctc_loss)

train_images, train_labels = preprocess_data(
    train_image_paths, train_labels, img_height, img_width, vocab
)
val_images, val_labels = preprocess_data(
    val_image_paths, val_labels, img_height, img_width, vocab
)

history = model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=20,
    batch_size=32,
    callbacks=[
        # tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True)
    ]
)

# train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
# val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=ctc_loss)

# def preprocess_test_images(image_paths, img_height, img_width):
#     images = []
#     for path in image_paths:
#         try:
#             img = tf.io.read_file(path)
#             img = tf.image.decode_png(img, channels=1)
#             img = tf.image.resize(img, [img_height, img_width])
#             img = tf.image.convert_image_dtype(img, tf.float32) / 255.0
#             images.append(img)
#         except Exception as e:
#             print(f"Error memproses gambar di path {path}: {e}")
    
#     return tf.convert_to_tensor(images)

# def decode_predictions(predictions, vocab):
#     num_to_char = StringLookup(vocabulary=list(vocab), mask_token=None, invert=True)
#     decoded_texts = []
#     for pred in predictions:
#         pred_decoded = tf.argmax(pred, axis=-1)
#         text = num_to_char(pred_decoded).numpy()
#         decoded_texts.append("".join([char.decode('utf-8') for char in text if char]))
#     return decoded_texts

# test_image_paths = ["data_for_train_model/text_0.png", "data_for_train_model/text_1.png", "data_for_train_model/text_2.png"]
# test_labels = ["Energi 120 kkal", "Lemak total 2.50 g", "Lemak jenuh 2 g"]

# test_images = preprocess_test_images(test_image_paths, img_height, img_width)

# predictions = model.predict(test_images)

# decoded_predictions = decode_predictions(predictions, vocab)

# for i, (pred, true_label) in enumerate(zip(decoded_predictions, test_labels)):
#     print(f"Gambar {i+1}:")
#     print(f"  Prediksi: {pred}")
#     print(f"  Label Sebenarnya: {true_label}")