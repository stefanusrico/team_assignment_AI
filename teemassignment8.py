import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Resizing, RandomFlip, RandomRotation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt

# 1. PERSIAPAN DATASET
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()

x_train = tf.keras.applications.mobilenet_v2.preprocess_input(x_train.astype('float32'))
x_val = tf.keras.applications.mobilenet_v2.preprocess_input(x_val.astype('float32'))

# 2. MEMBANGUN ARSITEKTUR & FITUR OPTIMASI
def build_model(hp):
    model = Sequential()

    # --- PERBAIKAN 1: Resize & Data Augmentation ---
    # MobileNetV2 butuh minimal 96x96 piksel agar fiturnya terekstrak dengan baik
    model.add(Resizing(96, 96, input_shape=(32, 32, 3))) 
    model.add(RandomFlip("horizontal"))
    model.add(RandomRotation(0.1))
    
    # Memuat base model (sekarang tanpa mendefinisikan input_shape di sini karena sudah dihandle Resizing)
    base_model = MobileNetV2(include_top=False, weights='imagenet')
    base_model.trainable = False
    
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(BatchNormalization())

    # Tuning 1: Dense Layer
    hp_units = hp.Int('dense_units', min_value=64, max_value=256, step=64)
    model.add(Dense(units=hp_units, activation='relu'))

    # Tuning 2: Dropout Rate
    hp_dropout = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    model.add(Dropout(rate=hp_dropout))

    # Output Layer
    model.add(Dense(10, activation='softmax'))

    # Tuning 3: Learning Rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# 3. PROSES HYPERPARAMETER TUNING
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='tuning_dir_baru', # Ganti nama dir agar tidak bentrok dengan run sebelumnya
                     project_name='optimasi_cnn_cifar')

# --- PERBAIKAN 2: Early Stopping ---
# Berhenti otomatis jika val_loss tidak membaik selama 3 epoch, lalu kembalikan bobot terbaik
stop_early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("Starting process Hyperparameter Tuning...")
tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val), callbacks=[stop_early])

# Mengambil kombinasi parameter terbaik
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
Search finished! Best parameter found:
- Dense Units: {best_hps.get('dense_units')}
- Dropout Rate: {best_hps.get('dropout_rate')}
- Learning Rate: {best_hps.get('learning_rate')}
""")

# 4. TRAINING MODEL MATANG
print("Training final model using best hyperparameter...")
model_final = tuner.hypermodel.build(best_hps)

# Epoch bisa kita naikkan menjadi 20 karena kita menggunakan Early Stopping yang akan mencegah overfitting
history = model_final.fit(
    x_train, y_train, 
    epochs=20, 
    validation_data=(x_val, y_val),
    callbacks=[stop_early]
)

# 5. MENYIMPAN & EVALUASI MODEL
file_name = 'cnn_model_optimized_extra.keras'
model_final.save(file_name)
print(f"Succeed! Model saved with name '{file_name}'")

loss, accuracy = model_final.evaluate(x_val, y_val)
print(f"Validation Accuracy Akhir: {accuracy:.4f}")