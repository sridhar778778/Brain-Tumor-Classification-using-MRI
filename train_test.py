import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] Metal GPU.")
    except Exception as e:
        print("[WARN] GPU config failed:", e)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
TEST_DIR = os.path.join(BASE_DIR, "Testing")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model.keras")


IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

# CLASS WEIGHTS 
y_train = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print("[INFO] Computed class weights:", class_weights)

# (VGG16) 
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
for layer in base_model.layers:
    layer.trainable = False


x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(len(train_generator.class_indices), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# COMPILE 
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# CALLBACKS 
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_sparse_categorical_accuracy',
                             save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy',
                              factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_sparse_categorical_accuracy',
                           patience=7, restore_best_weights=True, verbose=1)

# FEATURE EXTRACTION 
print("\n[INFO] Starting initial training (frozen VGG16 layers)...")
history1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# FINE-TUNING
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

print("\n[INFO] Fine-tuning last convolutional block...")
history2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# PLOT TRAINING CURVES
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history1.history['sparse_categorical_accuracy'] + history2.history['sparse_categorical_accuracy'], label='Train Accuracy')
plt.plot(history1.history['val_sparse_categorical_accuracy'] + history2.history['val_sparse_categorical_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history1.history['loss'] + history2.history['loss'], label='Train Loss')
plt.plot(history1.history['val_loss'] + history2.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()

print(f"\n[INFO] Training complete. Best model saved to:\n{MODEL_SAVE_PATH}")