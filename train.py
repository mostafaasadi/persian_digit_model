from tensorflow import keras
import matplotlib.pyplot as plt

# --- Settings ---
DATASET_DIR = 'captcha_dataset'
MODEL_SAVE_PATH = 'persian_digit_model.keras'
IMG_WIDTH = 30
IMG_HEIGHT = 40
BATCH_SIZE = 128
EPOCHS = 2000  # Early stopping will likely stop before this
VALIDATION_SPLIT = 0.3  # Percentage of data for validation
NUM_CLASSES = 10   # Number of classes (digits 0-9)
SEED = 42
keras.mixed_precision.set_global_policy('mixed_float16')

# --- Loading Data ---
print("Loading data...")

datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,             # Normalize pixel values to [0, 1]
    rotation_range=5,           # Small random rotation
    width_shift_range=0.1,      # Small horizontal shift
    height_shift_range=0.1,     # Small vertical shift
    zoom_range=0.1,            # Small zoom in/out
    validation_split=VALIDATION_SPLIT  # Split validation data
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode='sparse',  # Labels will be integers (0-9)
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=True,
    seed=SEED,
    subset='validation'  # Specify this portion as validation data
)

print(f"Number of training images: {train_generator.samples}")
print(f"Number of validation images: {validation_generator.samples}")
print(f"Classes (labels): {train_generator.class_indices}")

if train_generator.num_classes != NUM_CLASSES:
    print(f"Error: Number of found classes ({train_generator.num_classes}) doesn't match NUM_CLASSES ({NUM_CLASSES})")
    print("Make sure dataset folder contains 10 subfolders named 0 through 9")
    exit()

# --- Define CNN Model ---
print("Building CNN model...")
input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)  # 1 for grayscale

inputs = keras.layers.Input(shape=input_shape)

x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D((2, 2))(x)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)

outputs = keras.layers.Dense(
    NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = keras.models.Model(inputs=inputs, outputs=outputs)

model.summary()

# --- Compile Model ---
print("Compiling model...")
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',  # Because labels are integers
    metrics=['accuracy'])

# --- Define Callbacks ---
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True)

# --- Train Model ---
print("Starting model training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping, model_checkpoint]
)
model.save(MODEL_SAVE_PATH)
print(f"Training completed. Best model saved to '{MODEL_SAVE_PATH}'")

# --- Plot Accuracy and Loss Graphs ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.suptitle('Model Training History')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('training_history.png', dpi=300)
plt.show()
