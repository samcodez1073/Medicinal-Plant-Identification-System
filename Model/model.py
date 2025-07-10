import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
# Set dataset directory (update this path as needed)
data_dir = r'C:\Users\AHMAD ALI\Desktop\medicinal-plant-identifier\data\Medicinal plant dataset'
# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest')
# Training generator
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    shuffle=True)
# Validation generator
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=True)
# Number of classes
num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")
# Load MobileNetV2 base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False  # Freeze initially
# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)
# Build model
model = Model(inputs=base_model.input, outputs=predictions)
# Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# Initial training (frozen base)
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, 
    restore_best_weights=True)])
# Unfreeze some top layers of base model for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False
# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
# Fine-tune
fine_tune_epochs = 5
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=fine_tune_epochs,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, 
    restore_best_weights=True)])
# Save model
output_path = os.path.join("models", "medicinal_plant_model.h5")
os.makedirs("models", exist_ok=True)
model.save(output_path)
print(f" ✅Model saved to '{output_path}'")