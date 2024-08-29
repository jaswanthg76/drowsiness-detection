import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your dataset
root_dir = 'C://Users/jaswa/OneDrive/Desktop/major project/mrleyedataset/'

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values
    validation_split=0.2,    # Reserve 20% of the data for validation
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    root_dir,
    target_size=(24, 24),
    batch_size=32,
    class_mode='binary'
)

validation_generator = datagen.flow_from_directory(
    root_dir,
    target_size=(24, 24),
    batch_size=32,
    class_mode='binary'
)

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Save the model
model.save('eye_state_detector.h5')
