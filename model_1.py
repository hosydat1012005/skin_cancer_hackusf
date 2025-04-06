#Import necessary libraries
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Paths to retrieve datasets
base_dir = "data/images"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Parameters to process and train the images
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15  

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255, #Normalize pixels to [0,1]
    rotation_range=20, #Rotate image randomly
    zoom_range=0.15, #Zoom in/out image randomly
    horizontal_flip=True, #Horizontal flip
    fill_mode='nearest' #Fill blanks pixels with nearest ones
)

#Simple normalizing pixels for testing dataset
test_datagen = ImageDataGenerator(rescale=1./255)

# Data loaders
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Build the pretrained model for transfer learning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#Not allowing pretrained weights to be adjusted, except for weights in the last 10 layers
base_model.trainable = False
for layer in base_model.layers[:-10]:
    layer.trainable = False

#Build the model for fine tuning
model = Sequential([
    base_model,
    GlobalAveragePooling2D(), #Average pooling for feature maps
    Dropout(0.3), #Set dropout rate to avoid overfitting
    Dense(64, activation='relu'), #64-neuron layer with ReLU activation function
    Dense(1, activation='sigmoid') #Output layer with sigmoid function
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Test Accuracy')
plt.savefig('accuracy_plot.png')
plt.show()

# Save model
model.save('skin_cancer_model.h5')
loss, accuracy = model.evaluate(test_data)
print(f'Test accuracy: {accuracy: .2f}')
