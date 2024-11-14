import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping



##############################################################################
# STEP 1. Data Processing
##############################################################################

##### Image sizing
width, height, channel = 192, 192, 3
batch_size_setting = 64

##### Modifications
# Set up data augmentation for training and re-scaling for validation/test
train_datagen = ImageDataGenerator(
    rescale=1.0/255,            # Converting pixel RGB ranges to a scale of 0-1. 
    shear_range=0.15,           # Adding some tilt randomness
    zoom_range=0.15,            # Adding some zoom randomness
    horizontal_flip=True        # Adding horizontal flipping randomly
)

test_datagen = ImageDataGenerator(
    rescale=1.0/255,            # Converting pixel RGB ranges to a scale of 0-1. 
    shear_range=0.15,           # Adding some tilt randomness
    zoom_range=0.15,            # Adding some zoom randomness
    horizontal_flip=True        # Adding horizontal flipping randomly
)


# No need for the modifications to the last dataset
validation_datagen = ImageDataGenerator(rescale=1.0/255)
   



##### Loading images
train_generator = train_datagen.flow_from_directory(
    './Data/train',
    target_size=(width, height),
    batch_size=batch_size_setting,
    class_mode='categorical'  # Use 'categorical' for multi-class classification
)

validation_generator = validation_datagen.flow_from_directory(
    './Data/valid',
    target_size=(width, height),
    batch_size=batch_size_setting,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    './Data/test',
    target_size=(width, height),
    batch_size=batch_size_setting,
    class_mode='categorical',
    shuffle=False             # Better for debugging and repeatability
)

##############################################################################
# Step 2: Neural Network Architecture Design 
##############################################################################

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channel)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout layer with 50% rate
model.add(layers.Dense(3, activation='softmax'))  # Softmax for probabilities


# debug
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),  # Matches one-hot labels
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(train_generator, epochs=50,
                    validation_data=validation_generator,
                    callbacks=[early_stopping])


##### Graphing
# Plot accuracy 
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()

# Is this Loss?
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, max(max(history.history['loss']), max(history.history['val_loss']))])
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
