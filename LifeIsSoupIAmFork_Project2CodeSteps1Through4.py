import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LeakyReLU

# So originally I wrote this code, and then when the model worked I saved the model 
# and kept revising the code to create the Icarus models. When I was writing the report,
# I realized it might be prudent to have this version available as well, so I resurrected
# it from GitHub revision history. TA, if you're reading this, go look at the IcarusIII
# code; it turned out better than this one anyway.

##############################################################################
# STEP 1. Data Processing
##############################################################################

##### Image sizing
width, height, channel = 256, 256, 3
batch_size_setting = 64

##### Modifications
# Set up data augmentation for training and re-scaling for validation/test. 
# I picked 0.18 for each of these because it seems reasonable that aircraft mechanics
# be trained to take consistent photos. 
train_datagen = ImageDataGenerator(
    rescale=1.0/255,            # Converting pixel RGB ranges to a scale of 0-1. 
    shear_range=0.18,           # Adding some tilt randomness
    zoom_range=0.18,            # Adding some zoom randomness
    horizontal_flip=True        # Adding horizontal flipping randomly
)

test_datagen = ImageDataGenerator(
    rescale=1.0/255,            # Converting pixel RGB ranges to a scale of 0-1. 
    shear_range=0.18,           # Adding some tilt randomness
    zoom_range=0.18,            # Adding some zoom randomness
    horizontal_flip=True        # Adding horizontal flipping randomly
)


# No need for the modifications to the testing datasets
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

##############################################################################
# Step 2/3: Neural Network Architecture and Hyperparameters
##############################################################################

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channel)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu')) # I Tried leaky, doesn't work that well in this case.

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dropout(0.2))  # Dropout layer with 20% rate
model.add(layers.Dense(3, activation='softmax'))  


# debug
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

#early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(train_generator, epochs=12,
                    validation_data=validation_generator)
                    #callbacks=[early_stopping])

model.save('LifeIsSoupIAmFork.keras')

##############################################################################
# Step 4: Model Evaluation
##############################################################################

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

# Final Metrics
# print(f"Test Loss: {test_loss:.4f}")
# print(f"Test Accuracy: {test_accuracy:.4f}")