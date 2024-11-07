import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator




##############################################################################
# STEP 1. Data Processing
##############################################################################

##### Image sizing
width, height, channel = 500, 500, 3
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
