import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


##############################################################################
# STEP 1. Data Processing
##############################################################################

# Define the image sizing
width, height, channel = 500, 500, 3
batch_size = 32