import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Loading model
model = tf.keras.models.load_model('IcarusIII_RevengeOfTheIDE.keras')

# Class names (I'm assuming it has to be the same as the folder names )
classNames = ['crack', 'missing-head', 'paint-off']

# Setting up image processing
def preprocess_image(image_path, target_size=(256, 256)):   # 256 because that's how I trained the model. Don't feel like
                                                            # playing around with the dimensions again at this junction
    img = load_img(image_path, target_size=target_size)
    imageArray = img_to_array(img)
    imageArray = imageArray / 255.0
    # Add a batch dimension (for prediction)
    imageArray = np.expand_dims(imageArray, axis=0)
    return imageArray

# Loading the test images
test_images = {
    "test_crack.jpg": "./Data/test/crack/test_crack.jpg",
    "test_missinghead.jpg": "./Data/test/missing-head/test_missinghead.jpg",
    "test_paintoff.jpg": "./Data/test/paint-off/test_paintoff.jpg"
}

# Now we get to see if my model is crap or not 
for name, path in test_images.items():
    imageArray = preprocess_image(path)
    predictions = model.predict(imageArray)
    predictedClass = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Print results
    print(f"Image: {name}")
    print(f"Predicted Class: {classNames[predictedClass]}")
    print(f"Confidence: {confidence:.2f}")
    print("-" * 30)


##### Creating the annotated images (As a function, so I can call it for each image and don't have to rewrite it 3 times)
def overlay_predictions(image_path, model, class_names, true_label, output_path):
    imageArray = preprocess_image(image_path)

    # Getting predictions from the model
    predictions = model.predict(imageArray)[0]  # Extract the prediction array from batch
    predicted_index = np.argmax(predictions)  # Get the index of the class with the highest confidence
    predicted_label = class_names[predicted_index]  # Predicted class name
    
    # Matches the prediction %s to their names
    predictions_dict = {class_names[i]: predictions[i] * 100 for i in range(len(class_names))}
    
    # Loading the image and resizing back to original so the text isn't wonky
    img = cv2.imread(image_path)
    img = cv2.resize(img, (500, 500))  # Resize for consistent text overlay
    
    # Adding text overlay (I totally tried a bunch of different fonts to find the worst one)
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    font_scale = 1.0
    thickness = 2
    y_offset = 40
    
    # Adding labels: true, then predicted, then confidences
    cv2.putText(img, "Model used: Icarus III", (10, y_offset), font, font_scale, (0, 0, 0), thickness)
    cv2.putText(img, f"True Label: {true_label}", (10, y_offset + 40), font, font_scale, (0, 0, 0), thickness)
    cv2.putText(img, f"Predicted Label: {predicted_label}", (10, y_offset + 40*2), font, font_scale, (0, 0, 0), thickness)
    for i, (label, confidence) in enumerate(predictions_dict.items()):
        cv2.putText(img, f"{label}: {confidence:.1f}%", (10, 380 + i * 40), font, font_scale, (0, 128, 0), thickness)  # Green means good, right? We'll go with green...
    
    # Saving the output image
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Don't ask me what this is, but without it the pictures save blank.
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.show()


# Let it be known that on this day in 2024 I discovered python requires forward slashes to designate file directories.
# My dissappointment is immeasurable and my day is ruined.


### 1st image: test crack
image_path = "./Data/test/crack/test_crack.jpg" 
true_label = "crack"
output_path = "IcarusIII_test-crack-result.png"

overlay_predictions(
    image_path=image_path,
    model=model,
    class_names=classNames,
    true_label=true_label,
    output_path=output_path
)

### 2nd image: missing head
image_path = "./Data/test/missing-head/test_missinghead.jpg"
true_label = "missing head"
output_path = "IcarusIII_test-headoff-result.png"

overlay_predictions(
    image_path=image_path,
    model=model,
    class_names=classNames,
    true_label=true_label,
    output_path=output_path
)

### 3rd image: missing head
image_path = "./Data/test/paint-off/test_paintoff.jpg"
true_label = "paint off"  
output_path = "IcarusIII_test-paintchip-result.png"

overlay_predictions(
    image_path=image_path,
    model=model,
    class_names=classNames,
    true_label=true_label,
    output_path=output_path
)

