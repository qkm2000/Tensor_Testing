from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np


# def center_crop(img, dim):
#     """Returns center cropped image
#     Args:
#     img: image to be center cropped
#     dim: dimensions (width, height) to be cropped
#     """
#     width, height = img.shape[1], img.shape[0]

#     # process crop width and height for max available dimension
#     crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
#     crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0] 
#     mid_x, mid_y = int(width/2), int(height/2)
#     cw2, ch2 = int(crop_width/2), int(crop_height/2)
#     crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
#     return crop_img


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()
    
    # image = image[0:720, 280:1000]
    # image = center_crop(image, [500, 500])

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
