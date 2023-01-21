import cv2
import sys
import os
import numpy as np
import urllib
import matplotlib.pyplot as plt
import tensorflow as tf

# def camera_test():
#     s = 0
#     if len(sys.argv) > 1:
#         s = sys.argv[1]

#     source = cv2.VideoCapture(s)

#     win_name = 'Camera Preview'
#     cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

#     while cv2.waitKey(1) != 27:  # Escape
#         has_frame, frame = source.read()
#         if not has_frame:
#             break
#         cv2.imshow(win_name, frame)

#     source.release()
#     cv2.destroyWindow(win_name)

# camera_test()

interpreter = tf.lite.Interpreter(model_path="D:\OneDrive - Nanyang Technological University\Year 2 Sem 2\SC2079 MDP\Tensor_Testing\Tensorflow Testing\model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['bbpooh', 'bbtig', 'bbroo', 'lola']

cap = cv2.VideoCapture(0)
ok, frame_image = cap.read()
original_image_height, original_image_width, _ = frame_image.shape

while True:
    ok, frame_image = cap.read()
    if not ok:
        break
    # help
    # cropped_image = frame_image[250:430, 830:1490]
    # cv2.imshow("testwindow", frame_image)
    resize_img = cv2.resize(frame_image, (180, 180))
    reshape_image = resize_img.reshape(180, 180, 3)
    image_np_expanded = np.expand_dims(reshape_image, axis=0)
    image_np_expanded = image_np_expanded.astype('float32')  # float32

    cv2.namedWindow('detect_result', cv2.WINDOW_NORMAL)
    cv2.imshow("detect_result", frame_image)

    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break
    elif key == 32:
        cv2.waitKey(0)
        continue

cap.release()
cv2.destroyAllWindows()