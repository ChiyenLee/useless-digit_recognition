# process_digit.py
# John Lee 2018.11.20
# Code that runs on the Raspberry PI
# that waits for a button click before 
# classifying an image to a digit from 0 - 10

import numpy as np
import cv2
import math
import pickle
import gpiozero
import time

def main():

    # Define camera button
    camera_button = gpiozero.InputDevice(18) # at pin 18

    # load the trained svm model 
    print('loading model')
    f = open('trained_svm_model_small', 'rb')
    classifier = pickle.load(f)
        
    # Initiate camera
    camera = cv2.VideoCapture(0)
    counter = 0
    while True:
        print('Please press the button: ')
        while camera_button.value == False:
            if camera_button.value:
                print('Captured! Processing ')
                break

        return_value, image = camera.read()

        # First blur the image for easier processing
        blurred = cv2.GaussianBlur(image, (15,15),0)

        # convert it to gray scale
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        retval, thres1 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        # find contour to extract the number
        im2, contours, hierarchy = cv2.findContours(thres1, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        detected_digits = []

        # if the contour is around 180 wide and 240 tall. it must 
        # be a digit. NEEDS calibration before start
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if (w > 10 and w < 320) and (h > 180 and h < 280):
                detected_digits.append(c)

        if detected_digits == []:
            print("Nothing detected... Try again")

        else:
            (x,y,w,h) = cv2.boundingRect(detected_digits[0])
            digit = thres1[y:y+h, x:x+w]
            # use interarea to shrink without losing pixel value
            if digit.shape[0] > digit.shape[1]:
                # check which dimesion is the dominant one and shrink it to 20
                resize_digit = image_resize(digit, height=20, inter = cv2.INTER_AREA)
            else:
                resize_digit = image_resize(digit, width=20, inter=cv2.INTER_AREA)

            # now fill in the gap so that the final dimension is 28 x 28
            cols = resize_digit.shape[1]
            rows = resize_digit.shape[0]
            colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
            rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
            padded = np.lib.pad(resize_digit,(rowsPadding,colsPadding), 'constant',constant_values=(255, 255))
            
            # Normalize the Pixel value such that it's between -1 to 1
            flatten = (255 - padded.flatten())/255 * 2 - 1
            result = classifier.predict(flatten.reshape(1,-1))
            print("The number you entered is:  ", result[0])
            cv2.imwrite( "digit_" + str(counter) + ".jpg", digit);

            counter = counter + 1

        time.sleep(1)


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    '''Resize the image so it maintains the correct ratio'''
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

if __name__ == "__main__":
    main()