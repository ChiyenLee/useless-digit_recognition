import numpy as np
import cv2
import math
import pickle
import gpiozero
import time

def main():

    # define out put pins to the FPGA
    digit_pin_0 = gpiozero.OutputDevice(4)
    digit_pin_1 = gpiozero.OutputDevice(17)
    digit_pin_2 = gpiozero.OutputDevice(27)
    digit_pin_3 = gpiozero.OutputDevice(22)
    load = gpiozero.OutputDevice(26)

    # Define camera button
    camera_button = gpiozero.InputDevice(25) # at pin 18

    # load the trained svm model 
    print('loading model')
    f = open('trained_svm_model_big', 'rb')
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
        # clear cmaera buffer
        for i in range(4):
            camera.grab()
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
            if (w > 10 and w < 320) and (h > 100 and h < 300):
                detected_digits.append(c)

        if detected_digits == []:
            print("Nothing detected... Try again")

        else:
            load.off()

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

            result_binary = format(result[0], '04b')
        
            load.on()
            digit_pin_0.on()  if result_binary[3] == '1' else digit_pin_0.off()
            digit_pin_1.on()  if result_binary[2] == '1' else digit_pin_1.off()
            digit_pin_2.on()  if result_binary[1] == '1' else digit_pin_2.off()
            digit_pin_3.on()  if result_binary[0] == '1' else digit_pin_3.off()



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
