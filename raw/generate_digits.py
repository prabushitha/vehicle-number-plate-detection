import os
import cv2
import numpy as np

# ============================================================================    

def extract_chars(img):
    bw_image = cv2.bitwise_not(img)
    contours = cv2.findContours(bw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    char_mask = np.zeros_like(img)
    bounding_boxes = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        x,y,w,h = x-2, y-2, w+4, h+4
        bounding_boxes.append((x,y,w,h))


    characters = []
    for bbox in bounding_boxes:
        x,y,w,h = bbox
        char_image = img[y:y+h,x:x+w]
        characters.append(char_image)

    return characters

# ============================================================================    

def output_chars(chars, labels):
    for i, char in enumerate(chars):
        filename = "chars/%s.png" % labels[i]
        char = cv2.resize(char
            , None
            , fx=3
            , fy=3
            , interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(filename, char)

# ============================================================================    

if(not os.path.exists("chars")):
    os.makedirs("chars")

img_digits = cv2.imread("train_digits.png", 0)
img_letters = cv2.imread("train_letters.png", 0)

digits = extract_chars(img_digits)
letters = extract_chars(img_letters)

DIGITS = [0, 9, 8 ,7, 6, 5, 4, 3, 2, 1]
LETTERS = [chr(ord('A') + i) for i in range(25,-1,-1)]

output_chars(digits, DIGITS)
output_chars(letters, LETTERS)
