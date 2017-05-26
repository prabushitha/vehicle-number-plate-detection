import cv2
# Importing the Opencv Library
import numpy as np
# Importing NumPy,which is the fundamental package for scientific computing with Python



def extract_number_plate(img):

    r = 400.0 / img.shape[1]
    dim = (400, int(img.shape[0] * r))
    
    # perform the actual resizing of the image and show it
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # Display image

    # RGB to Gray scale conversion
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    noise_removal = cv2.bilateralFilter(img_gray,9,75,75)

    # Histogram equalisation for better results
    equal_histogram = cv2.equalizeHist(noise_removal)
    
    # Morphological opening with a rectangular structure element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=15)
    #cv2.imwrite('outputs/10_morph_image.png', morph_image)

    # Image subtraction(Subtracting the Morphed image from the histogram equalised Image)
    sub_morp_image = cv2.subtract(equal_histogram,morph_image)
    #cv2.imwrite('outputs/11_sub_morp_image.png', sub_morp_image)

    # Thresholding the image
    ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)
    #cv2.imwrite('outputs/12_thresh_image.png', thresh_image)
    # Applying Canny Edge detection
    canny_image = cv2.Canny(thresh_image,250,255)

    canny_image = cv2.convertScaleAbs(canny_image)
    #cv2.imwrite('outputs/13_canny_image.png', canny_image)


    # dilation to strengthen the edges
    kernel = np.ones((3,3), np.uint8)
    # Creating the kernel for dilation
    dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
    #cv2.imwrite('outputs/14_dilated_image.png', dilated_image)

    # Finding Contours in the image based on edges
    new,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    # Sort the contours based on area ,so that the number plate will be in top 10 contours
    screenCnt = None
    # loop over our contours
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximating with 6% error
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:  # Select the contour with 4 corners
            screenCnt = approx
            break
    final = cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

    # Masking the part other than the number plate
    mask = np.zeros(img_gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)
    
    return new_image
