import sys
import cv2
import numpy as np
import number_plate as nplate
# ============================================================================

def reduce_colors(img, n):
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = n
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2 

# ============================================================================

def clean_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resized_img = cv2.resize(gray_img
        , None
        , fx=5.0
        , fy=5.0
        , interpolation=cv2.INTER_CUBIC)

    resized_img = cv2.GaussianBlur(resized_img,(5,5),0)
    cv2.imwrite('outputs/1_licence_plate_large.png', resized_img)

    equalized_img = cv2.equalizeHist(resized_img)
    cv2.imwrite('outputs/2_licence_plate_equ.png', equalized_img)


    reduced = cv2.cvtColor(reduce_colors(cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR), 8), cv2.COLOR_BGR2GRAY)
    cv2.imwrite('outputs/3_licence_plate_red.png', reduced)


    ret, mask = cv2.threshold(reduced, 64, 255, cv2.THRESH_BINARY)
    cv2.imwrite('outputs/4_licence_plate_mask.png', mask) 

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel, iterations = 1)
    cv2.imwrite('outputs/5_licence_plate_mask2.png', mask)

    return mask

# ============================================================================

def extract_characters(img):
    bw_image = cv2.bitwise_not(img)
    contours = cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

    char_mask = np.zeros_like(img)
    bounding_boxes = []
    
    small_count = 0
    medium_count = 0
    large_count = 0
    areas = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = w * h
        areas.append(area)

        if (area > 1000) and (area < 100000):
            small_count = small_count+1
            
        if (area > 2500) and (area < 100000):
            medium_count = medium_count+1
            
        if (area > 5000) and (area < 100000):
            large_count = large_count+1
    min_ar = 1000
    if large_count > 5:
        min_ar = 5000
    elif medium_count > 5:
        min_ar = 2500
    elif small_count > 5:
        min_ar = 1000
    else:
        #no number plate found
        return -1, -1
    #print(areas)
        
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = w * h
        center = (x + w/2, y + h/2)
        if (area > min_ar) and (area < 100000):
            x,y,w,h = x-4, y-4, w+8, h+8
            bounding_boxes.append((center, (x,y,w,h)))
            cv2.rectangle(char_mask,(x,y),(x+w,y+h),255,-1)

    cv2.imwrite('outputs/licence_plate_mask3.png', char_mask)

    clean = cv2.bitwise_not(cv2.bitwise_and(char_mask, char_mask, mask = bw_image))

    bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])  

    characters = []
    for center, bbox in bounding_boxes:
        x,y,w,h = bbox
        char_image = clean[y:y+h,x:x+w]
        characters.append((bbox, char_image))

    return clean, characters


def highlight_characters(img, chars):
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for bbox, char_img in chars:
        x,y,w,h = bbox
        cv2.rectangle(output_img,(x,y),(x+w,y+h),255,1)

    return output_img

# ============================================================================    



def getPlateNumber(img):
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    r = 300.0 / img.shape[1]
    dim = (300, int(img.shape[0] * r))
    
    # perform the actual resizing of the image and show it
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    #end resize

    img = clean_image(img)
    clean_img, chars = extract_characters(img)
    
    #if not match found
    if type(clean_img) is int and clean_img==-1:
        return -1

    output_img = highlight_characters(clean_img, chars)
    cv2.imwrite('outputs/licence_plate_out.png', output_img)


    samples = np.loadtxt('data_generation/char_samples.data',np.float32)
    responses = np.loadtxt('data_generation/char_responses.data',np.float32)
    responses = responses.reshape((responses.size,1))


    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    plate_chars = ""
    for bbox, char_img in chars:
        small_img = cv2.resize(char_img,(10,10))
        small_img = small_img.reshape((1,100))
        small_img = np.float32(small_img)
        retval, results, neigh_resp, dists = model.findNearest(small_img, k = 1)
        plate_chars += str(chr((results[0][0])))

    return plate_chars

#img = cv2.imread("test_img1.png")
if len(sys.argv)==2:
    path = sys.argv[1]
else:
    path = "inputs/test_1.jpg"
try:
    img = cv2.imread(path)
    cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image",img)

    final = nplate.extract_number_plate(img)

    cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
    cv2.imshow("Final_image",final)

    numberplate = getPlateNumber(final)
    letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    #print(letters)
    if not(type(numberplate) is int) and not(numberplate==-1):
        final_plate = ""
        
        for x in range(len(numberplate)):
            #print(x)
            #print(letters.index(str(numberplate[x])))
            if x==3 and (numberplate[x] in letters):
                pass
            elif x==2 and (numberplate[x] in letters) and not(numberplate[0] in letters[0:3]):
                pass
            else:
                final_plate = final_plate+numberplate[x]
            
        print("Licence plate: %s\n" % final_plate)
    else:
        print("Number Plate Characters couldn't identified properly")

    cv2.waitKey() # Wait for a keystroke from the user
except:
    print("Image is not valid")
