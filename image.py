import cv2 as cv
import numpy as np

def colour_mask(image_path):
    img = cv.imread(image_path)
    img = resize(img)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # define range of black color in HSV
    black_lower_val = np.array([0,0,0])
    black_upper_val = np.array([180,255,30])

    # define range for silver color in HSV
    silver_lower_val = np.array([0,0,168])
    silver_upper_val = np.array([180,25,255])

    # Threshold the HSV image to get only black colors
    black_mask = cv.inRange(hsv, black_lower_val, black_upper_val)

    # Threshold the HSV image to get only silver colors
    silver_mask = cv.inRange(hsv, silver_lower_val, silver_upper_val)

    # Combine the two masks
    combined_mask = cv.bitwise_or(black_mask, silver_mask)

    res = cv.bitwise_and(img,img, mask= combined_mask)
    cv.imshow('Original',img)
    cv.imshow('Mask',combined_mask)
    cv.imshow('Result',res)
    cv.waitKey(0)
    cv.destroyAllWindows()

def edge_detection(image_path):
    img = cv.imread(image_path)
    img = resize(img)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshold1, threshold2 = 100, 180
    edges = cv.Canny(gray_image,threshold1,threshold2)
    cv.imshow('Original',img)
    cv.imshow('Edges',edges)
    cv.waitKey(0)
    cv.destroyAllWindows()


def resize(image):
    img = image
    # Example: Resize to 30% of the original size to zoom out
    scale_percent = 30  # Percentage of the original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize image
    resized_image = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return resized_image

    

if __name__ == "__main__":
    image_path = 'images/top1.jpg'
    #colour_mask(image_path)
    edge_detection(image_path)
