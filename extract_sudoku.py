######################################################################
# This module will read and extract Sudoku image from given picture.
# The extracted image will be saved in image directory
######################################################################

import numpy as np
import cv2


def image_preprocess(img, skip_dilate=False):
    ''' Cnvert the image to gray scale, blur image, apply adaptive threshold to highlight main features of the image '''
        
    # Gaussian blur to image with kernal size of (9, 9)
    img_proc = cv2.GaussianBlur(img, (9, 9), 0)
    
    # Adaptive threshold using 11 nearest neighbour pixels
    img_proc = cv2.adaptiveThreshold(img_proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Invert colours, so gridlines have non-zero pixel values.
    # Necessary to dilate the image, otherwise will look like erosion instead.
    img_proc = cv2.bitwise_not(img_proc, img_proc) 
    
    if not skip_dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
        img_proc = cv2.dilate(img_proc, kernel)
    
    return img_proc


def find_largest_contour_points(img):
    ''' find the largest contour in the image'''
    
    # Find contours in the image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort the contours in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Draw contour in image
    #cv2.drawContours(contours[0], bigContour, -1, (255, 0, 0), 3)
    
    # Find the perimeter of the lagest contour
    perimeter = cv2.arcLength(contours[0], True)
    
    # Find the polygon details from the contour
    get_ploy = cv2.approxPolyDP(contours[0], 0.02 * perimeter, True)
    
    # Reorder Contour points
    points = reorder_points(get_ploy)
    
    return points


def reorder_points(points):
    ''' reorder contour points'''
    
    # reshape contour points array
    points = points.reshape((4,2))
    
    #print(f'Contour points : { points }')
    
    # array to hold re-ordered points
    points_new = np.zeros((4,1,2), np.int32)
    
    # (right, bottom) (left, top)
    add = points.sum(axis=1)
    points_new[0] = points[np.argmin(add)]
    points_new[2] = points[np.argmax(add)]
    
    # (lef, bottom) (right, top)
    diff = np.diff(points, axis = 1)
    points_new[1] = points[np.argmin(diff)]
    points_new[3] = points[np.argmax(diff)]
    
    return points_new


def calculate_distance(pt1, pt2):
    
    # calculate distance between two points
    distance = np.sqrt(((pt1[0][0] - pt2[0][0]) ** 2 ) + ((pt1[0][1] - pt2[0][1]) ** 2))
    #print(f'Distance calculated { distance }')
    
    return distance


def get_warp(image, contour_points):
    ''' function to corp and warp the image'''
    
    # calculate the maximum value of side length
    side = max([calculate_distance(contour_points[0], contour_points[1]),
               calculate_distance(contour_points[1], contour_points[2]),
               calculate_distance(contour_points[2], contour_points[3]),
               calculate_distance(contour_points[3], contour_points[0])])
    
    #print(f'Side Calculated : { side }')
    
    # points source array for perspective transformation
    pts1 = np.float32(contour_points)
    
    # points destination array for perspective transformation
    pts2 = np.float32([[0, 0], [int(side)-1, 0], [int(side)-1, int(side)-1], [0, int(side)-1]])
    
    # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    
    # Performs the transformation on the original image
    image_out = cv2.warpPerspective(image, matrix, (int(side), int(side)))
    
    return image_out


def get_digit_boxes(img):
    ''' function to find the corners of individual sqaures'''
    
    digit_boxes = []
    side_length = img.shape[:1]
    side_length = side_length[0] / 9
    
    # the rectangles are stored in the list reading left-right instead of top-down
    for j in range(9):
        for i in range(9):
            # Top left corner of a bounding box
            pt1 = (i * side_length, j * side_length)
            
            # Bottom right corner of bounding box
            pt2 = ((i +1 ) * side_length, (j + 1) * side_length)
            
            digit_boxes.append((pt1, pt2))
    
    return digit_boxes


def cut_from_rect(image, box):
    return image[int(box[0][1]):int(box[1][1]), int(box[0][0]):int(box[1][0])]


def find_largest_feature(digit, scan_top_left, scan_btm_rght):
    
    height, width = digit.shape[:2]
    
    max_area = 0
    seed_point = (None, None)
    
    if scan_top_left is None:
        scan_top_left = [0, 0]
        
    if scan_btm_rght is None:
        scan_btm_rght = [height, width]
        
    # Loop through the image
    for x in range(scan_top_left[0], scan_btm_rght[0]):
        for y in range(scan_top_left[1], scan_btm_rght[1]):
            if digit.item(y, x) == 255 and x < width and y < height:
                area = cv2.floodFill(digit, None, (x, y), 64)
                if area[0] > max_area:
                    max_area = area[0]
                    seed_point = (x, y)
                    
    # Colour everything grey
    for x in range(width):
        for y in range(height):
            if digit.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(digit, None, (x, y), 64)
    
    # Mask that is 2 pixels bigger than the image
    mask = np.zeros((height + 2, width + 2), np.uint8)
    
    # Highlight the main feature
    if all([p is not None for p in seed_point]):
        cv2.floodFill(digit, mask, seed_point, 255)
        
    top, bottom, left, right = height, 0, width, 0
    
    for x in range(width):
        for y in range(height):
            if digit.item(y, x) == 64:  # Hide anything that isn't the main feature
                cv2.floodFill(digit, mask, (x, y), 0)

            # Find the bounding parameters
            if digit.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right
    
    # bounding box
    bbox = [[left, top], [right, bottom]]
    
    return np.array(bbox, dtype='float32'), seed_point


def scale_and_centre(img, size, margin=0, background=0):
    """Scales and centres an image onto a new background square."""
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))


def extract_digit(image, box, size):
    '''Extracts a digit (if one exists) from a Sudoku square.'''
    
    # Get the digit box from the whole square
    digit = cut_from_rect(image, box)
    
    # use floodfill feature to find the largest feature in the rectange
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
    
    digit = cut_from_rect(digit, bbox)
    
    # Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]
    
    # Ignore any small bounding boxes
    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
        return scale_and_centre(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)


def get_digit_images(image, digit_boxes, size=28):
    
    digit_images = []
    
    image = image_preprocess(image, skip_dilate=True)
    
    for digit_box in digit_boxes:
        digit_images.append(extract_digit(image, digit_box, size))
        
    return digit_images


def show_image(img):
    """Shows an image until any key is pressed"""
    
    #cv2.imshow('image', img)  # Display the image
    cv2.imwrite('images/extract_sudoku.jpg', img)
    #cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    #cv2.destroyAllWindows()  # Close all windows
    return img


def show_digits(digits, colour=255):
    """Shows list of 81 extracted digits in a grid format"""
    rows = []
    with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
    for i in range(9):
        row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        rows.append(row)
    img = show_image(np.concatenate(rows))
    return img


def extract_sudoku(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    imgContour = image.copy()

    # preprocess image
    image_process = image_preprocess(image)

    # find the corner points of largest image
    contour_points = find_largest_contour_points(image_process)

    # crop and warp image
    image_cropped = get_warp(image, contour_points)
    cv2.imwrite('images/Cropped_sudoku.jpg', image_cropped)

    # find the corners of individual digit boxes in puzzle
    digit_boxes = get_digit_boxes(image_cropped)

    # get the individual images from the cropped original and resize for machine learning compatibility
    digit_images = get_digit_images(image_cropped, digit_boxes, size=28)
    
    final_image = show_digits(digit_images)

    return final_image
