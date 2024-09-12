import cv2
import numpy as np

# Load the image
image = cv2.imread('monkey.jpeg')

# Resize the image for better processing speed (keeping aspect ratio)
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        # Calculate the ratio of the height and construct the new dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # Calculate the ratio of the width and construct the new dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Resize the image
    return cv2.resize(image, dim, interpolation=inter)

# Use the resize function with aspect ratio preserved (resize based on width or height)
resized_image = resize_with_aspect_ratio(image, width=500)  # Resize based on width while keeping aspect ratio

cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)

# Create an initial mask for GrabCut
mask = np.zeros(resized_image.shape[:2], np.uint8)

# Define the background and foreground models (used internally by GrabCut)
bg_model = np.zeros((1, 65), np.float64)
fg_model = np.zeros((1, 65), np.float64)

# Define a rectangle around the object (manually adjust to the size of the object)
rect = (50, 50, resized_image.shape[1] - 100, resized_image.shape[0] - 100)  # x, y, width, height

# Apply GrabCut algorithm
cv2.grabCut(resized_image, mask, rect, bg_model, fg_model, 10, cv2.GC_INIT_WITH_RECT)

# Modify the mask: mark sure foreground and probable foreground as 1, others as 0
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Segment the object using the modified mask
segmented = resized_image * mask2[:, :, np.newaxis]


# Display the results
cv2.imshow("Segmented Animal ", segmented)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
