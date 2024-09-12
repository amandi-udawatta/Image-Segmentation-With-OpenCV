import cv2
import numpy as np

# Load image
image = cv2.imread('dog.jpg')

# Display original image
cv2.imshow('Original', image)
cv2.waitKey(0)

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display grayscale image
cv2.imshow('Grayscale', gray_image)
cv2.waitKey(0)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Display blurred image
cv2.imshow('Blurred', blurred_image)
cv2.waitKey(0)

# Apply threshold to segment the animal (on the grayscale image, not edges)
_, thresholded = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)

# Display thresholded image
cv2.imshow('Thresholded Image', thresholded)
cv2.waitKey(0)

# Find contours on the thresholded image
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

# Display image with contours
cv2.imshow('Image with Contours', image_with_contours)
cv2.waitKey(0)

# Step 7: Create a mask for the animal and remove the background
mask = np.zeros(image.shape[:2], dtype="uint8")  # Create a black mask of the same size as the image
cv2.drawContours(mask, contours, -1, 255, -1)  # Fill the contours on the mask with white

# Apply the mask to the original image
segmented_animal = cv2.bitwise_and(image, image, mask=mask)

# Display the segmented animal
cv2.imshow('Segmented Animal', segmented_animal)
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
