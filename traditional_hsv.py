import cv2
import numpy as np
from sklearn.cluster import KMeans

# Function to find multiple dominant colors using KMeans clustering
def get_dominant_colors(image, k=2):
    # Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))
    
    # Apply KMeans to find k dominant colors
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Get the k dominant colors
    dominant_colors = kmeans.cluster_centers_.astype(int)
    
    return dominant_colors

# Load image
image = cv2.imread('animal.jpg')

# Convert the image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv image', hsv_image)
cv2.waitKey(0)

# Get the top k dominant colors
dominant_colors_bgr = get_dominant_colors(image, k=2)  # Get top 2 dominant colors
print('Dominant Colors (BGR):', dominant_colors_bgr)

# Initialize an empty mask for combining multiple color ranges
final_mask = np.zeros(image.shape[:2], dtype="uint8")

# Loop over each dominant color and create a mask for each
for dominant_color_bgr in dominant_colors_bgr:
    dominant_color_hsv = cv2.cvtColor(np.uint8([[dominant_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    
    # Define color range for this dominant color
    lower_color = np.array([dominant_color_hsv[0] - 10, 50, 50])
    upper_color = np.array([dominant_color_hsv[0] + 10, 255, 255])
    
    # Create a mask for this color range
    color_mask = cv2.inRange(hsv_image, lower_color, upper_color)
    cv2.imshow('mask', color_mask)
    cv2.waitKey(0)
    
    # Combine this mask with the final mask
    final_mask = cv2.bitwise_or(final_mask, color_mask)

# Invert the final mask to segment the animal
final_mask = cv2.bitwise_not(final_mask)
cv2.imshow('final_mask', final_mask)
cv2.waitKey(0)

# Apply Gaussian blur to smooth the mask and reduce noise
blurred_mask = cv2.GaussianBlur(final_mask, (5,5), 0)

# Display the blurred mask
cv2.imshow('Blurred Mask', blurred_mask)
cv2.waitKey(0)

#not used because it selects a wrong contour
# edges = cv2.Canny(blurred_mask, 50, 150)

# # Display the edges
# cv2.imshow('Edges', edges)
# cv2.waitKey(0)

# Find contours based on the blurred mask
contours, _ = cv2.findContours(blurred_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area and take the largest one (assuming it's the animal)
# largest one is at [-1]
cnt = sorted(contours, key=cv2.contourArea)[-1]
print('Contours:', cnt)

# Draw the largest contour on the image
image_with_contour = image.copy()
cv2.drawContours(image_with_contour, [cnt], -1, (0, 255, 0), 3)  # Green contour, thickness=3
cv2.imshow('Largest Contour', image_with_contour)
cv2.waitKey(0)

# Create a mask for the animal and remove the background
mask = np.zeros(image.shape[:2], dtype="uint8")
masked_image = cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
cv2.imshow("masked_image", masked_image)
cv2.waitKey(0)

# Apply the mask to the original image to segment the animal
segmented_animal = cv2.bitwise_and(image, image, mask=masked_image)
cv2.imshow('Segmented Animal', segmented_animal)
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()