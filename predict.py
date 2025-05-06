import cv2
import keras
import random
import numpy as np
from os import listdir, path

# --- Settings ---
MODEL_PATH = 'persian_digit_model.keras'
CAPTCHA_DIR = 'test/'
IMG_WIDTH_MODEL = 30
IMG_HEIGHT_MODEL = 40
NUM_DIGITS_EXPECTED = 5
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"

# --- Load Model ---
print(f"Loading model from {MODEL_PATH}...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Load and Preprocess Captcha Image ---

captcha_files = [f for f in listdir(CAPTCHA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not captcha_files:
    print(f"No valid images found in '{CAPTCHA_DIR}' directory.")
    exit()

selected_file = random.choice(captcha_files)
CAPTCHA_IMAGE_PATH = path.join(CAPTCHA_DIR, selected_file)
print(f"Selected test image: {CAPTCHA_IMAGE_PATH}")

# Read image with OpenCV (for better processing)
image_bgr = cv2.imread(CAPTCHA_IMAGE_PATH)
if image_bgr is None:
    print(f"Error: Unable to read image from path '{CAPTCHA_IMAGE_PATH}'")
    exit()

image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Adaptive thresholding
thresh = cv2.adaptiveThreshold(
    image_gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,
    30)  # Parameters may need adjustment

# --- Find Contours for Digit Detection ---
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Number of contours found: {len(contours)}")

digit_bounding_boxes = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    min_h, max_h = 15, IMG_HEIGHT_MODEL + 20
    min_w, max_w = 5, IMG_WIDTH_MODEL + 15
    if min_h < h < max_h and min_w < w < max_w:
        digit_bounding_boxes.append((x, y, w, h))

print(f"Number of valid contours (likely digits): {len(digit_bounding_boxes)}")

# --- Sort contours by horizontal position (left to right) ---
digit_bounding_boxes.sort(key=lambda item: item[0])

if len(digit_bounding_boxes) < NUM_DIGITS_EXPECTED:
    print(f"Warning: Number of digits found ({len(digit_bounding_boxes)}) is less than expected ({NUM_DIGITS_EXPECTED})")
    print("May need to adjust threshold or contour filter parameters.")
elif len(digit_bounding_boxes) > NUM_DIGITS_EXPECTED:
    print(f"Warning: Number of digits found ({len(digit_bounding_boxes)}) is more than expected ({NUM_DIGITS_EXPECTED})")
    print("  Attempting to select top 5 contours (based on position)...")
    # Simple solution: take first 5 after sorting
    digit_bounding_boxes = digit_bounding_boxes[:NUM_DIGITS_EXPECTED]

# --- Predict each digit ---
predicted_text = ""
output_image = image_bgr.copy()

for i, (x, y, w, h) in enumerate(digit_bounding_boxes):
    # Extract digit image from thresholded image
    # Add some padding for safety
    pad = 5
    digit_roi = thresh[max(0, y-pad):min(thresh.shape[0], y+h+pad),
                       max(0, x-pad):min(thresh.shape[1], x+w+pad)]

    if digit_roi.size == 0:
        continue

    # Resize to model input dimensions
    resized_digit = cv2.resize(digit_roi, (IMG_WIDTH_MODEL, IMG_HEIGHT_MODEL))
    # Prepare for model input (normalize, add batch and channel dimensions)
    processed_digit = resized_digit.astype('float32') / 255.0
    processed_digit = np.expand_dims(processed_digit, axis=-1)
    processed_digit = np.expand_dims(processed_digit, axis=0)

    # Make prediction
    prediction = model.predict(processed_digit)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_digit_char = PERSIAN_DIGITS[predicted_class_index]

    predicted_text += predicted_digit_char
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

print(f"Image: {CAPTCHA_IMAGE_PATH}")
print(f"Predicted Text: {predicted_text} \n")

# Display image with detection boxes
cv2.imshow("Captcha Prediction Result", output_image)
print("\nClose the result window to exit the program...")
cv2.waitKey(0)
cv2.destroyAllWindows()
