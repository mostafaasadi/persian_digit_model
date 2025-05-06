import cv2
import json
import keras
import base64
import numpy as np
from re import sub
from os import getenv
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from requests import post, get


def load_captcha_model(model_path):
    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()


def to_english_digits(s):
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    for i in range(10):
        s = s.replace(persian_digits[i], str(i))
        s = s.replace(arabic_digits[i], str(i))
    return s


def is_valid_card_number(card_number):
    card_number = str(card_number)
    card_number = to_english_digits(card_number)
    cleaned = sub(r"[^\d]", "", card_number)
    is_valid = len(cleaned) == 16
    return is_valid, cleaned


def preprocess_captcha_image(captcha_image_base64):
    try:
        img_data = base64.b64decode(captcha_image_base64)
        pil_img = Image.open(BytesIO(img_data))
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) 

        image_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            image_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            30)

        # Contour detection
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digit_bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            min_h, max_h = 15, IMG_HEIGHT_MODEL + 20
            min_w, max_w = 5, IMG_WIDTH_MODEL + 15
            if min_h < h < max_h and min_w < w < max_w:
                digit_bounding_boxes.append((x, y, w, h))

        # Sort contours by horizontal position (left to right)
        digit_bounding_boxes.sort(key=lambda item: item[0])

        if len(digit_bounding_boxes) < CAPTCHA_LENGTH:
            return False, None
        elif len(digit_bounding_boxes) > CAPTCHA_LENGTH:
            digit_bounding_boxes = digit_bounding_boxes[:CAPTCHA_LENGTH]

        return digit_bounding_boxes, thresh

    except Exception as e:
        print(f"Error in preprocessing CAPTCHA image: {e}")
        return False, None


def solve_captcha(model, digit_bounding_boxes, thresh):
    predicted_captcha = []
    try:
        for i, (x, y, w, h) in enumerate(digit_bounding_boxes):
            pad = 5
            digit_roi = thresh[
                max(0, y-pad):min(thresh.shape[0], y+h+pad),
                max(0, x-pad):min(thresh.shape[1], x+w+pad)
            ]

            if digit_roi.size == 0:
                predicted_captcha.append('X')
                continue

            resized_digit = cv2.resize(
                digit_roi, (IMG_WIDTH_MODEL, IMG_HEIGHT_MODEL))

            processed_digit = resized_digit.astype('float32') / 255.0
            processed_digit = np.expand_dims(processed_digit, axis=-1)
            processed_digit = np.expand_dims(processed_digit, axis=0)

            # predict the digit
            prediction = model.predict(processed_digit)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_captcha.append(int(predicted_class_index))

        predicted_captcha = ''.join(map(str, predicted_captcha))
        return predicted_captcha
    except Exception as e:
        print(e)
        return False


def fetch_captcha():
    url = f"{BASE_ENDPOINT}/wp-admin/admin-ajax.php"
    payload = {
        "action": "ira_iban_captcha"
    }
    response = post(url, data=payload)
    data = json.loads(response.text)
    captcha_image_base64 = data["captcha"].split(",")[1]
    key = data["key"]
    return captcha_image_base64, key


def extract_nonce():
    try:
        response = get(f"{BASE_ENDPOINT}/sheba/")
        response.raise_for_status()
    except Exception:
        return False

    soup = BeautifulSoup(response.content, 'html.parser')
    form = soup.find('form', class_='iban-form')
    if form:
        nonce_value = form.get('data-nonce')
        return nonce_value
    else:
        return None


def submit_form(card_number, key, captcha_solution, nonce):
    url = f"{BASE_ENDPOINT}/wp-admin/admin-ajax.php"
    payload = {
        "action": "ira_iban_action",
        "cardnumber_or_accound": card_number,
        "bank_code": "",
        "key": key,
        "_wpnonce": nonce,
        "captcha": captcha_solution,
    }
    try:
        response = post(url, data=payload)
        return response.json()
    except Exception:
        return False


def print_formatted_result(result):
    if not result["success"]:
        print("Error occurred:")
        print(f"  Message: {result.get('message', 'N/A')}")
        if result["errors"]:
            print(f"  Errors: {result['errors']}")
        return

    print("\n\nRequest Successful!")
    print("-" * 50)  # Separator line

    result_data = result["result"]

    print(f"  Operation Time: {result_data.get('operation_time', 'N/A')}")
    print(f"  Card Number: {result.get('card_number', 'N/A')}")
    print(f"  Reference ID:   {result_data.get('ref_id', 'N/A')}")
    print(f"  IBAN Number:    {result_data.get('iban_number', 'N/A')}")
    print(f"  Deposits:       {result_data.get('deposits', 'N/A')}")
    print(f"  First Name:     {result_data.get('first_name', 'N/A')}")
    print(f"  Last Name:      {result_data.get('last_name', 'N/A')}")
    print(f"  Bank ID:        {result_data.get('Bank-Id', 'N/A')}")

    print("-" * 50)


def main():
    # get the card number from the user
    while True:
        user_input = input("\n\n\tEnter your card number: ")
        is_valid, card_number = is_valid_card_number(user_input)
        if is_valid:
            break
        else:
            print("Invalid card number. Please enter a 16-digit number.")

    # Load the CAPTCHA solving model
    model = load_captcha_model(MODEL_PATH)

    digit_bounding_boxes = None
    predicted_captcha = None
    nonce = None

    for _ in range(3):
        nonce = extract_nonce()
        if not nonce:
            print("Failed to extract nonce. Retrying...")
            continue

        captcha_image_base64, key = fetch_captcha()
        if not captcha_image_base64:
            print("Failed to fetch CAPTCHA. Retrying...")
            continue

        digit_bounding_boxes, thresh = preprocess_captcha_image(captcha_image_base64)
        if not digit_bounding_boxes:
            print("Failed to preprocess CAPTCHA image. Retrying...")
            continue

        predicted_captcha = solve_captcha(model, digit_bounding_boxes, thresh)
        if not predicted_captcha:
            print("Failed to solve CAPTCHA. Retrying...")
            continue

    result = submit_form(card_number, key, predicted_captcha, nonce)
    result['card_number'] = card_number
    result['captcha'] = predicted_captcha
    if result["success"]:
        print_formatted_result(result)
    else:
        print("Request failed, exiting.")
        exit()


if __name__ == "__main__":
    load_dotenv()
    MODEL_PATH = 'persian_digit_model.keras'
    BASE_ENDPOINT = getenv('BASE_ENDPOINT')
    CAPTCHA_LENGTH = 5
    IMG_HEIGHT_MODEL = 40
    IMG_WIDTH_MODEL = 30
    main()
