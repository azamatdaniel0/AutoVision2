import torch
from init_models import country_model, upsampler
import cv2
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.color import rgb2gray
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip
import time


start_time = time.time()
number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading",
                                              image_loader="opencv")
print(f"Pipeline initialization took {time.time() - start_time:.2f} seconds")


# Define the time measurement wrapper
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start_time:.2f} seconds")
        return result
    return wrapper

# PaddleOCR takes cv2 image
@measure_time
def paddle(img, lang):
    recognition = PaddleOCR(use_angle_cls=True, lang=lang, det=False)
    result = recognition.ocr(img)
    results_array = []
    confidences_array = []
    for idx in range(len(result[0])):
        res = result[0][idx][1][0]
        confidence = result[0][idx][1][1]
        results_array.append(res)
        confidences_array.append(confidence)
    combined_element = "".join(results_array)
    final = combined_element.replace(" ", "")

    # returns text and confidential probability
    return final, confidences_array

# Define a function for preprocessing an image
@measure_time
def preprocess(img):
    # Detect rotation angle (Your original rotation logic)
    rot_angle = 0
    grayscale = rgb2gray(img)
    edges = canny(grayscale, sigma=3.0)
    out, angles, distances = hough_line(edges)
    _, angles_peaks, _ = hough_line_peaks(out, angles, distances, num_peaks=20)
    angle = np.mean(np.rad2deg(angles_peaks))

    # Adjust rotation angle (Your original rotation logic)
    if 0 <= angle <= 90:
        rot_angle = angle - 90
    elif -45 <= angle < 0:
        rot_angle = angle - 90
    elif -90 <= angle < -45:
        rot_angle = 90 + angle
    if abs(rot_angle) > 20:
        rot_angle = 0

    # Rotate and prepare for cropping
    rotated = rotate(img, rot_angle, resize=True) * 255
    rotated = rotated.astype(np.uint8)

    H, W = rotated.shape[:2]  # Get height and width

    # Dynamic Cropping Logic
    crop_margin = 0  # Initialize (no cropping by default)
    if W / H > 1.5:  # Adjust 1.5 value if needed for rectangle definition
        crop_margin = np.abs(int(np.sin(np.radians(rot_angle)) * H))
    elif W / H < 0.8:  # Adjust this for your idea of 'square'
        crop_margin = np.abs(int(np.sin(np.radians(rot_angle)) * W))

    # Apply the Crop
    if crop_margin > 6:
        rotated = rotated[crop_margin:-crop_margin, crop_margin:-crop_margin]

    return rotated


# Define symbols as a constant outside the function
SYMBOLS = {
    "-", "#", "_", "+", "=", "!", "@", "$", "%", "*", "&", "(", ")", "^", "/", "|", ";", ":", ".", ",", "·", "<", ">", "[", "]"
}

@measure_time
def del_symbols(input_string):
    return ''.join(char for char in input_string if char not in SYMBOLS)



@measure_time
def calculate_mean(numbers):
    if len(numbers) == 0:
        return None

    total = sum(numbers)
    mean = total / len(numbers)
    return mean


@measure_time  # This will time the entire function
def lp_det_reco(img_path):
    try:
        result = number_plate_detection_and_reading(
            [img_path]
        )
        (
            images, images_bboxs,
            images_points, images_zones, region_ids,
            region_names, count_lines,
            confidences, texts
        ) = unzip(result)

    except Exception as e:
        print(e)

    try:
        # lp lines
        print(images_bboxs[0][0])
        x_min, y_min, x_max, y_max, _, _, _ = images_bboxs[0][0]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        print(x_min, y_min, x_max, y_max)
        pro = preprocess(images[0][y_min:y_max, x_min:x_max])
        try:
            if count_lines[0][0] == 1:
                pro = images_zones[0][0].astype("uint8")
        except:
            pass
        pro_resized = cv2.resize(pro, (224, 224))

        # country
        pred_country, pred_idx, probs_country = country_model.predict(pro_resized)
        probs_country = f"{probs_country[pred_idx]:.4f}"
        country = (pred_country, probs_country)

        # enchance img or not
        H, W, _ = pro.shape
        if H >= 100 and H <= 300 and W >= 100 and W <= 300:
            img_enh, _ = upsampler.enhance(pro, outscale=1)
        else:
            img_enh = pro
        img_final = Image.fromarray(img_enh)
        H, W, _ = img_enh.shape
        # OCR
        match country[0]:
            # OCR if this is Chinese license plate
            case "CN":
                combined_element_without_spaces, conf = paddle(img_enh, "ch")
                combined_element_without_spaces = del_symbols(
                    combined_element_without_spaces
                )
                if (
                    combined_element_without_spaces[0] == "0"
                    or combined_element_without_spaces[1] == "0"
                ):
                    combined_element_without_spaces = (
                        combined_element_without_spaces.replace("0", "Q", 1)
                    )
                combined_element_without_spaces = texts[0][0]
            case "KG":
                number_text = texts[0][0]
                number_text = list(number_text)
                number_text.insert(2, "KG")
                conf = confidences[0][0]
                combined_element_without_spaces = "".join(number_text)

            # OCR if this is other country license plates
            case _:
                conf = confidences[0][0]
                combined_element_without_spaces = texts[0][0]

        combined_element_without_spaces = del_symbols(combined_element_without_spaces)
        conf = calculate_mean(conf)
    except Exception as error:
        print('error = ', error)
        combined_element_without_spaces = None
        conf = None
        country = [None, None]
    return {
        "license_plate_number": combined_element_without_spaces,
        "license_plate_number_score": conf,
        "license_plate_country": country[0],
        "license_plate_country_score": country[1],
        "car_brand": "BMW_crop",
        "car_brand_score": 0.9997,
        "car_color": "grey",
        "car_color_score": 0.9911,
        "car_type_body": "SUV",
        "car_type_body_score": 1
    }