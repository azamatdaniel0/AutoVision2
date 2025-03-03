import time
from ultralytics import YOLO
import torch
from decouple import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = config("MODEL_PATH")
model = YOLO(MODEL_PATH)

def predict(image_path):
    try:
        results = model.predict(image_path)
        start_time = time.time()

        if results[0].probs is not None and len(results[0].probs) > 0:
            probs = results[0].probs.data.tolist()
            class_id = probs.index(max(probs))
            class_name = model.names[class_id]
            confidence = max(probs)
            predicted_label = class_name
            confidence_score = confidence
            print("time: ", time.time() - start_time)
        else:
            predicted_label = None
            confidence_score = None
    except Exception as ex:
        print(ex)
        predicted_label = None
        confidence_score = None
    return {
        "car_type_body": predicted_label,
        "car_type_body_score": confidence_score
    }