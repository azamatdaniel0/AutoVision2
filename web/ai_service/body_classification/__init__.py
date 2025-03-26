import time
import torch
from init_models import body_classification_model as model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
            confidence_score = round(confidence_score, 4)
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