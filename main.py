from nomeroff_net import pipeline
from nomeroff_net.tools import unzip
import time

number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading",
                                              image_loader="opencv")

while True:
    file_path = input("Enter file path: ")
    start_time = time.time()
    (images, images_bboxs,
     images_points, images_zones, region_ids,
     region_names, count_lines,
     confidences, texts) = unzip(number_plate_detection_and_reading([file_path]))

    print(texts)
    print("region_ids = ", region_ids)
    print("region_names = ", region_names)
    print("count_lines = ", count_lines)
    print("time_elapsed = ", time.time() - start_time)
