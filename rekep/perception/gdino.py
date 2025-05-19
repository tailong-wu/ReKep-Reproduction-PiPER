from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task

import os
import numpy as np
import cv2
import supervision as sv
from rekep.perception.rle_util import rle_to_array

API_TOKEN = os.getenv("DDS_CLOUDAPI_TEST_TOKEN")

class GroundingDINO:
    def __init__(self):
        config = Config(API_TOKEN)
        self.client = Client(config)
    
    def get_dinox(self, image_path):
        image_url = self.client.upload_file(image_path)
        task = V2Task(api_path="/v2/task/dinox/detection", 
        api_body={
        "model": "DINO-X-1.0",
        "image": image_url,
        "prompt": {
            "type":"universal",
        },
        "targets": ["bbox", "mask"], 
        "bbox_threshold": 0.25,
        "iou_threshold": 0.8
        })
        self.client.run_task(task)
        predictions = task.result["objects"]
        return predictions
    
    def visualize_bbox_and_mask(self, predictions, img_path, output_dir):
        # decode the prediction results
        classes = [pred["category"] for pred in predictions]
        classes = list(set(classes))
        class_name_to_id = {name: id for id, name in enumerate(classes)}
        class_id_to_name = {id: name for name, id in class_name_to_id.items()}

        boxes = []
        masks = []
        confidences = []
        class_names = []
        class_ids = []

        for idx, obj in enumerate(predictions):
            boxes.append(obj["bbox"])
            masks.append(rle_to_array(obj["mask"]["counts"], obj["mask"]["size"][0] * obj["mask"]["size"][1]).reshape(obj["mask"]["size"]))
            confidences.append(obj["score"])
            cls_name = obj["category"].lower().strip()
            class_names.append(cls_name)
            class_ids.append(class_name_to_id[cls_name])

        boxes = np.array(boxes)
        masks = np.array(masks)
        class_ids = np.array(class_ids)
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]

        img = cv2.imread(img_path)
        detections = sv.Detections(
            xyxy = boxes,
            mask = masks.astype(bool),
            class_id = class_ids,
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(os.path.join(output_dir, "dinox_bbox.jpg"), annotated_frame)


        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(output_dir, "dinox_mask.jpg"), annotated_frame)
        
        print(f"Annotated image {img_path} has already been saved to {output_dir}")
        print(f"\033[92mDebug: # Boxes: {len(boxes)}\033[0m")
        return boxes, masks

    