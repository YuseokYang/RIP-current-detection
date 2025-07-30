import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from torchvision import transforms
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

class RipCurrentDetector:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.ToTensor()

    def _load_model(self, checkpoint_path):
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def predict(self, image_bytes: bytes, conf_threshold=0.1, iou_threshold=0.3):
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)[0]

        boxes = outputs["boxes"]
        scores = outputs["scores"]

        keep = nms(boxes, scores, iou_threshold)
        boxes = boxes[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()

        # confidence 필터링
        results = []
        for box, score in zip(boxes, scores):
            if score >= conf_threshold:
                x1, y1, x2, y2 = map(int, box)
                results.append({
                    "box": [[x1, y1], [x2, y2]],
                    "score": round(float(score), 3)
                })
        return results
