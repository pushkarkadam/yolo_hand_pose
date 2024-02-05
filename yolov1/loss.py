import torch
import torch.nn as nn
from .utils import *


class YoloLoss(nn.Module):
    """YOLO loss"""
    def __init__(self, S=7, B=2, C=2, K=21, batch_size=1):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.K = K
        self.batch_size=batch_size
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        self.lamda_keypoint = 0.5
        
    def forward(self, predictions, target):
        pred = yolo_head(predictions, num_boxes=self.B, num_landmarks=self.K, num_classes=self.C, grid_size=self.S, batch_size=self.batch_size)
        
        # Extracting Prediction
        pred_boxes_xy = pred['bboxes_xy']
        pred_boxes_wh = pred['bboxes_wh']
        pred_boxes_classes = pred['classes']
        pred_boxes_conf = pred['confidence']
        pred_boxes_landmarks = pred['landmarks']
        
        # Extracting Ground Truth Target
        target_box_xy = target['box_xy']
        target_box_wh = target['box_wh']
        target_box_classes = target['classes_gt']
        target_box_confidence = target['confidence_gt']
        target_box_landmarks = target['landmarks_gt']
        
        target_box_corners = yolo_boxes_to_corners(target_box_xy, target_box_wh)
        
        boxes_iou = []
        
        for pred_box_xy, pred_box_wh in zip(pred_boxes_xy, pred_boxes_wh):
            pred_box_corners = yolo_boxes_to_corners(pred_box_xy, pred_box_wh)
            
            box_iou = iou(pred_box_corners, target_box_corners)
            
            boxes_iou.append(box_iou)
            
        boxes_iou = [b.unsqueeze(0) for b in boxes_iou]
    
        boxes_iou = torch.cat(boxes_iou, dim=0)

        iou_maxes, best_box_index = torch.max(boxes_iou, dim=0)
        
        # Indicator function that is used to test if the object exists
        exists_box = target['confidence_gt']
        
        # Squeezing the box dimensions to convert from (1, 2, S, S) to (2, S, S)
        pred_boxes_xy = [box.squeeze(0) for box in pred_boxes_xy]
        pred_boxes_wh = [box.squeeze(0) for box in pred_boxes_wh]
        pred_boxes_lmk = [box.squeeze(0) for box in pred_boxes_landmarks]
        
        # Finding the box responsible for prediction
        predictor_xy = predictor_box(pred_boxes_xy, best_box_index.squeeze(0))
        predictor_wh = predictor_box(pred_boxes_wh, best_box_index.squeeze(0))
        predictor_conf = predictor_box(pred_boxes_conf, best_box_index.squeeze(0))
        
        # Implement predictor_box function for keypoints
        predictor_lmk = predictor_box(pred_boxes_lmk, best_box_index.squeeze(0))

        # Coordinate losses
        # xy loss
        predictor_xy = exists_box * predictor_xy
        xy_loss = self.lambda_coord * self.mse(torch.flatten(predictor_xy), torch.flatten(target_box_xy))

        # wh loss
        predictor_wh = exists_box * predictor_wh
        wh_loss = self.lambda_coord * self.mse(torch.flatten(torch.abs(predictor_wh)**(1/2)), torch.flatten(torch.abs(target_box_wh)**(1/2)))

        # conf object loss
        predictor_conf = exists_box * predictor_conf
        obj_loss = self.mse(torch.flatten(predictor_conf), torch.flatten(target_box_confidence))

        # confidence not object loss
        no_exists_box = (1 - exists_box)
        no_predictor_conf = no_exists_box * predictor_conf
        noobj_loss = self.lambda_noobj * self.mse(torch.flatten(no_predictor_conf), torch.flatten(target_box_confidence)) 

        # class loss
        pred_boxes_classes = exists_box * pred_boxes_classes
        class_loss = self.mse(torch.flatten(pred_boxes_classes), torch.flatten(target_box_classes))

        # Keypoint loss
        # Reshaping from (1,2*K,S,S) to (2*K, S, S)
        target_lmk = target_box_landmarks.squeeze()

        # Reshaping to broadcast multiplication across the channels of landmark tensor
        exists_lmk_box = exists_box.reshape(self.batch_size, self.S, self.S, 1)

        # Broadcasting multiplication
        pred_lmk = exists_lmk_box * predictor_lmk.transpose(dim0=1, dim1=-1)

        # Reshaping the tensor
        pred_lmk = pred_lmk.reshape(self.batch_size, self.K*2, self.S, self.S)
        
        lmk_xy_target = relative_cartesian_tensor(target_lmk)
        lmk_xy_pred = relative_cartesian_tensor(pred_lmk)
        
        landmark_loss = self.mse(torch.flatten(lmk_xy_target), torch.flatten(lmk_xy_pred))

        # Computing total loss
        loss = xy_loss + wh_loss + obj_loss + noobj_loss + class_loss + landmark_loss
        
        return loss