import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import sys
import copy
from collections import Counter
import yaml


def show_landmarks(image, 
                   bounding_box, 
                   landmarks, 
                   image_class,
                   font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_thickness=1,
                   box_color=(255, 0, 0),
                   box_thickness=2,
                   font_scale=0.2,
                   font_color=(0, 0, 0),
                   label_font_color=(255, 255, 255),
                   label_font_scale=0.5,
                   label_font_thickness=2):
    """Shows landmarks on the image.
    
    Parameters
    ----------
    image: numpy.ndarray
        A numpy image.
    bounding_box: list
        A list of bounding box coordinates.
    landmarks: list
        A list of landmark coordinates.
    """
    
    frame = copy.deepcopy(image)

    if image_class == 0:
        image_label = 'right_hand'
        box_color = (255, 0, 0)
    else:
        image_label = 'left_hand'
        box_color = (255, 192, 203)
    
    EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
    
    H, W, _ = frame.shape
    x,y,w,h = np.int32(np.array(bounding_box) * np.array([W, H, W, H]))
    
    landmarks = landmarks * H
    
    xmin = x - w/2
    ymin = y - h/2
    xmax = x + w/2
    ymax = y + h/2

    # Landmarks
    uv = [(np.int32(i[0]), np.int32(i[1])) for i in landmarks]
    
    for e in EDGES:
        frame = cv2.line(frame, uv[e[0]], uv[e[1]], (255, 255, 255), 2)
    
    for n, landmark in enumerate(uv):
        frame = cv2.circle(frame, landmark, 2, (255, 0, 0), -1)
        frame = cv2.putText(frame, 
                            text=str(n), 
                            org=landmark, 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.2,
                            color=(0, 0, 0),
                            thickness=1
                           )
        
    # Bounding box
    start_point = (np.int32(xmin), np.int32(ymin))
    end_point = (np.int32(xmax), np.int32(ymax))
    frame = cv2.rectangle(frame, start_point, end_point, box_color, 2)
    
    # Bounding box center
    frame = cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
    
    text = str(f"{image_label}")
    text_size, _ = cv2.getTextSize(text, font, label_font_scale, font_thickness)
    text_w, text_h = text_size
    text_end_point = (start_point[0] + text_w, start_point[1] + text_h)
    frame = cv2.rectangle(frame, start_point, text_end_point , box_color, -1)
    frame = cv2.putText(frame,
                        text=text,
                        org=(start_point[0], start_point[1]+int(text_h)),
                        fontFace=font,
                        fontScale=label_font_scale,
                        color=label_font_color,
                        thickness=label_font_thickness
                                   )
        
    return frame

def render_sample(labels, data_path, file_type, image_dir, seed=0, sample_size=4, sample_image_number=9):
    """Generates rendered images with bounding box.
    
    Parameters
    ----------
    labels: pandas.DataFrame
        A pandas datafile with annotation labels.
    data_path: str
        The path where the images are stored.
    file_type: str
        The file type example: ``training`` or ``evaluation``.
    seed: int, default ``0``
        The seed for random generator.
    sample_size: int, default ``4``
        The number of sample images to generate

    Returns
    -------
    rendered_images: dict
        A dict that maps the sample number to rendered image.
    names: dict
        A dict that maps the sample number to image name.

    """
    np.random.seed(seed)
    
    samples = []
    
    rendered_images = dict()
    names = dict()
    
    for s in range(sample_size):
        rendered_images[s] = []
        names[s] = []
    
        idx = list(np.random.randint(0,labels.shape[0], size=sample_image_number))

        for i in idx:
            image_name = labels.iloc[i, 0]
            names[s].append(image_name)

            image_class = labels.iloc[i, 1]
            bounding_box = labels.iloc[i, 2:6]
            bounding_box = np.asarray(bounding_box, dtype=float)
            landmarks = labels.iloc[i,6:]
            landmarks = np.asarray(landmarks, dtype=float).reshape(-1,2)

            image = cv2.imread(os.path.join(data_path, file_type, image_dir, image_name))

            H, W, _ = image.shape

            frame = show_landmarks(image, bounding_box, landmarks, image_class)

            rendered_images[s].append(frame)
    
    return rendered_images, names

def plot_rendered_grid(rendered_images, names, dir_name='check', path='.', save_images=False):
    """Creates a directory with the grid plots.
    
    Parameters
    ----------
    rendered_images: dict
        A dictionary that maps the sample size number to rendered image.
    names: dict
        A dictionary that maps the sample size number to image name.
    dir_name: str, default ``'check'``
        The name of the directory where the images will be saved.
    path: str, default ``'.'``
        The path where the directory for image will be created.
    save_images: bool, default ``False``
        Saves the images in the directory ``dir_name``.

    """
    dir_path = os.path.join(path, dir_name)
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"New directory {dir_name} created at {path}")
    
    for s in list(rendered_images.keys()):
        r_images = rendered_images[s]
        r_names = names[s]
        fig = plt.figure(figsize=(10, 10))
        for i in range(len(r_images)):
            ax = fig.add_subplot(3,3,i+1)
            plt.imshow(r_images[i])
            plt.title(r_names[i])
            plt.axis('off')
                
        if save_images:
            file_name = f"{dir_name}_{s}.jpg"
            file_path = os.path.join(dir_path, file_name)
            fig.savefig(file_path)

def conv_array_dim(input_size, kernel_size, stride=1, padding=0):
    """Returns the dimension of the array after convolution operation.
    
    Parameters
    ----------
    input_size: int
        The size of the input array.
    kernel_size: int
        The size of the kernel.
    stride: int, default ``1``
        The stride over the convolution operation.
    padding: int, default ``0``
        The padding added to the array.

    Returns
    -------
    int
        The dimension fo the output array.

    """
    return int(np.floor((input_size - kernel_size + 2 * padding)/stride) + 1)

def maxpool_dim(input_size, kernel_size, stride=1, padding=0):
    """Returns the dimension of the output array after maxpooling function.
    
    Parameters
    ----------
    input_size: int
        The size of the input array.
    kernel_size: int
        The size of the kernel.
    stride: int, default ``1``
        The stride over the convolution operation.
    padding: int, default ``0``
        The padding added to the array.

    Returns
    -------
    int
        The dimension fo the output array.
    
    """

    return int(np.floor(input_size/kernel_size))

def load_architecture(model_path):
    """Loads model architecture from the yaml file.
    
    Parameters
    ----------
    model_path: str
        The path where the model architecture is stored in a YAML file.
        Example: ``'../models/LeNet.yaml'``
        The YAML file must have ``architecture`` as a key.
        Example:
        ```
        architecture:
            [
                #[from, number, module, args: [in_channel, out_channel, kernel_size, stride, padding]
                [-1, 1, Conv, [3, 6, 5, 1, 0]],
                [-1, 1, MaxPool, [2, 2, 0]], # [kernel_size, stride, padding]
                [-1, 1, Conv, [6, 16, 3, 1, 0]],
                [-1, 1, MaxPool, [2, 2, 0]],
                [-1, 1, Fc, [120], ReLU], # [from, number, module, units, activation]
                [-1, 1, Fc, [84], ReLU],
                [-1, 1, Fc, [2], None]
            ]
        ```

    """
    # Loading YAML file
    with open(model_file, 'r') as f:
        model = yaml.safe_load(f)

    return model['architecture']


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cpu",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])