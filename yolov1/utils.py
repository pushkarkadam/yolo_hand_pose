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
                   class_names = ["Right Hand", "Left Hand"],
                   class_probs=None,
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
        A list of landmark coordinates in the format ``[px1, py1, ..., pxn, pyn]``

    """
    # Creating the copy of the image
    frame = copy.deepcopy(image)

    if image_class == 0:
        if class_probs:
            image_label = class_names[0] + ': ' + str(round(class_probs, 2))
        else:
            image_label = class_names[0]
        box_color = (255, 0, 0)
    else:
        if class_probs:
            image_label = class_names[1] + ': ' + str(round(class_probs, 2))
        else:
            image_label = class_names[1]
        box_color = (255, 192, 203)

    if type(landmarks) == list:
        landmarks = np.asarray(landmarks, dtype=float).reshape(-1,2)
    
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
    
    text_start_point = start_point[0], start_point[1] - text_h 
    
    text_end_point = (start_point[0] + text_w, start_point[1])

    frame = cv2.rectangle(frame, text_start_point, text_end_point , box_color, -1)
    frame = cv2.putText(frame,
                        text=text,
                        org=(text_start_point[0], text_end_point[1]),
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

    return int(np.floor((input_size - kernel_size + 2 * padding)/stride) + 1)

def load_architecture(model_file_path):
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
    with open(model_file_path, 'r') as f:
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


# YOLO dev function
def get_relative_landmarks(labels):
    """Gets relative landmarks to the center of bounding box.
    
    Relative landmarks are the polar coordinates of the keypoints w.r.t. bounding box center.
    ``relative_landmarks`` is a list of ``[r1, alpha1, ..., rn, alphan]``.
    
    Parameters
    ----------
    labels: list
        A list of all the annotation of format ``[class, x, y, w, h, px1, px2, ... ,pxn, pyn]``
        
    Returns
    -------
    list
    
    Examples
    --------
    >>> from utils import get_relative_landmarks
    >>> labels = [1, 0.6,0.6,0.4,0.4,0.2,0.2,0.6,0.2,0.8,0.6]
    >>> get_relative_landmarks(labels)
    [0.565685424949238, 0.625, 0.39999999999999997, 0.75, 0.20000000000000007, 0.0]
    
    """
    class_label = labels[0]
    box_dim = labels[1:5]
    landmarks = labels[5:]
    relative_landmarks = []

    x,y,w,h = box_dim

    i = 0
    while i < int(len(landmarks)):
        px, py = landmarks[i:i+2]

        # Relative from bounding box center
        px_r = px - x
        py_r = py - y

        # polar
        r = np.sqrt(px_r**2 + py_r**2)
        theta = np.arctan2(py_r, px_r)

        if theta < 0:
            alpha = theta + 2*np.pi
        else:
            alpha = theta

        # normalising
        alpha = alpha / (2 * np.pi)

        relative_landmarks.append(r)
        relative_landmarks.append(alpha)

        # incrementing
        i += 2
    
    return relative_landmarks

def relative_to_cartesian(labels, landmarks):
    """Converts the the cartesian landmarks.
    
    Function to check if the conversion is correct.
    
    Parameters
    ----------
    labels: list
        A list of labels.
    landmarks: list
        A list in the polar coordinates w.r.t. the center of bounding box.
        
    Returns
    -------
    list:
        A list of converted coordinates.
    
    Examples
    --------
    >>> from utils import relative_to_cartesian
    >>> labels = [1, 0.6,0.6,0.4,0.4,0.2,0.2,0.6,0.2]  # [class, x, y, w, h, px1, py1, px2, py2]
    >>> results = [0.5657, 0.625, 0.4, 0.75] # [r1, alpha1, r2, alpha2]
    >>> relative_to_cartesian(labels, results)
    [0.19998969388277, 0.1999896938827701, 0.5999999999999999, 0.19999999999999996]
    
    """
    
    class_label = labels[0]
    box_dim = labels[1:5]
    cartesian_landmarks = []

    x,y,w,h = box_dim

    i = 0
    while i < int(len(landmarks)):
        r, alpha = landmarks[i:i+2]
        
        # Re-normalizing
        alpha = alpha * (2 * np.pi)

        # Relative from bounding box center
        px = r * np.cos(alpha)
        py = r * np.sin(alpha)
        
        px_c = px + x
        py_c = py + y
        
        cartesian_landmarks.append(px_c)
        cartesian_landmarks.append(py_c)

        # incrementing
        i += 2
    
    return cartesian_landmarks

def break_boxes(boxes):
    """Breaks the bounding box and landmarks.
    
    Parameters
    ----------
    boxes: list
        A list of torch.tensor.
        
    Returns
    -------
    bboxes_xy: list
        A list of bounding box centers ``torch.tensor``
    bboxes_wh: list
        A list of bounding box dimension ``torch.tensor``
    landmarks: list
        A list of landmarks ``torch.tensor``
        
    """
    bboxes_xy = []
    bboxes_wh = []
    landmarks = []
    
    for box in boxes:
        # box centers
        temp_box_xy = box[:,:2,...]
        bboxes_xy.append(temp_box_xy)
        
        # width and height
        temp_box_wh = box[:,2:4,...]
        bboxes_wh.append(temp_box_wh)
        
        temp_landmarks = box[:,4:,...]
        landmarks.append(temp_landmarks)
        
    return bboxes_xy, bboxes_wh, landmarks

def yolo_head(predictions, num_boxes, num_landmarks, num_classes, grid_size, batch_size=1, image_name=None):
    """Returns the tensors.
    
    Parameters
    ----------
    predictions: torch.tensor
        The prediction encodings from YOLO network output.
    num_boxes: int
        Number of boxes
    num_landmarks: int
        Number of landmarks
    grid_size: int
        Size of the grid
    batch_size: int
        The batch size of the data
    
    Returns
    -------
    confidence: torch.tensor
        A tensor of confidence score
    boxes: list
        A list of tensors for bounding box and keypoints
    classes: torch.tensor
        A tensor of classes
    
    """
    S = grid_size
    
    channels = num_boxes * (5 + 2 * num_landmarks) + num_classes

    output = predictions.reshape(batch_size,channels, S, S)
    
    # storing confidence score
    confidence = []
    
    # Storing boxes
    boxes = []
    
    # Assigning start and end for tensor slicing
    start = 0
    end = 0
    
    # Iterating over the boxes
    for box in range(num_boxes):
        temp_confidence = output[:,start,...]
        confidence.append(temp_confidence.unsqueeze(1))
        start += 1
        # 4 (x,y,w,h) + 2 * 21 (px, py) or (r, alpha)
        end = start + 4 + 2 * num_landmarks 
        temp_box = output[:,start:end, ...]
        boxes.append(temp_box)
        start = end
        
    # Classes tensor    
    classes = output[:,-num_classes:,...]
    
    bboxes_xy, bboxes_wh, landmarks = break_boxes(boxes)
    
    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=1)
    
    confidence = [sigmoid(conf) for conf in confidence]
    bboxes_xy = [sigmoid(bbox_xy) for bbox_xy in bboxes_xy]
    bboxes_wh = [sigmoid(bbox_wh) for bbox_wh in bboxes_wh]
    landmarks = [sigmoid(landmark) for landmark in landmarks]
    classes = softmax(classes)
    
    data = {"confidence": confidence,
            "bboxes_xy": bboxes_xy,
            "bboxes_wh": bboxes_wh,
            "landmarks": landmarks,
            "classes": classes,
            "image_name": image_name
           }
    
    return data

def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners.
    
    This is useful for converting the box coordinates in (x,y,w,h) format
    to (x_min, y_min, x_max, y_max) format 
    
    Parameters
    ----------
    box_xy: torch.tensor
        The xy coordinates of the bounding box.
    box_wh: torch.tensor
        The width and height value tensors of bounding box.
    
    Returns
    -------
    list
    
    """
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    box_mins[box_mins < 0.0] = 0
    box_mins[box_mins > 1.0] = 1
    
    box_maxes[box_maxes > 1.0] = 1
    box_maxes[box_maxes < 0.0] = 0
    
    x_min = box_mins[:,0,...]
    y_min = box_mins[:,1,...]
    x_max = box_maxes[:,0,...]
    y_max = box_maxes[:,1,...]
    
    boxes = [x_min, y_min, x_max, y_max]
    return boxes

def iou(box1, box2):
    """Returns IOU
    
    Parameters
    ----------
    box1: list
        A list of torch.tensor ``[x1_min, y1_min, x1_max, y1_max]``
    box2: list
        A list of torch.tensor ``[x2_min, y2_min, x2_max, y2_max]``    
    
    Returns
    -------
    torch.tensor
    
    Examples
    --------
    >>> x1 = [torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([1]), torch.Tensor([1])]
    >>> x2 = [torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([1]), torch.Tensor([1])]
    >>> iou(x1, x2)
    tensor([1.])

    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    xi_min = torch.max(x1_min, x2_min)
    yi_min = torch.max(y1_min, y2_min)
    
    xi_max = torch.min(x1_max, x2_max)
    yi_max = torch.min(y1_max, y2_max)
    
    inter_width = xi_max - xi_min
    inter_height = yi_max - yi_min
    
    inter_area = torch.max(inter_width, torch.tensor(0)) * torch.max(inter_height, torch.tensor(0))
    
    # Box area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    mask = (union_area != 0)

    iou = torch.full_like(inter_area, fill_value=float(0))
    
    iou[mask] = inter_area[mask] / union_area[mask]
    
    return iou

def cell_to_image(box_xy):
    """Converts x and y tensor from cell to image relative.
    
    Parameters
    ----------
    box_xy: torch.tensor
        The ``box_xy`` tensor which is relative to the cell.
        
    """
    grid_size = box_xy.shape[-1]
    
    x = box_xy[:,0,...]
    y = box_xy[:,1,...]
    
    x_mask = (x != 0)
    y_mask = (y != 0)
    
    x = (x + torch.tensor(range(grid_size))) / grid_size
    x = x * x_mask
    
    y = (y + torch.tensor(range(grid_size))) / grid_size
    y = y * y_mask
    
    return x, y

def predictor_box(boxes, index):
    """A list of boxes.
    
    Parameters
    ----------
    boxes: list
        A list of ``torch.Tensor``
    index: torch.Tensor
        A tensor of index that indicates which tensor has max iou score.
        
    Returns
    -------
    torch.Tensor
    
    """
    # Getting shape of the index
    m, row, col = index.shape
    
    # Creating a torch tensor of zeros
    best_box = torch.zeros(boxes[0].shape)
    
    for i in range(m):
        for r in range(row):
            for c in range(col):
                box = boxes[index[i,...].tolist()[r][c]]
                best_box[...,r,c] = box[...,r,c]
        
    return best_box

def relative_cartesian_tensor(landmarks):
    """Returns the cartesian coordinates of the polar tensor.
    
    This is useful during computing the loss function.
    The input tensor is organised as ``(2*K, S, S)`` dimensional tensor.
    The distance ``r`` and angle ``alpha`` for each landmark are organised
    sequentially as channels of the tensors. 
    So, ``[r_1, alpha_1, ... , r_K, alpha_K]`` are the matrix of ``(S, S)`` dimension
    along with the ``2*K`` landmakrs as the channels.
    
    The output of this function is a tensor of size ``(2*K, S, S)`` with the channels
    ``[px_1, py_1, ..., px_K, py_K]`` with each of the channels of ``(S, S)`` dimensional matrix.
    
    Parameters
    ----------
    landmarks: torch.Tensor
        A tensor of size ``(2*K, S, S)``.
    
    Returns
    -------
    torch.Tensor
    
    """
    lmk_xy = torch.zeros(landmarks.shape)
    
    m, K, S, _ = landmarks.shape  
    
    for i in range(m):
        j = 0
        while j < int(K):
            r, alpha = landmarks[i,j:j+2, ...]
            alpha = alpha * (2 * torch.pi)

            px = r * torch.cos(alpha)
            py = r * torch.sin(alpha)

            lmk_xy[i,j:j+2,...] = torch.cat([px.unsqueeze(0), py.unsqueeze(0)], dim=0)

            j += 2
    
    return lmk_xy

def yolo_filter(predictions, threshold=0.5):
    """Filters the yolo prediction boxes.
    
    Parameters
    ----------
    predictions: dict
        A dictionary that consists of all the types of predictions from the yolo output.
    threshold: float, default ``0.5``
        A threshold value for bounding box confidence.
        
    Returns
    -------
    dict
        A dict of box and values that extracts the bounding box data based on threshold.
        
    """
    # Extracting the box information
    box_confidence = predictions['confidence']
    box_xy = predictions['bboxes_xy']
    box_wh = predictions['bboxes_wh']
    box_lmk = predictions['landmarks']
    box_class_probs = predictions['classes']
    image_name = predictions['image_name']
    
    # Calculating box score
    m, _, S, _ = box_confidence[0].shape
    l = len(box_confidence)

    box_confidence = torch.cat(box_confidence, dim=0)
    box_confidence = box_confidence.reshape((m, l, S, S))
    
    
    box_scores = torch.multiply(box_confidence, box_class_probs)
    max_class_conf, index = torch.max(box_scores, dim=1)
    
    # Computing the box confidence and the position of best bounding box
    # responsible for prediction
    best_box_conf, best_box_index = torch.max(box_confidence, dim=1)
    
    # Using the best bounding box in the data
    predictor_xy = predictor_box(box_xy, best_box_index)
    predictor_wh = predictor_box(box_wh, best_box_index)
    predictor_lmk = predictor_box(box_lmk, best_box_index)
    
    if len(predictor_xy.shape) == 3:
        predictor_xy = predictor_xy.unsqueeze(0)
        
    if len(predictor_wh.shape) == 3:
        predictor_wh = predictor_wh.unsqueeze(0)
        
    if len(predictor_lmk.shape) == 3:
        predictor_lmk = predictor_lmk.unsqueeze(0)
    
    # Creating a mask using the threshold values for filtering
    mask = best_box_conf.ge(threshold)
    
    # Shape of the mask
    # m --> sample size
    m, r, c = mask.shape

    detection_grid = []
    detection_xy = []
    detection_wh = []
    detection_lmk = []
    detection_class_conf = []
    detection_class_label = []

    for s in range(m):
        detection_grid.append([])
        detection_xy.append([])
        detection_wh.append([])
        detection_lmk.append([])
        detection_class_conf.append([])
        detection_class_label.append([])

        for i in range(r):
            for j in range(c):
                if mask[s, i, j]:
                    detection_grid[s].append((i,j))
                    detection_xy[s].append(predictor_xy[s,:, i, j])
                    detection_wh[s].append(predictor_wh[s,:, i, j])
                    detection_lmk[s].append(predictor_lmk[s,:,i, j])
                    detection_class_conf[s].append(box_scores[s,:,i, j])
                    detection_class_label[s].append(int(torch.argmax(box_scores[s,:,i, j])))
    
    detections = {'grid': detection_grid, 
                  'xy': detection_xy,
                  'wh': detection_wh,
                  'landmarks': detection_lmk,
                  'class_confidence': detection_class_conf,
                  'image_name': image_name,
                  'class_label': detection_class_label
                 }
    
    return detections

def extract_rendering_data(boxes, img_shape=(224, 224), grid_size=7, polar_coord=True, num_landmarks=21):
    """Extracts data for rendering.
    
    Parameters
    ----------
    boxes: dict
        A dict with keys ``['grid', 'xy', 'wh', 'landmarks', 'class_confidence', 'image_name']``
        Obtained from ``yolo_filter()`` function.
    img_shape: tuple, default ``(224,224)``
        Target image shape
    grid_size: int, default ``7``
        Size of the yolo grid.
    polar_coord: bool, default ``True``
        Whether the landmarks are in polar coordinates.
    num_landmarks: int, default ``21``.
        The number of landmarks or keypoints.
    
    Returns
    -------
    dict
        A dictionary with keys ``['bounding_box', 'landmarks_xy', 'image_class', 'image_name']``
        
    """
    
    W, H = img_shape
    S = grid_size

    grid = boxes['grid']
    xy = boxes['xy']
    wh = boxes['wh']
    lmk = boxes['landmarks']
    class_conf = boxes['class_confidence']
    image_name = boxes['image_name']

    # Image numbers
    img_num = len(grid)

    xy_p = []
    wh_p = []
    lmk_p = []

    bounding_box = []
    landmarks_xy = []
    image_class = []
    class_confidence = []
    confidence_detection = []

    # Iterating through the images
    for n in range(img_num):
        bounding_box.append([])
        landmarks_xy.append([])
        image_class.append([])
        class_confidence.append([])
        confidence_detection.append([])
        for grid_, xy_, wh_, lmk_, class_conf_ in zip(grid[n], xy[n], wh[n], lmk[n], class_conf[n]):
            # Bounding box coordinate extraction
            # Extracting grid information
            x_grid, y_grid = grid_

            # Extracting x and y relative to cell
            x_cell, y_cell = xy_.tolist()
            w_img, h_img = wh_.tolist()

            # Converting x and y relative to cell to x and y relative to image
            x_img = (x_cell + x_grid) / S
            y_img = (y_cell + y_grid) / S

            # Creating a list of the bounding box coordinate [x, y, w, h]
            det_bbox = [x_img, y_img, w_img, h_img]

            # Appending to the bounding box list
            bounding_box[n].append(det_bbox)

            # landmarks
            lmk_p = lmk_.tolist()
            lmk_xy = []
            i = 0
            while i < int(num_landmarks*2):
                if polar_coord:
                    r, alpha = lmk_p[i:i+2]

                    alpha = alpha * (2 * np.pi)

                    px = r * np.cos(alpha)
                    py = r * np.sin(alpha)

                    px = x_img + px
                    py = y_img + py

                    lmk_xy.append(px)
                    lmk_xy.append(py)
                else:
                    px, py = lmk_p[i:i+2]
                    lmk_xy.append(px)
                    lmk_xy.append(py)

                i += 2
            lmk_xy = np.asarray(lmk_xy, dtype=float).reshape(-1,2)

            landmarks_xy[n].append(lmk_xy)

            # Image class
            class_pred = int(torch.argmax(class_conf_))
            image_class[n].append(class_pred)
            class_confidence[n].append(class_conf_)
            
            conf_det = torch.max(class_conf_)
            confidence_detection[n].append(conf_det)

    data = {'bounding_box': bounding_box,
            'landmarks_xy': landmarks_xy,
            'image_class': image_class,
            'image_name': image_name,
            'class_confidence': class_confidence,
            'confidence_detection': confidence_detection
           }
    return data

def extract_high_conf_boxes(filtered_boxes, conf_threshold=0.5):
    """Returns data based on the confidence threshold.
    
    Parameters
    ----------
    filtered_boxes: dict
        A dictionary from ``yolo_filter`` function.
    conf_threshold: float, default ``0.5``
        A confidence threshold used to eliminate any boxes with class confidence score below the ``conf_threshold``.
    
    Returns
    -------
    dict
        A dictionary similar to the input dictionary with the boxes below confidence threshold removed.
        Dictionary attributes:
        - ``grid``: The grid responsible for predicting the object. A list of tuples.
        - ``xy``: Coordinates of bounding box center. A list of tuples. 
        - ``wh``: Width and height of the bounding box. A list of tuples.
        - ``landmarks``: X and Y coordinates of the landmarks. A list of tuples.
        - ``image_name``: Name of the image for its prediction. A list.
        - ``class_label``: The label of class. A list.
    
    Examples
    --------
    >>> import torch.Tensor as tensor
    >>> filtered_boxes = {'grid': [[(2, 5), (3, 3)], [(3, 3)]],
                        'xy': [[tensor([0.4249, 0.4200]), tensor([0.4887, 0.5586])],
                        [tensor([0.5466, 0.5732])]],
                        'wh': [[tensor([0.7234, 0.6053]), tensor([0.3499, 0.4393])],
                        [tensor([0.3300, 0.5470])]],
                        'landmarks': [[tensor([0.5015, 0.2045, 0.6612, 0.6174, 0.3108, 0.6142, 0.3017, 0.4078, 0.5052,
                                0.6232, 0.4719, 0.2791, 0.4858, 0.6631, 0.4902, 0.4663, 0.4014, 0.6370,
                                0.7618, 0.5192, 0.4738, 0.3789, 0.5154, 0.5296, 0.5018, 0.5160, 0.5350,
                                0.6524, 0.3323, 0.3265, 0.2819, 0.4321, 0.2729, 0.7427, 0.6268, 0.3795,
                                0.4017, 0.4644, 0.5799, 0.4513, 0.6295, 0.5629]),
                        tensor([0.1519, 0.2422, 0.1223, 0.3806, 0.1429, 0.3072, 0.0679, 0.5526, 0.2126,
                                0.7724, 0.1113, 0.2452, 0.0942, 0.7150, 0.0790, 0.1675, 0.2222, 0.7141,
                                0.0839, 0.5405, 0.1198, 0.5832, 0.1882, 0.5563, 0.2314, 0.7414, 0.1286,
                                0.4857, 0.2017, 0.5567, 0.2153, 0.6046, 0.3392, 0.5060, 0.1370, 0.4668,
                                0.1802, 0.6444, 0.1912, 0.6844, 0.2249, 0.6341])],
                        [tensor([0.1705, 0.1884, 0.1263, 0.4050, 0.1384, 0.5039, 0.0520, 0.5292, 0.1101,
                                0.7032, 0.1235, 0.1361, 0.0758, 0.7285, 0.0607, 0.2070, 0.0946, 0.4740,
                                0.0396, 0.5527, 0.1168, 0.6569, 0.1272, 0.5881, 0.1953, 0.8032, 0.1051,
                                0.4681, 0.1067, 0.6021, 0.1953, 0.5714, 0.1705, 0.6113, 0.1954, 0.5020,
                                0.1669, 0.4305, 0.2172, 0.4953, 0.2182, 0.5083])]],
                        'class_confidence': [[tensor([0.0446, 0.0151]), tensor([0.2991, 0.6041])],
                        [tensor([0.6293, 0.3343])]],
                        'image_name': ['00000002_l.jpg', '00000002.jpg']}
    >>> extract_high_conf_boxes(filtered_boxes)
    {'grid': [[(3, 3)], [(3, 3)]],
    'xy': [[tensor([0.4887, 0.5586])], [tensor([0.5466, 0.5732])]],
    'wh': [[tensor([0.3499, 0.4393])], [tensor([0.3300, 0.5470])]],
    'landmarks': [[tensor([0.1519, 0.2422, 0.1223, 0.3806, 0.1429, 0.3072, 0.0679, 0.5526, 0.2126,
            0.7724, 0.1113, 0.2452, 0.0942, 0.7150, 0.0790, 0.1675, 0.2222, 0.7141,
            0.0839, 0.5405, 0.1198, 0.5832, 0.1882, 0.5563, 0.2314, 0.7414, 0.1286,
            0.4857, 0.2017, 0.5567, 0.2153, 0.6046, 0.3392, 0.5060, 0.1370, 0.4668,
            0.1802, 0.6444, 0.1912, 0.6844, 0.2249, 0.6341])],
    [tensor([0.1705, 0.1884, 0.1263, 0.4050, 0.1384, 0.5039, 0.0520, 0.5292, 0.1101,
            0.7032, 0.1235, 0.1361, 0.0758, 0.7285, 0.0607, 0.2070, 0.0946, 0.4740,
            0.0396, 0.5527, 0.1168, 0.6569, 0.1272, 0.5881, 0.1953, 0.8032, 0.1051,
            0.4681, 0.1067, 0.6021, 0.1953, 0.5714, 0.1705, 0.6113, 0.1954, 0.5020,
            0.1669, 0.4305, 0.2172, 0.4953, 0.2182, 0.5083])]],
    'class_confidence': [[tensor([0.2991, 0.6041])], [tensor([0.6293, 0.3343])]],
    'image_name': ['00000002_l.jpg', '00000002.jpg'],
    'class_label': [[1], [0]]}

    """
    # Empty list for storing confidence value mask
    conf_mask = []
    
    # Creating a dictionary structure
    data_dict = {'grid': [],
                 'xy': [],
                 'wh': [],
                 'landmarks': [],
                 'class_confidence': [],
                 'image_name': [],
                 'class_label': []
                }
    
    # Extracting class confidence score from the filtered_boxes dictionary
    class_confidence = filtered_boxes['class_confidence']
    
    # Creating a mask be going through class_confidence scores
    for idx, img_conf in enumerate(class_confidence):
        conf_mask.append([])
        for conf in img_conf:
            conf_mask[idx].append(conf.ge(conf_threshold).tolist())

    # Iterating over the number of images
    for i, img_conf in enumerate(conf_mask):
        # Creating a list for each image data to be separate
        data_dict['grid'].append([])
        data_dict['xy'].append([])
        data_dict['wh'].append([])
        data_dict['landmarks'].append([])
        data_dict['class_confidence'].append([])
        data_dict['class_label'].append([])
        
        # Iterating over the mask value for image
        for j, mask_val in enumerate(img_conf):
            # Iterating over the mask value for prediction
            for k, val in enumerate(mask_val):
                # Checking if the boolean value for the mask is true. This is above the conf_threshold
                if val:
                    data_dict['grid'][i].append(filtered_boxes['grid'][i][j])
                    data_dict['xy'][i].append(filtered_boxes['xy'][i][j])
                    data_dict['wh'][i].append(filtered_boxes['wh'][i][j])
                    data_dict['landmarks'][i].append(filtered_boxes['landmarks'][i][j])
                    data_dict['image_name'].append(filtered_boxes['image_name'][i])
                    
                    # Computing class label
                    class_conf_scores = filtered_boxes['class_confidence'][i][j]
                    class_label = int(torch.argmax(class_conf_scores))
                    
                    # Appending the values back to the dictionary
                    data_dict['class_confidence'][i].append(class_conf_scores)
                    data_dict['class_label'][i].append(class_label)

    return data_dict

def separate_labels(labels):
    """Separates the labels"""

    m, _ = labels.shape
    cl = []
    xy = []
    wh = []
    lmk = []
    for i in range(m):
        cl.append(labels[i][0])
        xy.append(labels[i][1:3])
        wh.append(labels[i][3:5])
        lmk.append(labels[i][5:])
        
    return cl, xy, wh, lmk

def evaluation_metric(prediction, target_data, filter_threshold=0.5, iou_threshold = 0.5, num_classes=2):
    """Calculates predictions"""

    filtered_boxes = yolo_filter(prediction, threshold=filter_threshold)

    final_pred = extract_high_conf_boxes(filtered_boxes)

    # storing the prediction values
    pred_xy = []
    pred_wh = []
    pred_xyxy = []
    pred_cls = []

    for i in range(len(final_pred['xy'])):
        pred_xy.append([])
        pred_wh.append([])
        pred_xyxy.append([])
        pred_cls.append([])
        for j in range(len(final_pred['xy'][i])):
            p_xy = final_pred['xy'][i][j].unsqueeze(0)
            p_wh = final_pred['wh'][i][j].unsqueeze(0)
            p_cl = final_pred['class_label'][i][j]
            p_xyxy = yolo_boxes_to_corners(p_xy, p_wh)

            pred_xy[i].append(p_xy)
            pred_wh[i].append(p_wh)
            pred_xyxy[i].append(p_xyxy)
            pred_cls[i].append(p_cl)
    # ------------
    # GROUND TRUTH
    # ------------
    target = copy.deepcopy(target_data)

    gt_cl, gt_xy, gt_wh, gt_lmk = separate_labels(target_data['labels'])
    
    gt_xyxy = [yolo_boxes_to_corners(xy_.unsqueeze(0),wh_.unsqueeze(0)) for xy_, wh_ in zip(gt_xy, gt_wh)]
    
    # -----------------
    # IOU and precision
    # -----------------
    
    ious = []
    
    evaluations = []
    eval_dict = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0,
    }
    
    confusion_matrix = torch.zeros((num_classes+1, num_classes+1))
    
    for n in range(num_classes):
        evaluations.append(copy.deepcopy(eval_dict))
    
    for i in range(len(pred_xyxy)):
        ious.append([])
        for j in range(len(pred_xyxy[i])):
            det_iou = iou(pred_xyxy[i][j], gt_xyxy[i])
            ious[i].append(det_iou)
            pred_class_label = int(pred_cls[i][j])
            gt_class_label = int(gt_cl[i])
            
            # Checking if the prediction class is empty i.e. no detection.
            # Update false negative for both the class
            if not pred_cls[i]:
                for n in range(num_classes):
                    evaluations[n]['FN'] += 1
                continue
                
            # Checking if the object is detected with IOU threshold   
            if det_iou > iou_threshold:
                # Checking if the class prediction matching ground truth class
                confusion_matrix[pred_class_label][gt_class_label] += 1
                if pred_class_label == gt_class_label:
                    # Checking if the ground truth is right hand
                    # Increment the TP counter for right hand
                    evaluations[pred_class_label]['TP'] += 1
                else:
                    evaluations[pred_class_label]['FP'] += 1
            else:
                evaluations[pred_class_label]['FP'] += 1
                    
    
    class_accuracy = []
    class_precision = []
    class_recall = []
    
    for n in range(num_classes):
        # Assigning the dictionary of the class in evaluations list
        e = evaluations[n]
        
        # Accuracy calculation
        try:
            acc = e['TP'] / (e['TP'] + e['TN'] + e['FP'] + e['FN'])
        except:
            acc = 0
        class_accuracy.append(acc)
        
        # precision calcuation
        try:
            prec = e['TP'] / (e['TP'] + e['FP'])
        except:
            prec = 0
        class_precision.append(prec)
        
        try:
            recall = e['TP'] / (e['TP'] + e['FN'])
        except:
            recall = 0
        class_recall.append(recall)

    return {"class_precision":class_precision,
            "class_accuracy": class_accuracy,
            "class_recall": class_recall,
            "ious": ious,
            "gt_xyxy": gt_xyxy,
            "pred_xyxy": pred_xyxy,
            "pred_cls": pred_cls,
            "evaluations": evaluations,
            "confusion_matrix": confusion_matrix
           }