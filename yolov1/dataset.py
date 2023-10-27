import torch 
import os 
import pandas as pd 
from PIL import Image
import cv2
import pickle
import copy 
import time 
import sys 
import os 
import numpy as np 
from tqdm import tqdm 
from torchvision.io import read_image 
import torch.nn.functional as F 
import torchvision.transforms as transforms 


class FreiHand:
    """A class that extracts the annotation and stores them in YOLO format.
    
    Parameters
    ----------
    path: str
        The root to the path where the data is stored.
    file_type: str, default ``'training'``
        The type of data used.
        Options: ``'training'``, ``'evaluation'``
    mask_dir: str
        The directory where the mask images are stored.
    
    Attributes
    ----------
    K_array: list
        A list of the intrinsic camera matrix.
    verts_array: list
        A list of verts.
    xyz_array: list
        A list of xyz coordinates of hand landmark.
    uv: dict
        A dictionary that maps the image name to the landmarks.
    image_filenames: list
        A list of all the image files.
    EDGES: list
        A list of landmark point graph connection.
    images_shape: dict
        A dictionary that maps image file names to the shape of the image.
    yolo_annot: dict
        A dictionary that maps the image file names to YOLO annotations.
        YOLO format: [class x y w h]
    yolo_pose: dict
        A dictionary that maps the image file names to YOLO pose annotations.
        YOLO pose format: [class x y w h px1 py1 ... px21 py21]
        Pose consists of YOLO annotations along with 21 hand landmark.
    bounding_box: dict
        A dictionary that maps the image file names to bounding box coordinates.
        format: [xmin, ymin, xmax, ymax]
    annotation_df: None
        This is used to store the ``pandas.DataFrame`` when the annotations are converted to dataframe.
    hand_contours: dict
        Use to store the contours of the hand from the mask images.
        Image file names are mapped to the ``numpy.array`` of contour.
        
    Methods
    -------
    load_json_files()
        Loads the json files.
    convert_json_to_pickle()
        Converts json to pickle files.
    load_data_files()
        Loads the pickle files
    read_image_files(images_path='rgb')
        Reads image files to store the image shapes.
    project_landmarks()
        Converts the pose coordinates from the datafiles to YOLO and YOLO pose formats.
    save_images(save_path='.', image_location = 'rgb' , directory='Freihand_images', image_extension='.jpg')
        Saves the images that has annotations to the given directory.
    mask_contour(threshold=0, maxval=255, threshold_type=cv2.THRESH_BINARY, retrieval_mode=cv2.RETR_EXTERNAL, contour_approx_mode=cv2.CHAIN_APPROX_SIMPLE)
        Reads the segmentation mask images.
    generate_annotations(save_csv=False, save_path='.', file_name="annotations.csv", image_extension='.jpg', save_index=False, annot_col=["image_name","class","x","y","w","h","px1","py1","px2","py2","px3","py3","px4","py4","px5","py5","px6","py6","px7","py7","px8","py8","px9","py9","px10","py10","px11","py11","px12","py12","px13","py13","px14","py14","px15","py15","px16","py16","px17","py17","px18","py18","px19","py19","px20","py20","px21","py21"])
        Generates the annotations and saves them in a pandas.DataFrame object called ``annotations_df``.
        If the boolean ``save_csv=True`` then the csv file is saved.

    Examples
    --------
    >>> from lfdtrack import *
    >>> data_path = '~/path/to/data/FreiHand/'
    >>> training = FreiHand(data_path, mask_dir='mask', file_type='training') # mask_dir='segmap' --> evaluation data
    >>> training.load_data_files()
    >>> training.read_image_files()
    >>> training.mask_contour()
    >>> training.project_landmarks()
    >>> training.generate_annotations(file_name='annotations.csv')
    >>> training.save(annotations=training.yolo_pose, directory='FreiHand_training_labels')
    >>> training.save_images(directory='Friehand_training')
    
    """
    
    def __init__(self, path, mask_dir, file_type='training'):
        self.path = path
        
        # Path where the segmentation masks are stored
        self.mask_dir = mask_dir
        
        # Hand contour
        self.hand_contours = dict()
        
        self.file_type = file_type
        
         # Camera calibration matrix
        self.K_array = []
        
        # Vertices
        self.verts_array = []
        
        # XYZ coordinates
        self.xyz_array = []
        
        # Image coordinates
        self.uv = dict()
        
        # Images list
        self.image_filenames = []
        
        # annotations dataframe
        self.annotation_df = None
        
        self.mask_filenames = []
        
        # graph
        self.EDGES = [[0,1], [1,2], [2,3], [3,4], 
                      [0,5], [5,6], [6,7], [7,8],
                      [0,9], [9,10],[10,11], [11,12],
                      [0,13],[13,14], [14,15], [15,16],
                      [0,17],[17,18], [18,19], [19,20]]
        
        # image shapes
        # shape format (H, W) --> numpy shape format for (row, col)
        self.images_shape = dict()
        
        self.yolo_annot = dict()
        self.yolo_pose = dict()
        self.bounding_boxes = dict()
        
    def load_json_files(self):
        """Loads the json file.
        
        Paramters
        ---------
        self.file_type: str, default ``'training'``
            The type of files to pick up.
            Available options: ``'training'``, ``'evaluation'``
            
        """
        
        start = time.time()
        
        with open(f'{self.path}/{self.file_type}_K.json') as K_fp:
            print("Reading K...")
            self.K_array = json.load(K_fp)
            
        with open(f'{self.path}/{self.file_type}_verts.json') as verts_fp:
            print("Reading verts...")
            self.verts_array = json.load(verts_fp)
            
        with open(f'{self.path}/{self.file_type}_xyz.json') as xyz_fp:
            print("Reading xyz...")
            self.xyz_array = json.load(xyz_fp)
            
        end = time.time()
        
        time_elapsed = end - start
        
        print(f"Time elapsed: {time_elapsed:.2f}s")
        
    def convert_json_to_pickle(self):
        """Converts all the files to pickle."""
        
        files = []
        
        with os.scandir(self.path) as entries:
            for entry in entries:
                file, extension = os.path.splitext(entry.name)
                if extension == '.json':
                    files.append(file)
                    
        for file in files:
            json_location = os.path.join(self.path, file + '.json')
            pickle_location = os.path.join(self.path, file + '.pickle')
            
            with open(json_location) as f:
                print(f'Reading {json_location}...')
                json_file = json.load(f)
                
            with open(pickle_location, 'wb') as pf:
                print(f'Saving {pickle_location}...')
                pickle.dump(json_file, pf)
                
            del json_file
            
    def load_data_files(self):
        """Loads the pickle data files.
        
        Paramters
        ---------
        self.file_type: str, default ``'training'``
            The type of files to pick up.
            Available options: ``'training'``, ``'evaluation'``
        
        """
        
        start = time.time()
        
        with open(f'{self.path}/{self.file_type}_K.pickle', 'rb') as K_fp:
            print(f"Reading {self.file_type}_K...")
            self.K_array = pickle.load(K_fp)
            
        with open(f'{self.path}/{self.file_type}_verts.pickle', 'rb') as verts_fp:
            print("Reading verts...")
            self.verts_array = pickle.load(verts_fp)

        with open(f'{self.path}/{self.file_type}_xyz.pickle', 'rb') as xyz_fp:
            print("Reading xyz...")
            self.xyz_array = pickle.load(xyz_fp)
            
        end = time.time()
        
        time_elapsed = end - start
        
        print(f"Time elapsed: {time_elapsed:.2f}s")
            
    def read_image_files(self, images_path='rgb'):
        """Reads and populates the images file list.
        
        Parameters
        ----------
        images_path: str, default ``'rgb'``
            The directory where the images are stored.
        
        """
        # Using the full path for the images
        # root path + training/evalutate + 'rgb'
        full_path = os.path.join(self.path, self.file_type, images_path)
        
        with os.scandir(full_path) as entries:
            for entry in tqdm(entries):
                file, extension = os.path.splitext(entry.name)
                self.image_filenames.append(file)
                
                image = cv2.imread(os.path.join(full_path, entry))
                
                # adding the shape
                self.images_shape[file] = image.shape
                
                # deleting the image to free memory
                del image
                
        # sorting the list
        self.image_filenames.sort()
        
        # sorting the image shape dictionary
        self.images_shape = dict(sorted(self.images_shape.items()))
        
    def mask_contour(self, 
                     threshold=0, 
                     maxval=255, 
                     threshold_type=cv2.THRESH_BINARY,
                     retrieval_mode=cv2.RETR_EXTERNAL,
                     contour_approx_mode=cv2.CHAIN_APPROX_SIMPLE
                    ):
        """Reads the segmentation mask images.
        
        Parameters
        ----------
        threshold: int, default ``0``
            The threshold value for the thresholding operation.
        maxval: int, default `255``
            The maximum value to keep during thresholding.
        threshold_type: int, default ``cv2.THRESH_BINARY``
            The type of thresholding operation.
            The default value is the enum in opencv that corresponds to ``0``.
        retrieval_mode: int, default ``cv2.RETR_EXTERNAL``
            The type of contouring operation.
            Since we are interested in getting the outer boundary of the hand,
            we choose external contour.
        contour_approx_mode: int, default ``cv2.CHAIN_APPROX_SIMPLE``
            The contour approximation mode.
            Enum in opencv.
            
        """
        # Using the full path for the images
        # root path + training/evalutate + 'mask'
        full_path = os.path.join(self.path, self.file_type, self.mask_dir)
        
        with os.scandir(full_path) as entries:
            for entry in tqdm(entries):
                file, extension = os.path.splitext(entry.name)
                # Reading grayscale image for segmentation mask
                mask = cv2.imread(os.path.join(full_path, entry), 0)
                _, thresh_image = cv2.threshold(mask, threshold, maxval, threshold_type)
                
                # Find contours
                contours, _ = cv2.findContours(thresh_image, retrieval_mode, contour_approx_mode)
                
                self.hand_contours[file] = contours[0][:,0]
                
                self.mask_filenames.append(file)
        
    def project_landmarks(self):
        """Projects the landmarks"""
        
        for K, xyz, file in zip(tqdm(self.K_array), self.xyz_array, self.image_filenames):
            xyz_i = np.array(xyz)
            K_i = np.array(K)
            
            # Matrix multiplication for the camera intrinsic matrix and the homogenous coordinates
            uv_i = np.matmul(K_i, xyz_i.T).T
            
            # converting the homogenous coordinates
            landmarks = (uv_i[:, :2] / uv_i[:,-1:]).astype(np.int32)
            
            # number of landmarks
            n_landmarks, _ = uv_i.shape
            
            self.uv[file] = landmarks
            
            # get image shapes
            H, W, _ = self.images_shape[file]
            
            pose = []
            
            # Getting the contour detected for the hand using mask
            contour = self.hand_contours[file]
            
            # bounding box
            x = contour[:,0]
            y = contour[:,1]
            
            # min values
            xmin = np.min(x)
            ymin = np.min(y)
            
            # max values
            xmax = np.max(x)
            ymax = np.max(y)
            
            box_height = ymax - ymin
            box_width = xmax - xmin
            
            yolo_coord = [0, 
                          (xmin + box_width/2) / W, 
                          (ymin + box_height/2) / H, 
                          box_width / W,
                          box_height / H
                         ]
            
            x_ = np.array(landmarks[:, 0] / W).reshape((n_landmarks, 1))
            y_ = np.array(landmarks[:, 1] / H).reshape((n_landmarks, 1))
            
            pose_xy = np.concatenate((x_, y_), axis=1)
            
            for n in pose_xy:
                pose.append(n[0])
                pose.append(n[1])
            
            pose_coord = yolo_coord + pose
            
            self.yolo_annot[file] = yolo_coord
            self.yolo_pose[file] = pose_coord
                
            # Bounding boxes | format: [xmin, ymin, xmax, ymax]
            self.bounding_boxes[file] = [xmin, ymin, xmax, ymax]                
            
    def save_images(self, save_path='.', image_location = 'rgb' , directory='Freihand_images', image_extension='.jpg'):
        """Reads and writes the images.
        
        Parameters
        ----------
        save_path: str, default ``'.'``
            Save location of the images.
        image_location: str, default ``'rgb'``
            Location where to look for the images in the ``'training'`` or ``'evaluation'`` directory.
        directory: str, default ``'Freihand_images'``
            Directory to store the images.
        image_extension: str, default ``'.jpg'``
            Extension of the images.
            
        """
        
        image_dir_path = os.path.join(save_path, directory)
        
        if not os.path.exists(image_dir_path):
            os.makedirs(image_dir_path)
            print(f"New directory {directory} created at {image_dir_path}")
        
        files = list(self.yolo_pose.keys())
        
        for fn in tqdm(files):
            image_path = os.path.join(self.path, self.file_type, image_location, fn + image_extension)
            image_save_path = os.path.join(image_dir_path, fn + image_extension)
            
            try:
                image = cv2.imread(image_path)
                cv2.imwrite(image_save_path, image)
                del image
            except Exception as e:
                print(e)
                continue
                
    def generate_annotations(self, save_csv=False, save_path='.', file_name="annotations.csv", image_extension='.jpg', save_index=False, annot_col=["image_name","class","x","y","w","h","px1","py1","px2","py2","px3","py3","px4","py4","px5","py5","px6","py6","px7","py7","px8","py8","px9","py9","px10","py10","px11","py11","px12","py12","px13","py13","px14","py14","px15","py15","px16","py16","px17","py17","px18","py18","px19","py19","px20","py20","px21","py21"]):
        """Saves annotation to csv file.
        
        Parameters
        ----------
        save_csv: bool, default ``False``
            A boolean to save the annotations.
        save_path: str, default ``'.'``
            Path where the file is to be saved.
        file_name: str, default ``"annotations.csv"``
            Name of the file.
        image_extension: str, default ``'.jpg'``
            Extension of the image to add to the csv ``image_name`` column.
        save_index: bool, default ``False``
            If ``False`` then it does not add an index column to csv file while saving.
        annot_col: list, default ``["image_name","class","x","y","w","h","px1","py1","px2","py2","px3","py3","px4","py4","px5","py5","px6","py6","px7","py7","px8","py8","px9","py9","px10","py10","px11","py11","px12","py12","px13","py13","px14","py14","px15","py15","px16","py16","px17","py17","px18","py18","px19","py19","px20","py20","px21","py21"]``
            Column names for the annotation.
            
        """
        # combine path names
        save_file_path = os.path.join(save_path, file_name) 
        
        d = []
        
        for k, v in self.yolo_pose.items():
            image_name = k + image_extension
            row = [image_name] + v
            d.append(row)
            
        df = pd.DataFrame(d, columns=annot_col)
        
        self.annotation_df = copy.deepcopy(df)

        if save_csv:
            df.to_csv(save_file_path, index=save_index)
                   
    def save_annotations(self, save_path='.', directory='combined_labels', label_extension='.txt'):
        """Saves the annotations.
        
        Parameters
        ----------
        save_path: str, default ``'.'``
            The path where the result directory and files are to be saved.
        directory: str, default ``'labels'``
            The directory name under which the labels text files are to be saved.
            
        """
        # Labels path to read the labels from
        labels_path = os.path.join(save_path, directory)
        
        # Creating a directory if it does not exist
        if not os.path.exists(labels_path):
            os.makedirs(labels_path)
            print(f"New directory {directory} created at {labels_path}")
            
        if self.yolo_annot:
            for k, v in self.yolo_annot.items():
                label_file = os.path.join(labels_path, k + label_extension)

                yolo_annotation = self.yolo_pose[k]

                with open(label_file, 'w') as file:
                    for label in yolo_annotation:
                        file.write('%s' % label)
                        file.write(' ')
                    file.write('\n')

def mirror_annotations(labels, escape_cols=2, u=1, v=0):
    """Mirrors the coordinates along y axis and translates them.
    
    Parameters
    ----------
    labels: pandas.DataFrame
        A dataframe of right hand annotations.
    escape_cols: int, default ``2``
        The number of columns to skip which could contain data like ``image_name`` and ``class``.
    u: int, default ``1``
        Translation of x coordinates.
        For normalised coordinates, it is ``1`` otherwise the size of the image.
    v: int, default ``1``
        Translation of y coordinates.
        Since reflecting along y axis and translating along x axis, the value of ``v`` is ``1``.
    
    Returns
    -------
    df_combined: pandas.DataFrame
        A combined dataframe of the annotations of both left and right hands.
        
    """
    # Extracting the coordinates from labels pandas.DataFrame and transposing the matrix
    M = np.array(labels.iloc[:, escape_cols:]).T

    # n -> number of features
    # m -> number of datapoints
    n, m = M.shape
    
    # Identify matrix
    I = np.eye(n+1)
    
    # vector of ones
    h = np.ones((1, m))

    # vertical stacking the vector for homogenous coordinates
    Mh = np.vstack([M, h])
    
    # creating vector k to multiply with identity matrix where x coordinates
    # become negative and y remains positive
    k1 = np.array([-1, 1, 1, 1])
    k2 = np.array([-1, 1] * 21)
    k3 = np.array([1])
    
    # horizontal stacking the vectors and reshaping them
    k = np.hstack([k1, k2, k3]).reshape(n+1,1)
    
    # Element-wise multiplication
    T = np.multiply(I, k)
    
    # column vector for translation of the coordinates
    c1 = np.array([u,v, 0, 0])
    c2 = np.array([u,v] * 21)
    c3 = np.array([1])
    
    # horizontal stacking the vectors created.
    c = np.hstack([c1, c2, c3]).reshape((n+1))
    
    # Changing the last column of the transformation matrix.
    T[:,-1] = c
    
    # Multiplying the transformation matrix T and Mh 
    Gh = np.dot(T, Mh)
    
    # Removing the last row of ones and transforming
    G = Gh[:-1,:].T
    
    # extracting the list of images
    image_list = list(labels.iloc[:,0])
    
    # Getting the extension of the images
    image_extension = image_list[0].rsplit('.', 1)[-1]
    
    # Separating the image name from its extension
    image_names = [image_name.rsplit('.', 1)[0] for image_name in image_list]
    
    # adding a subscript ``_l`` to the mirrored images indicating they are left hand now
    mirror_image_names = np.array([f"{i}_l.{image_extension}" for i in image_names]).reshape((m,1))
    
    # Creating a vector of labels for the left hand (right hand --> 0, left hand --> 1)
    image_class = np.ones((m,1))
    
    # stacking all the columns together to create an array of dataset entry
    df_m = np.hstack([mirror_image_names, image_class, G])
    
    # Column headings of the dataset
    columns = list(labels.columns)
    
    # Collecting the data from the input pandas.DataFrame
    labels_m = np.array(labels.iloc[:,:])
    
    # Vertically stacking the two datasets
    combined_data_m = np.vstack([labels_m, df_m]) 
    
    # Converting the newly created dataset into one pandas.DataFrame
    df_combined = pd.DataFrame(combined_data_m, columns=columns, index=None)
    
    return df_combined

class HandPoseDataset(Dataset):
    def __init__(self, annotations_file, image_dir, transform=None, target_transform=None):
        self.image_labels = pd.read_csv(annotations_file)
        self.image_dir = image_dir 
        self.transform = transform 
        self.target_transform = target_transform 

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_labels.iloc[idx, 0])
        # Reading grayscale image
        # TODO: change to color image in next development
        image = cv2.imread(image_path, 0)
        
        # Vector of labels
        label = self.image_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return (image / 255).astype(np.float32), label.astype(np.int_)

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                
                boxes.append([class_label, x, y, width, height])
                
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)
        
        if self.transform:
            image, boxes = self.transform(image, boxes)
            
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S * y), int(self.S * x)
            
            # Making x and y coordinates relative to the cell
            x_cell, y_cell = self.S * x - j, self.S * y - i
            
            # TODO: Making width and height relative to the cell.
            # This is INCORRECT. The paper states that the width and height are relative to the image.
            # Implementing this from the tutorials as other helper functions for rendering may rely on this.
            # TODO: Fix this in the later implementation
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )
            
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1
                
        return image, label_matrix