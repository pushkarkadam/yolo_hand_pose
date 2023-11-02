import torch
import torchvision.transforms as transforms 
import torch.optim as optim 
import torchvision.transforms.functional as FT 
from tqdm import tqdm 
from torch.utils.data import DataLoader
import time
from .model import *
from .dataset import *
from .utils import *
from .loss import *
from .convnet import *
import yaml

seed = 123 
torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False 
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "../data/images"
LABEL_DIR = "../data/labels"


class ModelTrain:
    """Trains the model"""
    def __init__(self, model_config):
        """Trains the model.
        
        Parameters
        ----------
        model_config: str
            A ``.yaml`` file path.
            Example: ``'../data/config.yaml'``
            YAML file should be as follows:
            ```
            file_paths:
                model_file: ../models/LeNet.yaml
                train_annotations: ../data/dev/train.csv
                train_images: ../data/dev/train
                test_annotations: ../data/dev/test.csv
                test_images: ../data/dev/test
                train_loc: ../data/runs
                train_dir: train

            hyperparameters:
                epochs: 5
                batch_size: 64
            ```
        model_class: class
            A class that is comp
        """

        self.architecture = None
        self.model = None
        # Loading YAML file
        with open(model_config, "r") as f:
            config_params = yaml.safe_load(f)

        # Extracting file paths
        self.file_paths = config_params['file_paths']

        # Extracting hyperparameters
        self.hyperparameters = config_params['hyperparameters']

        self.dataset_params = config_params['dataset_parameters']

        # Loading datasets
        self.train_data = HandPoseDataset(annotations_file=self.file_paths['train_annotations'],
                                    image_dir=self.file_paths['train_images']
                                    )

        self.test_data = HandPoseDataset(annotations_file=self.file_paths['test_annotations'],
                                    image_dir=self.file_paths['test_images']
                                )

        # Data loaders
        self.batch_size = self.hyperparameters['batch_size']
        self.shuffle = self.dataset_params['shuffle']
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=self.shuffle)

        # Load model
        self.architecture = load_architecture(self.file_paths['model_file'])

        # Create model
        # Get cpu, gpu or mps device for training.
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")

        # Using LeNet5 model
        self.model = ConvNet(self.architecture, input_size=tuple(self.dataset_params['input_size'])).to(self.device)

        print(self.model)

    def fit(self):
        """Trains the network"""
        # Loss and optimizer
        # TODO: Modify this when implementing YOLO. Put in a different submodule for loss.
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=float(self.hyperparameters['learning_rate']))

        # Training save directory setup
        # Creating runs/train<timestamp> directory
        train_loc = self.file_paths['train_loc']
        train_dir = self.file_paths['train_dir']
        timestamp = str(int(time.time()))
        train_dir = train_dir + '_' + timestamp
        train_path = os.path.join(train_loc, train_dir)

        # Last model path to save model after every epoch
        last_model_path = os.path.join(train_path, "last.pth")

        # final model that is saved after the training is complete
        best_model_path = os.path.join(train_path, "best.pth")

        if not os.path.exists(train_path):
            os.makedirs(train_path)
            print(f"New directory {train_dir} created at {train_path}")

        # Training
        epochs = self.hyperparameters['epochs']
        for t in range(epochs):
            dt = time.time()
            print(f"Epoch {t+1}\n------------------------------------")
            train(self.train_dataloader, self.model, loss_fn, optimizer, self.device)
            test(self.test_dataloader, self.model, loss_fn, self.device)
            time_elapsed = time.time() - dt 
            if time_elapsed < 100:
                print(f"Time Elapsed: {time_elapsed:>5f}s")
            else:
                print(f"Time Elapsed: {time_elapsed/3600:>5f}h")
            torch.save(self.model.state_dict(), os.path.join(train_path, "last.pth"))
        print("Done!")

        # Saving the final model
        torch.save(self.model.state_dict(), best_model_path)
        print(f"Saved Pytorch Model State to {best_model_path}")

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    dt = time.time()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.type(torch.LongTensor)
        y = y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            time_elapsed = time.time() - dt
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}] [Time Elapsed: {time_elapsed/3600:>4f} h]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.type(torch.LongTensor) 
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, img, bboxes):
#         for t in self.transforms:
#             img, bboxes = t(img), bboxes

#         return img, bboxes

# transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

# def train_fn(train_loader, model, optimizer, loss_fn):
#     loop = tqdm (train_loader, leave=True)
#     mean_loss = []

#     for batch_idx, (x, y) in enumerate(loop):
#         x, y = x.to(DEVICE), y.to(DEVICE)
#         out = model(x)
#         loss = loss_fn(out, y)
#         mean_loss.append(loss.item())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # update the progress bar 
#         loop.set_postfix(loss = loss.item())

#     print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

# def main():
#     model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
#     optimizer = optim.Adam(
#         model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
#     )
#     loss_fn = YoloLoss()

#     if LOAD_MODEL:
#         load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

#     train_dataset = VOCDataset(
#         "../data/8examples.csv",
#         transform=transform,
#         img_dir=IMG_DIR,
#         label_dir=LABEL_DIR
#     )

#     test_dataset = VOCDataset(
#         "../data/test.csv",
#         transform=transform,
#         img_dir=IMG_DIR,
#         label_dir=LABEL_DIR
#     )

#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=BATCH_SIZE,
#         num_workers=NUM_WORKERS,
#         pin_memory=PIN_MEMORY,
#         shuffle=True,
#         drop_last=False,
#     )

#     test_loader = DataLoader(
#         dataset=test_dataset,
#         batch_size=BATCH_SIZE,
#         num_workers=NUM_WORKERS,
#         pin_memory=PIN_MEMORY,
#         shuffle=True,
#         drop_last=True,
#     )

#     for epoch in range(EPOCHS):
#         pred_boxes, target_boxes = get_bboxes(
#             train_loader, model, iou_threshold=0.5, threshold=0.4
#         )

#         mean_avg_prec = mean_average_precision(
#             pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
#         )

#         print(f"Train mAP: {mean_avg_prec}")

#         train_fn(train_loader, model, optimizer, loss_fn)

# if __name__ == "__main__":
#     main()