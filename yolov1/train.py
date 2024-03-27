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
import pickle

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

def train_model(dataloaders, 
                model, 
                criterion, 
                optimizer,
                scheduler,
                dataset_sizes,
                batch_size,
                num_epochs=100, 
                num_boxes=2, 
                grid_size=7, 
                num_landmarks=21, 
                num_classes=2, 
                verbose=True, 
                save_model_path='../data/runs',
                train_dir='train'
               ):
    """Training function"""

    # Record start time
    since = time.time()

    if save_model_path:
        timestamp = str(int(since))
        train_dir = train_dir + '_' + timestamp
        train_path = os.path.join(save_model_path, train_dir)
    
        # Last model path to save model after every epoch
        last_model_path = os.path.join(train_path, "last.pt")
    
        # final model that is saved after the training is complete
        best_model_path = os.path.join(train_path, "best.pt")

    # Setting all the losses to dictionary
    all_losses = {'train': dict(), 'valid': dict()}

    best_acc = 0.0

    for epoch in tqdm(range(num_epochs), unit='batch', total=num_epochs):
        print(f"Epoch: {epoch + 1}")
        print(f"{len('Epoch: ' + str(epoch+1)) * '-'}")

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train() # set model to training mode
            else:
                model.eval()
        
            running_loss = 0.0
            running_corrects = 0
            # last_loss = 0.0

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            for i, data in enumerate(dataloaders[phase], start=0):
                inputs = data['image'].to(device)

                # Setting the target data to device
                for k, v in data.items():
                    if type(v) == torch.Tensor:
                        data[k] = v.to(device)
                
                # Assigning the current batch size
                current_batch_size = inputs.size(0)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
        
                    pred = yolo_head(outputs, 
                                     num_boxes=num_boxes, 
                                     num_landmarks=num_landmarks, 
                                     num_classes=num_classes, 
                                     grid_size=grid_size, 
                                     batch_size=current_batch_size, 
                                     image_name=data['image_name'])

                    
                    losses = criterion(outputs, data, batch_size=current_batch_size)
                    # import pdb;pdb.set_trace()
                    loss = losses['total_loss']

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
        
            print(f'{phase}: ', end='')
            for k, v in losses.items():
                print(f"{k}: {v:.3f}, ", end="")
                if k not in all_losses[phase]:
                    all_losses[phase][k] = []
                all_losses[phase][k].append(float(v))
            print("\n")

            # deep copy the model
            if save_model_path:
                if not os.path.exists(train_path):
                    os.makedirs(train_path)
                    print(f"New directory {train_dir} created at {train_path}")
                
                torch.save(model.state_dict(), last_model_path)
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    if save_model_path:
        torch.save(model.state_dict(), best_model_path)
        with open(os.path.join(train_path, 'all_losses.pickle'), 'wb') as handle:
            pickle.dump(all_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

    history = {'model': model,
               'all_losses': all_losses
              }
    return history
                
    print('Finished Training')