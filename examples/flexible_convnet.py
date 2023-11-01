import sys
import os
import yaml
import time
sys.path.append('..')
from yolov1 import * 


model_file = '../models/LeNet.yaml'
train_annotations = '../data/dev/train.csv'
train_images = '../data/dev/train'
test_annotations = '../data/dev/test.csv'
test_images = '../data/dev/test'
train_loc = '../data/runs'
train_dir = 'train'

# Creating runs/train<timestamp> directory
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

epochs = 5

# Loading datasets
train_data = HandPoseDataset(annotations_file=train_annotations,
                             image_dir=train_images
                            )

test_data = HandPoseDataset(annotations_file=test_annotations,
                            image_dir=test_images
                           )

# Data loaders
train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

# Load model file
with open(model_file, 'r') as f:
    model_architecture = yaml.safe_load(f)

architecture = model_architecture['architecture']


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Using LeNet5 model

model = ConvNet(architecture, input_size=(3,224,224)).to(device)

print(model)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Training
for t in range(epochs):
    dt = time.time()
    print(f"Epoch{t+1}\n------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)
    time_elapsed = time.time() - dt 
    if time_elapsed < 100:
        print(f"Time Elapsed: {time_elapsed:>5f}s")
    else:
        print(f"Time Elapsed: {time_elapsed/3600:>5f}h")
    torch.save(model.state_dict(), os.path.join(train_path, "last.pth"))
print("Done!")

# Saving the final model
torch.save(model.state_dict(), best_model_path)
print(f"Saved Pytorch Model State to {best_model_path}")

