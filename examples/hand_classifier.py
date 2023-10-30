import sys 
sys.path.append('..')
from yolov1 import * 


train_annotations = '../data/dev/train.csv'
train_images = '../data/dev/train'
test_annotations = '../data/dev/test.csv'
test_images = '../data/dev/test'
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
model = LeNet().to(device)

print(model)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


for t in range(epochs):
    print(f"Epoch{t+1}\n------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)
print("Done!")