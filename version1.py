import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
from torchvision.io import read_image
import csv

class Net(nn.Module): # TODO: add complexity and activation functions
    def __init__(self, start_dim, end_dim, bias=True):
        super(Net, self).__init__()
        self.conv2d1 = nn.Conv2d(start_dim, 5, 3, bias=bias)
        self.conv2d2 = nn.Conv2d(5, 10, 3, bias=bias)
        self.conv2d3 = nn.Conv2d(10, 3, 3, bias=bias)
        self.conv2d4 = nn.Conv2d(3, 1, 3, bias=bias)
        self.linear1 = nn.Linear(33600, 8192, bias=bias)
        self.linear2 = nn.Linear(8192, 1024, bias=bias)
        self.linear3 = nn.Linear(1024, 32, bias=bias)
        self.linear4 = nn.Linear(32, end_dim, bias=bias)

    def forward(self, x):
        y = self.conv2d1(x)
        y = self.conv2d2(y)
        y = self.conv2d3(y)
        y = self.conv2d4(y)
        y = y.squeeze(1).flatten(start_dim=1)
        y = self.linear1(y)
        y = self.linear2(y)
        y = self.linear3(y)
        y = self.linear4(y)
        return y # shape [BATCH_SIZE, NUM_CLASS]
    
def calc_loss(logits, y):
    loss_func = nn.CrossEntropyLoss(reduction='none') # TODO: look into weights parameter
    loss = loss_func(logits, y)
    return loss # shape [BATCH_SIZE]

def calc_accuracy(logits, y):
    y_hat = torch.argmax(logits, 1)
    accuracy = y_hat == y
    return accuracy # shape [BATCH_SIZE]

### SET PARAMETERS

# HYPERPARAMETERS
SEED            = 0
BATCH_SIZE      = 1024
LEARNING_RATE   = 0.01
MAX_GRAD_NORM   = 2
MAX_STEPS       = 10
LOG_INTERVAL    = 1 # doesn't affect learning

# MODEL PARAMETERS
# DIM_EMBED       = 16
# KERNEL_SIZE     = 3

# LOAD CHECKPOINT INFORMATION
CP_TIME         = ''
CP_STEP         = 0

### READ DATA
NUM_CLASS  = 4

label_to_int = {'NonDemented': 0, 'VeryMildDemented': 1, 'MildDemented': 2, 'ModerateDemented': 3}

x_train = []
y_train = []
data_path = 'train'
for folder in os.listdir(data_path)[1:]:
    folder_path = os.path.join(data_path, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        image = torchvision.io.read_image(file_path)
        x_train.append(image)
        y_train.append(folder)
x_train = torch.stack(x_train, 0) / 256
y_train = torch.tensor([label_to_int[y] for y in y_train])
# y_train_onehot = torch.scatter(torch.zeros(y_train.shape[0], NUM_CLASS), 1, y_train.unsqueeze(1), torch.ones(y_train.shape[0], 1))

x_test = []
y_test = []
data_path = 'test'
for folder in os.listdir(data_path)[1:]:
    folder_path = os.path.join(data_path, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        image = torchvision.io.read_image(file_path)
        x_test.append(image)
        y_test.append(folder)
x_test = torch.stack(x_test, 0) / 256
y_test = torch.tensor([label_to_int[y] for y in y_test])
# y_test_onehot = torch.scatter(torch.zeros(y_test.shape[0], NUM_CLASS), 1, y_test.unsqueeze(1), torch.ones(y_test.shape[0], 1))

TRAIN_SIZE = x_train.shape[0]
TEST_SIZE  = x_test.shape[0]
NUM_CHANS  = x_train.shape[1]
NUM_ROWS   = x_train.shape[2]
NUM_COLS   = x_train.shape[3]

### INITIALIZE MODEL
model = Net(NUM_CHANS, NUM_CLASS)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

### INITIALIZE SAVE DIRECTORY
characteristics = f'version{1}'
time_id         = time.strftime('%Y-%m-%d %H-%M-%S')
save_dir        = os.path.join(characteristics, time_id)
os.makedirs(save_dir)

### LOAD CHECKPOINT # TODO: move checkpoint to utils.py
cp_path = os.path.join(characteristics, CP_TIME, f'{CP_STEP}.pt')
if os.path.isfile(cp_path):
    print('Loading checkpoint...\n')
    checkpoint = torch.load(cp_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    print('Starting with no checkpoint...\n')

### SET LOG WRITER # TODO: move log to utils.py
log_name = f'{characteristics}.csv'
log_path = os.path.join(save_dir, log_name)
with open(log_path, 'w', newline='') as f:
    header = ['Step', 'Mean_train_loss', 'Train_accuracy', 'Mean_test_loss', 'Test_accuracy']
    writer = csv.writer(f)
    writer.writerow(header)

### TRAIN POLICY # TODO: requires_grad = True?
print('Training...\n')
start_time = time.time()
for epoch in range(1, MAX_STEPS + 1):
    loss_train = torch.empty((0))
    accuracy_train = torch.empty((0))
    # TODO: look into DataLoader for shuffling and batching
    shuffle = torch.randperm(TRAIN_SIZE)
    x_train_shuffled = x_train[shuffle]
    y_train_shuffled = y_train[shuffle]
    # y_train_onehot = y_train_onehot[shuffle]
    for i in range(0, TRAIN_SIZE, BATCH_SIZE):
        logits = model(x_train_shuffled[i: i + BATCH_SIZE])
        loss = calc_loss(logits, y_train_shuffled[i: i + BATCH_SIZE])
        accuracy = calc_accuracy(logits, y_train_shuffled[i: i + BATCH_SIZE])
        
        ### APPEND LOSS AND ACCURACY
        if epoch % LOG_INTERVAL == 0:
            loss_train = torch.cat((loss_train, loss))
            accuracy_train = torch.cat((accuracy_train, accuracy))
    
        ### ACTOR UPDATE
        model_loss = loss.sum().view(1) # losses are summed instead of averaged for differing batch sizes
        optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

    ### LOG
    if epoch % LOG_INTERVAL == 0:
        
        ### SAVE CHECKPOINT
        epoch_path = os.path.join(save_dir, f'{epoch}.pt')
        checkpoint = {}
        checkpoint['model'] = model.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        torch.save(checkpoint, epoch_path)
    
        ### START LOG
        end_time = time.time() - start_time
        print(f'\nStep: {epoch}, Time: {time.strftime("%H:%M:%S", time.gmtime(end_time))}')
        row = [epoch]
        
        ### LOG TRAIN LOSS AND ACCURACY
        print(f'Mean_train_loss:\t{loss_train.mean().item()}')
        print(f'Train_accuracy:\t\t{(accuracy_train.sum() / TRAIN_SIZE).item()}')
        row.append(loss_train.mean().item())
        row.append((accuracy_train.sum() / TRAIN_SIZE).item())
        
        ### TEST TRAINED MODEL
        print('Testing...')
        with torch.no_grad():
            logits = model(x_test)
            loss_test = calc_loss(logits, y_test)
            accuracy_test = calc_accuracy(logits, y_test)
            
        ### LOG TEST LOSS AND ACCURACY
        print(f'Mean_test_loss:\t\t{loss_test.mean().item()}')
        print(f'Test_accuracy:\t\t{(accuracy_test.sum() / TEST_SIZE).item()}')
        row.append(loss_test.mean().item())
        row.append((accuracy_test.sum() / TEST_SIZE).item())
        
        ### LOG
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)