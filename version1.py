import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
from torchvision.io import read_image
import csv

class Net(nn.Module): # TODO
    def __init__(self, dim, dim_embed, bias=True):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(dim, dim_embed, 3, bias=bias)

    def forward(self, x):
        return self.conv(x) # shape [BATCH_SIZE, NUM_CLASS]
    
def calc_loss(logits, y):
    loss_func = nn.CrossEntropyLoss() # TODO: look into weights parameter
    loss_func(logits, y)
    return loss # shape [BATCH_SIZE]

def calc_accuracy(logits, y):
    y_hat = logits.max # TODO: convert logits into prediction using argmax
    accuracy = (y_hat == y).sum() / y.shape[0]
    return accuracy # shape [BATCH_SIZE]

### SET PARAMETERS

# HYPERPARAMETERS
SEED            = 0
BATCH_SIZE      = 16
LEARNING_RATE   = 0.01
MAX_GRAD_NORM   = 2
MAX_STEPS       = 1000
LOG_INTERVAL    = 20 # doesn't affect learning

# MODEL PARAMETERS
DIM_EMBED       = 16
KERNEL_SIZE     = 3

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
y_train_onehot = torch.scatter(torch.zeros(y_train.shape[0], NUM_CLASS), 1, y_train.unsqueeze(1), torch.ones(y_train.shape[0], 1))

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
y_test_onehot = torch.scatter(torch.zeros(y_test.shape[0], NUM_CLASS), 1, y_test.unsqueeze(1), torch.ones(y_test.shape[0], 1))

TRAIN_SIZE = x_train.shape[0]
TEST_SIZE  = x_test.shape[0]
NUM_CHANS  = x_train.shape[1]
NUM_ROWS   = x_train.shape[2]
NUM_COLS   = x_train.shape[3]

### INITIALIZE MODEL
model = Net(NUM_CHANS, DIM_EMBED, KERNEL_SIZE)
optimizer = Adam(model.parameerters(), lr=LEARNING_RATE)

### INITIALIZE SAVE DIRECTORY
characteristics = '{}'.format(KERNEL_SIZE)
time_id         = time.strftime('%Y-%m-%d %H-%M-%S')
save_dir        = os.path.join(characteristics, time_id)
os.makedirs(save_dir)

### LOAD CHECKPOINT
cp_path = os.path.join(characteristics, CP_TIME, '{}.pt'.format(CP_STEP))
if os.path.isfile(cp_path):
    print('Loading checkpoint...\n')
    checkpoint = torch.load(cp_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    print('Continuing with no checkpoint... \n')

### SET LOG WRITER
log_name = '{}.csv'.format(characteristics)
log_path = os.path.join(save_dir, log_name)
with open(log_path, 'w', newline='') as f:
    header = ['Step', 'Mean_train_loss', 'Train_accuracy', 'Mean_test_loss', 'Test_accuracy']
    writer = csv.writer(f)
    writer.writerow(header)

### TRAIN POLICY
print('Training...\n')
start_time = time.time()
for epoch in range(1, MAX_STEPS + 1):
    loss_train = []
    accuracy_train = []
    # TODO: look into DataLoader for shuffling and batching
    shuffle = torch.randperm(TEST_SIZE)
    x_train_shuffled = x_train[shuffle]
    y_train = y_train[shuffle]
    y_train_onehot = y_train_onehot[shuffle]
    for i in range(0, TRAIN_SIZE, BATCH_SIZE):
        logits = model(x_train[i: i + BATCH_SIZE])
        loss = calc_loss(logits, y_train_onehot[i: i + BATCH_SIZE])
        accuracy = calc_accuracy(logits, y_train[i: i + BATCH_SIZE])
        
        ### APPEND LOSS AND ACCURACY
        if epoch % LOG_INTERVAL == 0:
            loss_train.append(loss)
            accuracy_train.append(accuracy)
    
        ### ACTOR UPDATE
        model_loss = loss.mean().view(1)
        optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

    ### LOG
    if epoch % LOG_INTERVAL == 0:
        
        ### SAVE CHECKPOINT
        epoch_path = os.path.join(save_dir, "{}.pt".format(epoch))
        checkpoint = {}
        checkpoint['model'] = model.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        torch.save(checkpoint, epoch_path)
    
        ### START LOG
        end_time = time.time() - start_time
        print('Step: {}, Time: {}'.format(epoch,  time.strftime('%H:%M:%S', time.gmtime(end_time))))
        row = [epoch]
        
        ### LOG TRAIN LOSS AND ACCURACY
        # TODO: print mean loss and accuracy
        # TODO: add mean training loss and accuracy to log with row.extend()
        
        ### TEST TRAINED MODEL
        print('Testing...')
        with torch.no_grad():
            logits = model(x_test)
            loss_test = calc_loss(logits, y_test_onehot)
            accuracy_test = calc_accuracy(logits, y_test)
            
        ### LOG TEST LOSS AND ACCURACY
        # TODO: print mean loss and accuracy
        # TODO: add mean testing loss and accuracy to log with row.extend()
        
        ### LOG
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)