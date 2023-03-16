import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
from torchvision.io import read_image
import csv

class Net(nn.Module):
    
    def __init__(self, dim, dim_embed, bias=True):
        super(Net, self).__init__()
        self.conv = nn.Conv3d(dim, dim_embed, 3, bias=bias)

    def forward(self, x):
        return self.conv(x)
    
def calc_loss(logits, target):
    loss_func = nn.CrossEntropyLoss() # TODO: look into weights parameter
    loss_func(logits, target)
    return loss

def calc_accuracy(logits, y): # TODO
    y_hat = logits.max # TODO: convert logits into one-hot prediction using argmax and scatter
    accuracy = (y_hat == y).sum() / y.shape[0]
    return accuracy

### READ FILES
# TODO
path = 'train'
print(os.listdir(path))
for item in os.listdir(path)[1:]:
    print(item)
    item_path = os.path.join(path, item)
    for image in os.listdir(item_path):
        image_path = os.path.join(item_path, image)
        print(image_path)
        x = torchvision.io.read_image(image_path)
        print(x)
        break

TRAIN_SIZE = ___
TEST_SIZE  = ___
#NUM_LAYS   = ___
NUM_ROWS   = ___
NUM_COLS   = ___
NUM_CHANS  = ___
NUM_CLASS  = 4

### SET PARAMETERS

# HYPERPARAMETERS
SEED            = 0
BATCH_SIZE      = 16
LEARNING_RATE   = 0.01
MAX_GRAD_NORM   = 2
MAX_STEPS       = 1000
LOG_INTERVAL    = 20

# MODEL PARAMETERS
DIM_EMBED       = 16
KERNEL_SIZE     = 3

# LOAD CHECKPOINT INFORMATION
CP_TIME         = ''
CP_STEP         = 0

### INITIALIZE MODEL
model = Net(NUM_CHANS, DIM_EMBED, KERNEL_SIZE)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

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
    header = ['Step', 'Mean_train_loss', 'train_accuracy', 'Mean_test_loss', 'test_accuracy']
    writer = csv.writer(f)
    writer.writerow(header)

### TRAIN POLICY
print('Training...\n')
start_time = time.time()
for epoch in range(1, MAX_STEPS + 1):
    loss_train = []
    accuracy_train = []
    #TODO: break training into batches (x and y) -> DataLoader?
    for _ in range(TRAIN_SIZE // BATCH_SIZE):
        logits = model(x) # TODO -> result should be shape [BATCH_SIZE, NUM_CLASS] of logits
        target = ___ # TODO: convert y into one-hot target vectors using scatter
        loss = calc_loss(logits, target)
        accuracy = calc_accuracy(logits, y)
        
        ### APPEND LOSS AND ACCURACY
        if epoch % LOG_INTERVAL == 0:
            loss_train.append(loss.mean())
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
        
        ### LOG TRAIN LOSS AND ACCURACY # TODO: from loss_train and accuracy_train
        # TODO: print mean loss and accuracy
        # TODO: add mean training loss and accuracy to log with row.extend()
        
        ### TEST TRAINED MODEL
        print('Testing...')
        with torch.no_grad():
            # TODO: test step (same as training)
            logits = model(x) # TODO
            target = ___ # TODO: convert y into one-hot binary target vectors
            loss_test = calc_loss(logits, target)
            accuracy_test = calc_accuracy(logits, y)
            
        ### LOG TEST LOSS AND ACCURACY # TODO: from loss_test and accuracy_test
        # TODO: print mean loss and accuracy
        # TODO: add mean testing loss and accuracy to log with row.extend()
        
        ### LOG
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)