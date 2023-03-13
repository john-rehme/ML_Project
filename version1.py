import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import csv

class Net(nn.Module):
    
    def __init__(self, dim, dim_embed, bias=True):
        super(Net, self).__init__()
        self.encode = nn.Conv2d(dim, dim_embed, 3, bias=bias)

    def forward(self, x): # 176x208
        return self.encode(x)
    
def sample(x, y):
    # TODO
    return loss

### SET PARAMETERS

# MODEL PARAMETERS
SEED            = 0
BATCH_SIZE      = 128
DIM_EMBED       = 128
LEARNING_RATE   = 0.0005
MAX_GRAD_NORM   = 2
MAX_STEPS       = 5000
LOG_INTERVAL    = 20
KERNEL          = 3

# LOAD CHECKPOINT INFORMATION
CP_TIME         = ''
CP_STEP         = 0

### INITIALIZE ENVIROMENT AND POLICIES
net = Net(___, DIM_EMBED, KERNEL) # TODO  # 176x208
optimizer = Adam(net.parameters(), lr=LEARNING_RATE)

### INITIALIZE SAVE DIRECTORY
characteristics = '{}'.format(BATCH_SIZE)
time_id         = time.strftime('%Y-%m-%d %H-%M-%S')
save_dir        = os.path.join(characteristics, time_id)
os.makedirs(save_dir)

### LOAD CHECKPOINT
trained_steps       = 0
cp_path             = os.path.join(characteristics, CP_TIME, '{}.pt'.format(CP_STEP))
if os.path.isfile(cp_path):
    print('Loading checkpoint...\n')
    trained_steps = CP_STEP
    checkpoint = torch.load(cp_path)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
total_steps = MAX_STEPS - trained_steps

### SET LOG WRITER
log_name = '{}.csv'.format(characteristics)
log_path = os.path.join(save_dir, log_name)
with open(log_path, 'w', newline='') as f:
    header = ['Step', 'Mean_train_loss', 'Mean_test_loss']
    writer = csv.writer(f)
    writer.writerow(header)

### TRAIN POLICY
print('Training...\n')
start_time = time.time()
for epoch in range(total_steps):
    loss_train = []
    for _ in range(EPOCH_SIZE // BATCH_SIZE):
        loss = net(___) # TODO
        
        ### APPEND DISTANCE AND PENALTY
        if (epoch + 1) % LOG_INTERVAL == 0:
            loss_train.append(loss.mean())
    
        ### ACTOR UPDATE # TODO
        actor_loss = loss.mean().view(1)
        optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
        optimizer.step()

    ### LOG
    if (epoch + 1) % LOG_INTERVAL == 0:
        
        ### SAVE CHECKPOINT
        epoch_path = os.path.join(save_dir, "{}.pt".format(epoch + trained_steps))
        checkpoint = {}
        checkpoint['net'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        torch.save(checkpoint, epoch_path)
    
        ### START LOG # TODO: fix row headers
        end_time = time.time() - start_time
        print('Step: {}, Time: {}'.format(epoch + trained_steps,  time.strftime('%H:%M:%S', time.gmtime(end_time))))
        row = [epoch + trained_steps]
        
        ### LOG TRAIN DISTANCE AND PENALTY # TODO
        print('Mean_train_loss:\t' + str(torch.stack(loss, 0).mean().item()))
        # TODO: add training loss to log with row.extend()
        
        ### TEST TRAINED POLICY # TODO
        print('Testing...')
        loss_test = []
        # for _ in range(TEST_SIZE // BATCH_SIZE):
        #     with torch.no_grad():
            # TODO: test step
            
        ### LOG TEST DISTANCE AND PENALTY # TODO
        # TODO: print loss
        # TODO: add testing loss to log with row.extend()
        
        ### LOG
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)