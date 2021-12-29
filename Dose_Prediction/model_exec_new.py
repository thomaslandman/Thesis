# import sys
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import time
import os
from data_augmentation import trans_list
# sys.path.append('C:\\Users\Thomas\AppData\Local\Programs\Python\Python39\Lib\site-packages')
import load_data_new
import data_augmentation as aug
from U_Net import UNet
import csv
import matplotlib.pyplot as plt
import sys
def weights_init(m):
    """Initialization script for the neural network

    :params m: model weights
    """
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
    if isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

def train_augmentation(aug_list, scan_list):
    for scanID in range(len(scan_list)):
        running_loss = []
        for i in range(len(aug_list)):
            data_path = os.path.join(dir_path, patient_list[scanID], visit_date_list[scanID])
            struct, dose = load_data_new.load_data(data_path, dim=[128, 256, 64])
            struct[0, :, :, :, :] = aug.structure_transform(struct[0, :, :, :, :], aug_list[i])
            dose[0, :, :, :, :] = aug.dose_transform(dose[0, 0, :, :, :], aug_list[i])
            dose_tens = torch.from_numpy(dose).to(device)
            struct_tens = torch.from_numpy(struct).to(device)
            optimizer.zero_grad()
            dose_pred = model(struct_tens)
            loss = loss_func(dose_pred, dose_tens)
            loss.backward()
            optimizer.step()
            running_loss = np.append(running_loss, loss.item())

        print("The average training loss after the augmentation set ", '%d'%(int(scanID+1)), ": ", '%.3f'%(np.mean(running_loss)))
        
# Initialize loss values
training_loss = []
std_train = []
validation_loss = []
std_validation = []
time_tot = 0.0
loss_path = "/exports/lkeb-hpc/tlandman/Dose_Prediction/log/training_loss.csv"

# Initialize the network
model = UNet()                                                  #Modify for different input UNet
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-03)
## Now load the parameters if using a pretrained model instead of applying the weight initialization ##
try:
    checkpoint = torch.load('/exports/lkeb-hpc/tlandman/Dose_Prediction/model/model.pth.tar', map_location="cuda:0")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print('Model loaded from Epoch ' + str(last_epoch))
except:
    model.apply(weights_init)
    last_epoch = 0
    print('No model found, start training from beginning')
    
    with open(loss_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Epoch', 'Training Loss', 'Std Train', 'Evaluation Loss', 'Std Eval'])
loss_func = nn.MSELoss()

# Give inputs
device = torch.device("cuda")
patient_list = []
visit_date_list = []
with open("/exports/lkeb-hpc/tlandman/Patient_Data/visits.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        patient_list.append(str(row[0]))
        visit_date_list.append(str(row[1]))

dir_path = "/exports/lkeb-hpc/tlandman/Patient_Dose/"
num_epoch = 30
do_augmen = False
start = time.time()
print('Training has started')

if last_epoch == 0 and do_augmen == True:
    scan_list = [5,25,45,65,85]
    aug_list = aug.trans_list()
    train_augmentation(aug_list, scan_list)

for epoch in range(last_epoch+1, num_epoch+1):
    print('Epoch ' + str(epoch))

    model.train()
    running_loss = []
    for scanID in range(144):
        data_path = os.path.join(dir_path, patient_list[scanID], visit_date_list[scanID])
        struct, dose = load_data_new.load_data(data_path, dim=[128, 256, 64])
        dose_tens = torch.from_numpy(dose).to(device)
        struct_tens = torch.from_numpy(struct).to(device)
        optimizer.zero_grad()
        dose_pred = model(struct_tens)
        loss = loss_func(dose_pred, dose_tens)
        loss.backward()
        optimizer.step()
        running_loss = np.append(running_loss, loss.item())

    training_loss = np.append(training_loss, np.mean(running_loss))
    std_train = np.append(std_train, np.std(running_loss))
    print("The average training loss in epoch ", '%d'%(int(epoch)), ": ", '%.3f'%(training_loss[-1]))

    model.eval()
    running_loss = []
    with torch.no_grad():
        for scanID in range(144, len(visit_date_list)):
            data_path = os.path.join(dir_path, patient_list[scanID], visit_date_list[scanID])
            struct, dose = load_data_new.load_data(data_path, dim=[128, 256, 64])
            dose_tens = torch.from_numpy(dose).to(device)
            struct_tens = torch.from_numpy(struct).to(device)
            dose_pred = model(struct_tens)
            loss = loss_func(dose_pred, dose_tens)
            running_loss = np.append(running_loss, loss.item())

    validation_loss = np.append(validation_loss, np.mean(running_loss))
    std_validation = np.append(std_validation, np.std(running_loss))
    print("The average validation loss in epoch ", '%d' % (int(epoch)), ": ", '%.3f' % (validation_loss[-1]))
    print("Time since start of training is: ", '%d'%(time.time()-start), "seconds")


print("Training is completed in ", '%d'%(time.time()-start), " seconds")

model_path = "/exports/lkeb-hpc/tlandman/Dose_Prediction/model/model.pth.tar"
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_path)

with open(loss_path, 'a', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    for i in range(len(training_loss)):
        writer.writerow([i+1, training_loss[i], std_train[i], validation_loss[i], std_validation[i]])

def train_augmentation(aug_list, scan_list):
    for scanID in range(len(scan_list)):
        running_loss = []
        for i in range(len(aug_list)):
            data_path = os.path.join(dir_path, patient_list[scanID], visit_date_list[scanID])
            struct, dose = load_data_new.load_data(data_path, dim=[96, 192, 48])
            struct[0,:,:,:,:]= aug.structure_transform(struct[0,:,:,:,:], aug_list[i])
            dose[0,:,:,:,:] = aug.dose_transform(dose[0,0,:,:,:], aug_list[i])
            dose_tens = torch.from_numpy(dose).to(device)
            struct_tens = torch.from_numpy(struct).to(device)
            optimizer.zero_grad()
            dose_pred = model(struct_tens)
            loss = loss_func(dose_pred, dose_tens)
            loss.backward()
            optimizer.step()
            running_loss = np.append(running_loss, loss.item())

        print("The average training loss after the augmentation ", '%d'%(int(scanID)), ": ", '%.3f'%(np.mean(running_loss)))
# print(np.shape(struct))
#         print(np.shape(dose))
#         print(np.max(struct))
#         print(np.max(dose))
#         print(type(struct[0,0,0,0,0]))
#         print(type(dose[0,0,0,0,0]))
#         dose_tens = torch.from_numpy(dose).to(device)
#         struct_tens = torch.from_numpy(struct).to(device)
#         print(struct_tens[0, 0, 0, 0, 0].dtype)
#         print(dose_tens[0, 0, 0, 0, 0].dtype)
#         dose_new = np.array(dose_tens.cpu())
#         struct_new = np.array(struct_tens.cpu())
#         print(np.shape(struct_new))
#         print(np.shape(dose_new))
#         print(type(struct_new[0, 0, 0, 0, 0]))
#         print(type(dose_new[0, 0, 0, 0,0 ]))
#         print(np.max(struct_new))
#         print(np.max(dose_new))
#
#         sys.exit()
#         # #print("Loaded scan (" + str(scanID+1) + "/" + str(len(visit_date_list)) + ") succesfully")
#         # aug_list = [[0,0,0,0]]
#         # for i in range(len(aug_list)):
#         #     struct_aug = aug.structure_transform(struct, aug_list[i]).to(device)
#         #     dose_aug = aug.dose_transform(dose, aug_list[i]).to(device)
#         #     optimizer.zero_grad()
#         #     output = model(struct_aug)
#         #     loss = loss_func(output, dose_aug)
#         #     loss.backward()
#         #     optimizer.step()
#         #     running_loss = np.append(running_loss, loss.item())
#         # if epoch==1 and scanID==4:
#         #     aug_list = [[0, 0, 0, 0]]