from __future__ import print_function
import argparse
import numpy  as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from data_loaders import Plain_Dataset, eval_data_dataloader
from deep_emotion import Deep_Emotion
from generate_data import Generate_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def Train(epochs,train_loader,val_loader,criterion,optimizer, scheduler,device):
    '''
    Training Loop
    '''
    print("===================================Start Training===================================")
    for e in range(epochs):
        print(optimizer.param_groups[0]["lr"])
        train_loss = 0
        validation_loss = 0
        train_correct = 0
        val_correct = 0
        # Train the model  #
        net.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs,1)
            train_correct += torch.sum(preds == labels.data)
        scheduler.step()
        #validate the model#
        net.eval()
        for data,labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            val_outputs = net(data)
            val_loss = criterion(val_outputs, labels)
            validation_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs,1)
            val_correct += torch.sum(val_preds == labels.data)

        train_loss = train_loss/len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        validation_loss =  validation_loss / len(validation_dataset)
        val_acc = val_correct.double() / len(validation_dataset)
        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Acuuarcy {:.3f}% \tValidation Acuuarcy {:.3f}%'
                                                           .format(e+1, train_loss,validation_loss,train_acc * 100, val_acc*100))

    torch.save(net.state_dict(),'deep_emotion-{}-{}-{}.pt'.format(epochs,batchsize,lr))
    print("===================================Training Finished===================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-s', '--setup', type=bool, help='setup the dataset for the first time')
    parser.add_argument('-d', '--data', type=str,required= True,
                               help='data folder that contains data files that downloaded from kaggle (train.csv and test.csv)')
    parser.add_argument('-hparams', '--hyperparams', type=bool,
                               help='True when changing the hyperparameters e.g (batch size, LR, num. of epochs)')
    parser.add_argument('-e', '--epochs', type= int, help= 'number of epochs')
    parser.add_argument('-lr', '--learning_rate', type= float, help= 'value of learning rate')
    parser.add_argument('-bs', '--batch_size', type= int, help= 'training/validation batch size')
    parser.add_argument('-t', '--train', type=bool, help='True when training')
    args = parser.parse_args()

    if args.setup :
        generate_dataset = Generate_data(args.data)
        generate_dataset.split_test()
        generate_dataset.save_images('train')
        generate_dataset.save_images('test')
        generate_dataset.save_images('val')

    if args.hyperparams:
        epochs = args.epochs
        lr = args.learning_rate
        batchsize = args.batch_size
    else :
        epochs = 300
        lr = 0.05
        batchsize = 64

    if args.train:
        net = Deep_Emotion()
        net.to(device)
        print("Model archticture: ", net)
        traincsv_file = args.data+'/'+'train.csv'
        validationcsv_file = args.data+'/'+'val.csv'
        train_img_dir = args.data+'/'+'train/'
        validation_img_dir = args.data+'/'+'val/'

        transformation_train = transforms.Compose([transforms.RandomRotation(degrees=45),
                                             transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.RandomVerticalFlip(p=0.05),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))])
        transformation_valid = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))])
        train_dataset= Plain_Dataset(csv_file=traincsv_file, img_dir = train_img_dir, datatype = 'train', transform = transformation_train)
        validation_dataset= Plain_Dataset(csv_file=validationcsv_file, img_dir = validation_img_dir, datatype = 'val', transform = transformation_valid)
        train_loader= DataLoader(train_dataset,batch_size=batchsize,shuffle = True,num_workers=0)
        val_loader=   DataLoader(validation_dataset,batch_size=batchsize,shuffle = True,num_workers=0)
        cweights = [1.02660468, 9.40661861, 1.00104606, 0.56843877, 0.84912748, 1.29337298, 0.82603942]
        class_weights = torch.FloatTensor(cweights).cuda()
        criterion= nn.CrossEntropyLoss(weight=class_weights)
        optimizer= optim.SGD(net.parameters(),lr= lr, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[81,121, 161,181, 241, 281], gamma=0.5)
        Train(epochs, train_loader, val_loader, criterion, optimizer, scheduler, device)