from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import PIL
from skimage import io, transform
import json
import glob
import re
import tensorflow as tf
import matplotlib.font_manager as fm

#from config import Config
if __name__ == '__main__':
    plt.ion()  
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    class FoodDataset(torch.utils.data.Dataset):
       """ Dataset object used to access the pre-processed VoxCelebDataset """
       def __init__(self, root, extension='.jpg',shuffle=False, transform=True, shuffle_frames=False, subset_size=None,phase=True):
           """
           Instantiates the Dataset.
           :param root: Path to the folder where the pre-processed dataset is stored.
           :param extension: File extension of the pre-processed video files.
           :param shuffle: If True, the video files will be shuffled.
           :param transform: Transformations to be done to all frames of the video files.
           :param shuffle_frames: If True, each time a video is accessed, its frames will be shuffled.
           """
           if (phase==True):
               self.root=os.path.join(root,'train')
           else:
               self.root=os.path.join(root,'test')
           print(self.root)
           self.transform = transform
           self.files = glob.glob(os.path.join(self.root, "*"))
           if shuffle:
               random.shuffle(self.files)
           self.length = len(self.files)
           with open(os.path.join(root, "all_labels.json"), "r", encoding="utf8") as f:
               self.all_labels = json.load(f)
           with open(os.path.join(root, "labels_count2.json"), "r", encoding="utf8") as f:
               self.labels_count = json.load(f)
           self.num_to_label = []
           for a_key in self.labels_count.keys():
               if self.labels_count[a_key] >= 3:
                    self.num_to_label.append(a_key)
           self.label_to_num = {}
           for i in range(len(self.num_to_label)):
               self.label_to_num[self.num_to_label[i]] = i
           
           self.label_len = len(self.num_to_label)
       
       def read_ingredients(self, file_name):
           with open(file_name, "r", encoding="utf8") as f:
               new_labels = f.read().split('\n')
           real_labels = []
           for label in new_labels:
               label = re.sub('\(.*?\)', '', label)
               label = label.replace(' ', '')
               if label != "":
                   real_labels.append(label)

           np_labels = np.zeros(self.label_len)
           for real_label in real_labels:
               processed_label = self.all_labels[real_label]
               #print(real_label)
               #print(processed_label)
               try:
                    a_idx = self.label_to_num[processed_label]
                    np_labels[a_idx] = 1
               except:
                    pass
               
               
           return np_labels
       def __len__(self):
           return self.length
       def __getitem__(self, idx):
           item_path = self.files[idx]
           uuid = item_path
           img_path = os.path.join(item_path, "thumbnail.jpg")
           ingredients_path = os.path.join(item_path, "ingredients.txt")
           x = PIL.Image.open(img_path)
          # print(uuid)
           if self.transform:
               torch_x = self.transform(x)
           np_y = self.read_ingredients(ingredients_path)
           torch_y = torch.from_numpy(np_y).float()
           return uuid, torch_x, torch_y
    data_dir = 'data'
    image_datasets = FoodDataset(root=data_dir,
        extension='.jpg',
        shuffle_frames=True,
        subset_size=140000,transform=data_transforms['train']
              ,phase=True)

    image_datasets_val = FoodDataset(root=data_dir,
        extension='.jpg',
        shuffle_frames=True,
        subset_size=140000,transform=data_transforms['val']
              ,phase=False)
    print(image_datasets)
    print(image_datasets_val)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=4,
                                                shuffle=True)

    dataloaders_val = torch.utils.data.DataLoader(image_datasets_val, batch_size=4,
                                                shuffle=True)
                
    dataset_sizes = { len(image_datasets) }
    class_names = ['pizza','pasta']

    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(1)  # pause a bit so that plots are updated
    #iter(torch.utils.data.DataLoader(dataloaders)).next()

    for i_batch, (real_idx, images,labely) in enumerate(dataloaders):
        #print(i_batch, real_idx, images,labely)
        print(images.size())
        print(labely.size())
        print(real_idx[0])
        break
        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

        
        for (batch_num,inputs) in enumerate(dataloaders):
            print('batch_num'+str(batch_num))
            print('input'+str(inputs))
        # Make a grid from batch
    labely, inputs, classes = next(iter(dataloaders))

# Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
 #   imshow(out)


    # Get a batch of training data
#    inputs, classes = enumerate(dataloaders)
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

      #  m = nn.Sigmoid()

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                if phase== 'val':
                    for i, (uuid,inputs, labels) in enumerate(dataloaders):
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(torch.sigmoid(outputs), labels)
                            if (i % 100 == 0):
                                print('loss', loss)
                                torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss,
                                }, 'models_sig/'+str(epoch)+'.pt')
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                       # running_loss += loss.item() * inputs.size(0)
                       # running_corrects += torch.sum(preds == labels.data)
                elif phase== 'train':
                    for i, (uuid,inputs, labels) in enumerate(dataloaders):
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)


                            _, preds = torch.max(outputs, 1)
                           # print(outputs.dtype)
                           # print(labels.dtype)
                            loss = criterion(torch.sigmoid(outputs), labels)
                            
                            if (i % 100 == 0):
                                print('loss', loss)
                                torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss,
                                }, 'models_sig/'+str(epoch)+'.pt')
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        #running_loss += loss.item() * inputs.size(0)
                        #running_corrects += torch.sum(preds == labels.data)


                if phase == 'train':
                    scheduler.step()

                #epoch_loss = running_loss / dataset_sizes[phase]
                #epoch_acc = running_corrects.double() / dataset_sizes[phase]

                #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                 #   phase, epoch_loss, epoch_acc))

                # deep copy the model
                #if phase == 'val' and epoch_acc > best_acc:
                #    best_acc = epoch_acc
                #    best_model_wts = copy.deepcopy(model.state_dict())

            #print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        #print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
    def visualize_model(model, datasetname, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        print(plt.rcParams['font.family']) #現在使用しているフォントを表示
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
        print(plt.rcParams['font.family']) #現在使用しているフォントを表示
        fig = plt.figure()
        with torch.no_grad():
            for i, (uuid,inputs, labels) in enumerate(dataloaders_val):
                print(uuid)
                print(i)
                fm.findSystemFonts()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                print(outputs)

                _, preds = torch.max(outputs, 1)
                outputs= outputs.to(device)
              #  print(outputs)
                #print(labels.tolist())


                for j in range(inputs.size()[0]):
                    ing_list=[]
                    for i in range(len(outputs.tolist()[j])):
                        if(outputs[j][i])>-1.5:
                            ing_list.append(datasetname.num_to_label[i])
                    print(ing_list)
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    i=len(ing_list)//2
                    if(len(ing_list)<5):
                        ax.set_title(u'predicted: {}'.format(ing_list), fontname="sans-serif")
                #    elif(len(ing_list)<16):
                 #       ax.set_title(u'predicted: {} \n{}'.format(ing_list[0:7],ing_list[7:15]), fontname="sans-serif")
                    else:
                        ax.set_title(u'predicted: {} \n{}'.format(ing_list[0:i],ing_list[i:]), fontname="sans-serif")

                    imshow(inputs.cpu().data[j])
                    if images_so_far == num_images:
                        time.sleep(1000)
                        model.train(mode=was_training)
                
            model.train(mode=was_training)
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
#    model_ft.fc = nn.Linear(num_ftrs,329)
    model_ft.fc = nn.Linear(num_ftrs,329)
   
    # Observe that all parameters are being optimized
    """ 
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    

    model_ft = model_ft.to(device) """
    model_ft = model_ft.to(device)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    try:
        checkpoint = torch.load('models/24.pt' ,map_location=torch.device('cpu'))
        model_ft.load_state_dict(checkpoint['model_state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print('epoch :{} loss: {}'.format(epoch,loss))
    except:
        print('no files')

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=25)
    visualize_model(model_ft,image_datasets_val)