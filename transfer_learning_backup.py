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
from PIL import Image
from skimage import io, transform
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

        def __init__(self, root, extension='.jpg',shuffle=False, transform=None, shuffle_frames=False, subset_size=None,phase=True):
            """
            Instantiates the Dataset.
            :param root: Path to the folder where the pre-processed dataset is stored.
            :param extension: File extension of the pre-processed video files.
            :param shuffle: If True, the video files will be shuffled.
            :param transform: Transformations to be done to all frames of the video files.
            :param shuffle_frames: If True, each time a video is accessed, its frames will be shuffled.
            """
            if (phase==True):
                root=os.path.join(root,'train')
                self.phase='train'
            else:
                root=os.path.join(root,'test')
                self.phase='test'
            print(root)
            self.root = root
            
            self.transform = transform

            self.files = [
                os.path.join(path, filename)
                for path, dirs, files in os.walk(root)
                for filename in files
                if filename.endswith(extension)
            ]
            self.files.sort()
            if subset_size is not None:
                self.files = self.files[:subset_size]
            self.length = len(self.files)
            self.indexes = [idx for idx in range(self.length)]
            self.ingredients=[
                os.path.join(path,filename)
                for path, dirs, files in os.walk(root)
                for filename in files
                if filename.endswith('ingredients.txt')
            ]
            
   
            if shuffle:
                random.shuffle(self.indexes)

            self.shuffle_frames = shuffle_frames

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            real_idx = self.indexes[idx]
            path = self.files[real_idx]
            with open(self.ingredients[real_idx]) as fp:
                line = fp.readline()
                ingredientslist=[]
                while line:
                    ingredientslist.append(line.strip())
                    line = fp.readline()
#            print(self.ingredients[real_idx])
            print(ingredientslist)
            
            x = Image.open(path)
         #   x=torchvision.datasets.ImageNet(path,download=True)
          #  x=io.imread(path)
            print(self.phase)
            x = self.transform[self.phase](x)
            matplotlib_imshow(img_grid, one_channel=True)
            x=torch.stack(x)
            return real_idx, x

    data_dir = 'data'
    image_datasets = FoodDataset(root=data_dir,
        extension='.jpg',
        shuffle_frames=True,
        subset_size=140000,transform=data_transforms['train']
              ,phase=True)
    print(image_datasets)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=4,
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

    for i_batch, (real_idx, images) in enumerate(dataloaders):
        print(i_batch, real_idx.size, images.size)

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

        

    # Get a batch of training data
#    inputs, classes = enumerate(dataloaders)
"""   for (batch_num,inputs) in enumerate(dataloaders):
        print('batch_num'+str(batch_num))
        print('input'+str(inputs))
    # Make a grid from batch
    out = torchvision.utils.make_grid(dataloaders)
    print(0)
    print(class_names)
    imshow(out, title=[class_names[x] for x in classes])
    """