import cv2
import os

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

class DatasetGen(Dataset):

    def __init__(self, dataset_path, is_train=True, transforms=transforms.ToTensor()): # DatasetGen object_name
        self.path = dataset_path
        self.transform = transforms
        assert os.path.isdir(self.path), "Not a valid dataset path"

        self.train = os.path.join(self.path,'train')
        self.val = os.path.join(self.path,'val')

        self.train_imgs = os.path.join(self.train,'images')
        self.train_masks = os.path.join(self.train,'masks')

        self.val_imgs = os.path.join(self.val,'images')
        self.val_masks = os.path.join(self.val,'masks')

        self.train_imgs = [os.path.join(self.train_imgs,x) for x in os.listdir(self.train_imgs)]
        self.train_masks = [os.path.join(self.train_masks,x) for x in os.listdir(self.train_masks)]

        self.val_imgs = [os.path.join(self.val_imgs,x) for x in os.listdir(self.val_imgs)]
        self.val_masks = [os.path.join(self.val_masks,x) for x in os.listdir(self.val_masks)] 

        if is_train == True:
            self.imgs = self.train_imgs
            self.masks = self.train_masks
        else:
            self.imgs = self.val_imgs
            self.masks = self.val_masks

    def __len__(self): # len(datagen_name)
        return len(self.imgs)

    def __getitem__(self,idx): # iter(dataloader_name).next()
        img = cv2.imread(self.imgs[idx],cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks[idx],cv2.IMREAD_GRAYSCALE)

        img = self.transform(img)
        mask = self.transform(mask)

        return img,mask

data_train = DatasetGen(dataset_path='data_dagm')

dataloader_train = DataLoader(batch_size=3,shuffle=True,num_workers=0,dataset=data_train)