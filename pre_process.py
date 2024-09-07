import torch
from torchvision import transforms
from util.superpoint import SuperPoint
import os
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from PIL import Image

class MyImageDataset(Dataset):
    """
        A class representing a dataset of images and their poses
    """

    def __init__(self, dataset_path, data_transform=None):
        """
        :param dataset_path: (str) the path to the dataset
        :param labels_file: (str) a file with images and their path labels
        :param data_transform: (Transform object) a torchvision transform object
        :return: an instance of the class
        """
        super(MyImageDataset, self).__init__()
        self.img_paths = glob(dataset_path+"*.png")
        self.dataset_size = len(self.img_paths)
        self.transform = data_transform

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])        
        if self.transform:
            img = self.transform(img)
        
        return img, self.img_paths[idx]

def run_superpoint():
    device = 'cuda:0'
    checkpoint = 'checkpoint/superpoint_v1.pth.tar'
    nms_radius = 4
    keypoint_threshold = 0.005
    max_keypoints = 1024
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
    }
    superpoint = SuperPoint(config).to(device)
    superpoint.eval()
    
    root_path = '/mnt/f/CambridgeLandmarks/OldHospital/train/'
    image_path = root_path + 'rgb/'    

    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor()])
    
    out_path = root_path + 'superpoint/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)    

 # -- init data-loaders/samplers
    dataset = MyImageDataset(image_path, data_transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
            
    for (img, img_path) in dataloader:
        img = img.to(device)
        with torch.no_grad():
            pred0 = superpoint({'image': img})
        file_name = os.path.splitext(os.path.basename(img_path[0]))[0]
        out_name  = out_path + file_name + '.npy'
        np.save(out_name, pred0)          
            

if __name__ == '__main__':
    run_superpoint()
    