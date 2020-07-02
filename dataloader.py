import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import numpy as np
from PIL import Image

"""
my game board position from image
x : 265 ~ 981
y : 1340 ~ 1771
"""

LU = {'x': 265, 'y': 1340}
RD = {'x': 981, 'y': 1771}


def input_transform():
    # mean and std of ImageNet
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class RDdataset(data.Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.images = os.listdir(os.path.join(root_dir, 'images'))
        self.labels = os.listdir(os.path.join(root_dir, 'labels'))
        assert len(self.images) == len(self.labels)
        self.input_transform = input_transform()

    def get_grid(self, image_tensor):
        my_board = image_tensor[LU['y']:RD['y'], LU['x']:RD['x'], :]
        my_board_height, my_board_width, _ = my_board.shape
        square_height = int(my_board_height/3)
        square_width = int(my_board_width/5)
        my_grid = []
        for row in range(3):
            for col in range(5):
                my_grid.append(my_board[square_height*row:square_height*row +
                                        square_height, square_width*col:square_width*col+square_width, :])

        # myGird shape
        # [15, H, W, 3]
        return torch.stack(my_grid)

    def read_labeltxt(self, label_text):
        with open(label_text, mode='r') as label:
            label_grid = [[int(x) for x in line.split()] for line in label]
            return torch.from_numpy(np.array(label_grid))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = Image.open(os.path.join(
            self.root_dir, 'images', self.images[index]))
        image_tensor = self.input_transform(image)
        my_grid = self.get_grid(image_tensor)

        label = self.read_labeltxt(os.path.join(
            self.root_dir, 'labels', self.labels[index]))

        return my_grid, label
