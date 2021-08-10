#!/usr/bin/env python3

import glob
import json
import re
from random import shuffle

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torchvision import transforms, utils

class DetectionModel:
    """Detection Model Class 
    
    Defines the functionality for the liftout detection model
    
    """

    def __init__(self, weights_file) -> None:

        self.weights_file = weights_file

        # transformations
        self.transformation = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((1024 // 4, 1536 // 4)),
                transforms.ToTensor(),
            ]
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # TODO: support GPU

        self.load_model()

    def preprocess_image(self, img):
        """ preprocess an image for model inference """
        img_t = self.transformation(img).unsqueeze(0).to(self.device)
        return img_t

    def load_model(self):
        """ helper function for loading model"""

        # load model
        self.model = smp.Unet(encoder_name="resnet18", in_channels=1, classes=3,)
        # load model weights
        self.model.load_state_dict(torch.load(self.weights_file, map_location="cpu"))
        self.model.to(self.device) #TODO: GPU support
        self.model.eval()

    def model_inference(self, img):

        """
            Helper function to run the image through model,
            and return image and predicted mask
        """
        # pre-process image (+ batch dim)
        img_t = self.preprocess_image(img=img)
        
        # model inference
        output = self.model(img_t)

        # calculate mask
        rgb_mask = self.decode_output(output)

        return rgb_mask

    def decode_output(self, output):
        """decodes the output of segmentation model to RGB mask"""
        output = F.softmax(output, dim=1)
        mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
        mask = self.decode_segmap(mask)
        return mask

    def decode_segmap(self, image, nc=3):

        """ Decode segmentation class mask into an RGB image mask"""

        # 0=background, 1=lamella, 2= needle
        label_colors = np.array([(0, 0, 0),
                                 (255, 0, 0),
                                 (0, 255, 0)])

        # pre-allocate r, g, b channels as zero
        r = np.zeros_like(image, dtype=np.uint8)
        g = np.zeros_like(image, dtype=np.uint8)
        b = np.zeros_like(image, dtype=np.uint8)

        # apply the class label colours to each pixel
        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        # stack rgb channels to form an image
        rgb_mask = np.stack([r, g, b], axis=2)
        return rgb_mask


