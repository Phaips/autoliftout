#!/usr/bin/env python3


import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torchvision import transforms
import liftout.detection.utils as det_utils
from pathlib import Path

class DetectionModel:
    """Detection Model Class 
    
    Defines the functionality for the liftout detection model
    
    """

    def __init__(self, weights_file) -> None:

        # TODO: make the path os agnostic
        self.weights_file = weights_file

        # transformations
        self.transformation = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((1024 // 4, 1536 // 4)),
                transforms.ToTensor(),
            ]
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = load_model(weights_file, device=self.device)

    def preprocess_image(self, img):
        """ preprocess an image for model inference """
        img_t = self.transformation(img).unsqueeze(0).to(self.device)
        return img_t


    def inference(self, img):

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
        mask = det_utils.decode_segmap(mask)
        return mask


def load_model(weights_file: Path, device):
    """ helper function for loading model"""

    # load model
    model = smp.Unet(encoder_name="resnet18", in_channels=1, classes=3,)

    # load model weights
    model.load_state_dict(torch.load(weights_file, map_location="cpu"))
    model.to(device)
    model.eval()

    return model