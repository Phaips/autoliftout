#!/usr/bin/env python3

# ref:
# https://towardsdatascience.com/train-a-lines-segmentation-model-using-pytorch-34d4adab8296
# https://discuss.pytorch.org/t/multiclass-segmentation-u-net-masks-format/70979/14
# https://github.com/qubvel/segmentation_models.pytorch

import argparse
import glob
import logging
from datetime import datetime
# from torchsummary import summary
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn import (Conv2d, Dropout2d, MaxPool2d, Module, ReLU, Sequential,
                      UpsamplingNearest2d)
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms, utils
from tqdm import tqdm


# change this to pre-processing- and cache
def load_images_and_masks_in_path(images_path, masks_path):
    images = []
    masks = []
    sorted_img_filenames = sorted(glob.glob(images_path + ".png"))
    sorted_mask_filenames = sorted(glob.glob(masks_path + ".png"))

    for img_fname, mask_fname in tqdm(
        list(zip(sorted_img_filenames, sorted_mask_filenames))
    ):

        image = np.asarray(Image.open(img_fname))
        mask = np.asarray(Image.open(mask_fname))

        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)


def decode_output(output):
    """decodes the output of segmentation model to RGB mask"""
    output = F.softmax(output, dim=1)
    mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    mask = decode_segmap(mask)
    return mask

def decode_segmap(image, nc=3):

    """ Decode segmentation class mask into an RGB image mask"""

    # 0=background, 1=lamella, 2= needle
    label_colors = np.array([(0, 0, 0), (255, 0, 0), (0, 255, 0)])

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

# transformations
transformation = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((1024 // 4, 1536 // 4)),
        # transforms.Lambda(
        #     lambda img: transforms.functional.adjust_contrast(img, contrast_factor=2)
        # ),
        transforms.ToTensor(),
    ]
)


class SegmentationDataset(Dataset):
    def __init__(self, images, masks, num_classes: int, transforms=None):
        self.images = images
        self.masks = masks
        self.num_classes = num_classes
        self.transforms = transforms

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transforms:
            image = self.transforms(image)

        mask = self.masks[idx]

        # - the problem was ToTensor was destroying the class index for the labels (rounding them to 0-1)
        # need to to transformation manually
        mask = Image.fromarray(mask).resize(
            (1536 // 4, 1024 // 4), resample=PIL.Image.NEAREST
        )
        mask = torch.tensor(np.asarray(mask)).unsqueeze(0)

        return image, mask

    def __len__(self):
        return len(self.images)


# helper functions
def show_img_and_mask(imgs, gts, mask, title="Image, Ground Truth and Mask"):

    n_imgs = len(imgs)

    fig, ax = plt.subplots(n_imgs, 3, figsize=(8, 6))
    fig.suptitle(title)

    for i in range(len(imgs)):

        img = imgs[i].permute(1, 2, 0)
        gt = decode_segmap(gts[i].permute(1, 2, 0).squeeze())  # convert to rgb mask

        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(gt)
        ax[1].set_title("Ground Truth")
        ax[2].imshow(mask)
        ax[2].set_title("Predicted Mask")

    # TODO: improve when batch size is larger
    plt.show()


def show_memory_usage():
    """Show total, reserved and allocated gpu memory"""
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved

    print("GPU memory", t, r, a, f, f"{f/r:.3}")


def show_values(ten):
    """ Show tensor statistics for debugging """
    unq = np.unique(ten.detach().cpu().numpy())
    print(ten.shape, ten.min(), ten.max(), ten.mean(), ten.std(), unq)


def save_model(model, epoch):

    # datetime object containing current date and time
    now = datetime.now()
    # format
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S") + f"_n{epoch+1}"
    model_save_file = f"models/{dt_string}_model.pt"
    torch.save(model.state_dict(), model_save_file)

    print(f"Model saved to {model_save_file}")


def train_model(model, device, data_loader, epochs, DEBUG=False):
    """ Helper function for training the model """
    # initialise loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    total_steps = len(train_data_loader)
    print(f"{epochs} epochs, {total_steps} total_steps per epoch")

    # accounting
    losses = []

    # training loop
    for epoch in tqdm(range(epochs)):
        i = 0
        print(f"------- Epoch {epoch+1} of {epochs}  --------")
        for images, masks in tqdm(train_data_loader):

            # set model to training mode
            model.train()

            # debugging training data
            # if DEBUG:
            # print(images.shape, masks.shape)
            # print("-"*50)

            # move img and mask to device, reshape mask
            images = images.to(device)
            masks = masks.type(torch.LongTensor)
            masks = masks.reshape(
                masks.shape[0], masks.shape[2], masks.shape[3]
            )  # remove channel dim
            masks = masks.to(device)

            # forward pass
            outputs = model(images).type(torch.FloatTensor).to(device)
            loss = criterion(outputs, masks)

            # backwards pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # evaluation
            model.eval()
            with torch.no_grad():

                outputs = model(images)
                output_mask = decode_output(outputs)

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}"
                )

                losses.append(loss.item())

                if DEBUG:
                    show_memory_usage()  # show gpu usage

                    # show images and masks
                    images = images.detach().cpu()
                    masks = masks.detach().cpu().unsqueeze(1)
                    show_img_and_mask(
                        images, masks, output_mask, title=f"Epoch {epoch+1} Evaluation",
                    )

            # print("-"*50)
            i += 1

        # save model checkpoint
        save_model(model, epoch)

        # show loss plot
        # plt.plot(losses)
        # plt.title("Loss Plot")
        # plt.show()

    return model


if __name__ == "__main__":

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        help="the directory containing the training data",
        dest="data",
        action="store",
        default="data",
    )
    parser.add_argument(
        "--debug",
        help="show debugging visualisation during training",
        dest="debug",
        action="store_true",
    )
    parser.add_argument(
        "--checkpoint",
        help="start model training from checkpoint",
        dest="checkpoint",
        action="store",
        default=None,
    )
    parser.add_argument(
        "--epochs",
        help="number of epochs to train",
        dest="epochs",
        action="store",
        type=int,
        default=2,
    )
    args = parser.parse_args()
    data_path = args.data
    DEBUG = args.debug
    model_checkpoint = args.checkpoint
    epochs = args.epochs

    ################################## LOAD DATASET ##################################
    print(
        "\n----------------------- Loading and Preparing Data -----------------------\n"
    )

    img_path = f"{data_path}/train/**/img"
    label_path = f"{data_path}/train/**/label"
    print(f"Loading data set from {img_path}")

    train_images, train_masks = load_images_and_masks_in_path(img_path, label_path)


    # hyperparams
    num_classes = 3
    batch_size = 1

    # load dataset
    seg_dataset = SegmentationDataset(
        train_images, train_masks, num_classes, transforms=transformation
    )

    # TODO: validation dataset

    val_size = 0.2
    dataset_size = len(seg_dataset)
    dataset_idx = list(range(dataset_size))
    split_idx = int(np.floor(val_size * dataset_size))
    train_idx = dataset_idx[split_idx:]
    val_idx = dataset_idx[:split_idx]

    train_sampler =  SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
# https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a
    train_data_loader = DataLoader(seg_dataset, batch_size=batch_size, shuffle=True, sampler=train_sampler)
    print(f"Train dataset has {len(train_data_loader)} batches of size {batch_size}")

    # from smp
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,  # grayscale images
        classes=3,  # background, needle, lamella
    )

    # load model checkpoint
    if model_checkpoint:
        model.load_state_dict(torch.load(model_checkpoint))
        print(f"Checkpoint file {model_checkpoint} loaded.")

    ################################## SANITY CHECK ##################################
    for i in range(2):
        # testing dataloader
        imgs, masks = next(iter(train_data_loader))

        # sanity check - model, imgs, masks
        output = model(imgs)
        pred = decode_output(output)

        print("imgs, masks, output")
        print(imgs.shape, masks.shape, output.shape)

        show_img_and_mask(
            imgs=imgs,
            gts=masks,
            mask=np.zeros_like(masks[0].squeeze(0)),
            title="Sanity Checks (No Prediction)",
        )
        show_img_and_mask(imgs, masks, pred, title="Checkpointed Model Predictions")

    ################################## TRAINING ##################################
    print("----------------------- Begin Training -----------------------\n")

    # Use gpu for training if available else use cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # train model
    model = train_model(model, device, train_data_loader, epochs, DEBUG=DEBUG)

    ################################## SAVE MODEL ##################################

    # TODO: validation and test dataset
