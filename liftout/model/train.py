#!/usr/bin/env python3

import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from liftout.model.dataset import *
from liftout.model.utils import *
from tqdm import tqdm

def save_model(model, epoch):
    """Helper function for saving the model based on current time and epoch"""
    
    # datetime object containing current date and time
    now = datetime.now()
    # format
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S") + f"_n{epoch+1}"
    model_save_file = f"models/{dt_string}_model.pt"
    torch.save(model.state_dict(), model_save_file)

    print(f"Model saved to {model_save_file}")

def train_model(model, device, train_data_loader, val_data_loader, epochs, DEBUG=False):
    """ Helper function for training the model """
    # initialise loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    total_steps = len(train_data_loader)
    print(f"{epochs} epochs, {total_steps} total_steps per epoch")

    # accounting
    train_losses = []
    val_losses = []

    # training loop
    for epoch in tqdm(range(epochs)):
        print(f"------- Epoch {epoch+1} of {epochs}  --------")
        
        train_loss = 0
        val_loss = 0
        
        data_loader = tqdm(train_data_loader)

        for i, (images, masks) in enumerate(data_loader):

            # set model to training mode
            model.train()

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
            train_loss += loss.item()
            data_loader.set_description(f"Train Loss: {loss.item():.04f}")

            if i % 100 == 0:
                if DEBUG:
                    model.eval()
                    with torch.no_grad():

                        outputs = model(images)
                        output_mask = decode_output(outputs)
                    show_memory_usage()  # show gpu usage

                    # show images and masks
                    images = images.detach().cpu()
                    masks = masks.detach().cpu().unsqueeze(1)
                    show_img_and_mask(
                        images, masks, output_mask, title=f"Epoch {epoch+1} Evaluation",
                    )        
        
        val_loader = tqdm(val_data_loader)
        for i, (images, masks) in enumerate(val_loader):
            
            model.eval()
            
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

            val_loss += loss.item()
            val_loader.set_description(f"Val Loss: {loss.item():.04f}")

        train_losses.append(train_loss / len(train_data_loader))
        val_losses.append(val_loss / len(val_data_loader))

        # save model checkpoint
        save_model(model, epoch)

        # show loss plot
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.legend(loc="best")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Plot")
        plt.show()

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
    model_checkpoint = args.checkpoint
    epochs = args.epochs
    DEBUG = args.debug

    # hyperparams
    num_classes = 3
    batch_size = 1

    ################################## LOAD DATASET ##################################
    print(
        "\n----------------------- Loading and Preparing Data -----------------------"
    )

    train_data_loader, val_data_loader = preprocess_data(data_path, num_classes=num_classes, batch_size=batch_size)

    print("\n----------------------- Data Preprocessing Completed -----------------------")

    ################################## LOAD MODEL ##################################
    print("\n----------------------- Loading Model -----------------------")
    # from smp
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,  # grayscale images
        classes=3,  # background, needle, lamella
    )
    
    # Use gpu for training if available else use cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load model checkpoint
    if model_checkpoint:
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        print(f"Checkpoint file {model_checkpoint} loaded.")

    ################################## SANITY CHECK ##################################
    print("\n----------------------- Begin Sanity Check -----------------------\n")

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
    print("\n----------------------- Begin Training -----------------------\n")

    # train model
    model = train_model(model, device, train_data_loader, val_data_loader, epochs, DEBUG=DEBUG)

    ################################## SAVE MODEL ##################################




# ref:
# https://towardsdatascience.com/train-a-lines-segmentation-model-using-pytorch-34d4adab8296
# https://discuss.pytorch.org/t/multiclass-segmentation-u-net-masks-format/70979/14
# https://github.com/qubvel/segmentation_models.pytorch
