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
import wandb

def save_model(model, epoch):
    """Helper function for saving the model based on current time and epoch"""
    
    # datetime object containing current date and time
    now = datetime.now()
    # format
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S") + f"_n{epoch+1:02d}"
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
            wandb.log({"train_loss": loss.item()})
            data_loader.set_description(f"Train Loss: {loss.item():.04f}")

            if i % 100 == 0:
          
                if DEBUG:
                    model.eval()
                    with torch.no_grad():

                        outputs = model(images)
                        output_mask = decode_output(outputs)
                        # TODO: hstack these outputs...

                        img_base = images.detach().cpu().squeeze().numpy()
                        img_rgb = np.dstack((img_base, img_base, img_base))
                        gt_base = decode_segmap(masks.detach().cpu().permute(1, 2, 0))

                        wb_img = wandb.Image(img_rgb, caption="Input Image")
                        wb_gt = wandb.Image(gt_base, caption="Ground Truth")
                        wb_mask = wandb.Image(output_mask, caption="Output Mask")
                        wandb.log({"image": wb_img, "mask": wb_mask, "ground_truth": wb_gt})
                           
        
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
            wandb.log({"val_loss": loss.item()})
            val_loader.set_description(f"Val Loss: {loss.item():.04f}")

        train_losses.append(train_loss / len(train_data_loader))
        val_losses.append(val_loss / len(val_data_loader))

        # save model checkpoint
        save_model(model, epoch)


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

    # weights and biases setup
    wandb.init(project="autoliftout", entity="patrickmonash")

    # hyperparams
    num_classes = 3
    batch_size = 1

    wandb.config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "num_classes": num_classes
    }

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
        imgs = imgs.to(device)
        output = model(imgs)
        pred = decode_output(output)

        print("imgs, masks, output")
        print(imgs.shape, masks.shape, output.shape)


        img_base = imgs.detach().cpu().squeeze().numpy()
        img_rgb = np.dstack((img_base, img_base, img_base))
        gt_base = decode_segmap(masks[0].permute(1, 2, 0).squeeze())

        wb_img = wandb.Image(img_rgb, caption="Input Image")
        wb_gt = wandb.Image(gt_base, caption="Ground Truth")
        wb_mask = wandb.Image(pred, caption="Output Mask")
        wandb.log({"image": wb_img, "mask": wb_mask, "ground_truth": wb_gt})

    ################################## TRAINING ##################################
    print("\n----------------------- Begin Training -----------------------\n")

    # train model
    model = train_model(model, device, train_data_loader, val_data_loader, epochs, DEBUG=DEBUG)

    ################################## SAVE MODEL ##################################


# ref:
# https://towardsdatascience.com/train-a-lines-segmentation-model-using-pytorch-34d4adab8296
# https://discuss.pytorch.org/t/multiclass-segmentation-u-net-masks-format/70979/14
# https://github.com/qubvel/segmentation_models.pytorch
