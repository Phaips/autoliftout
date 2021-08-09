#!/usr/bin/env python3

import numpy as np


import pytest
from liftout.detection import DetectionModel
import segmentation_models_pytorch as smp
from torchvision import transforms
import torch

# TODO: replace weights file with something stable

@pytest.fixture
def weights_file():
    # yield r"C:\Users\Admin\Github\autoliftout\liftout\model\models\fresh_full_n10.pt"
    yield "/Users/patrickcleeve/Documents/university/bio/demarco/autoliftout/liftout/model/models/fresh_full_n10.pt"

@pytest.fixture
def detection_model(weights_file):

    yield DetectionModel.DetectionModel(weights_file=weights_file)


def test_DetectionModel_init(detection_model, weights_file):

    assert  type(detection_model) == DetectionModel.DetectionModel
    assert detection_model.weights_file == weights_file

# load model test separately

def test_transformation(detection_model):

    img = np.zeros((1024, 1536, 1), dtype=np.uint8)
    img_t = detection_model.preprocess_image(img)

    assert img_t.shape == (1, 1, 256, 384)

def test_load_model(detection_model):
    # TODO: comment this properly

    assert detection_model.model.__class__ == smp.unet.model.Unet
    assert detection_model.model.training is False
    assert next(detection_model.model.parameters()).device == torch.device(type="cpu")
    # TODO: gpu support


def test_model_inference(detection_model):

    img = np.zeros((1024, 1536, 1), dtype=np.uint8)

    rgb_mask = detection_model.model_inference(img)

    assert rgb_mask.shape == (256, 384, 3)

def test_decode_segmap(detection_model):

    img = np.ones((256, 384, 1), dtype=np.uint8)

    rgb_mask = detection_model.decode_segmap(img)
    rgb_mask = rgb_mask.reshape(256, 384, 3) # TODO: this should be decode_output

    assert np.all(rgb_mask == [255, 0, 0]) # assert red

    zeros = np.zeros_like(img)
    rgb_mask = detection_model.decode_segmap(zeros)
    rgb_mask = rgb_mask.reshape(256, 384, 3)  # TODO: this should be decode_output

    assert np.all(rgb_mask == [0, 0, 0]) # assert black

    twos = np.ones_like(img) * 2
    rgb_mask = detection_model.decode_segmap(twos)
    rgb_mask = rgb_mask.reshape(256, 384, 3)  # TODO: this should be decode_output

    assert np.all(rgb_mask == [0, 255, 0]) # assert green
#
# det_model = DetectionModel.DetectionModel(weights_file=r"C:\Users\Admin\Github\autoliftout\liftout\model\models\fresh_full_n10.pt")
# test_decode_segmap(det_model)