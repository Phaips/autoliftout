#!/usr/bin/env python3

import numpy as np
import pytest
import segmentation_models_pytorch as smp
import torch
from liftout.detection import DetectionModel

# TODO: replace weights file with something stable

@pytest.fixture
def weights_file():
    # yield r"C:\Users\Admin\Github\autoliftout\liftout\model\models\fresh_full_n10.pt"
    yield "/Users/patrickcleeve/Documents/university/bio/demarco/autoliftout/liftout/model/models/fresh_full_n10.pt"


@pytest.fixture
def detection_model(weights_file):

    yield DetectionModel.DetectionModel(weights_file=weights_file)


def test_DetectionModel_init(detection_model, weights_file):

    assert type(detection_model) == DetectionModel.DetectionModel
    assert detection_model.weights_file == weights_file

def test_cuda_device(detection_model):
    """test whether the device is loaded correctly"""
    
    assert detection_model.device == torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_transformation(detection_model):
    """ test image preprocess transformation is correct """
    img = np.zeros((1024, 1536, 1), dtype=np.uint8)
    img_t = detection_model.preprocess_image(img)

    assert img_t.shape == (1, 1, 256, 384)

def test_cuda_image(detection_model):
    """ test image is loaded onto device correctly """
    img = np.zeros((1024, 1536, 1), dtype=np.uint8)
    img_t = detection_model.preprocess_image(img)

    assert img_t.device == torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_load_model(detection_model):
    """
    # test that:
    #   - the correct model has been loaded
    #   - the model is in evaluation mode
    #   - the model is on the correct device
    """

    assert detection_model.model.__class__ == smp.unet.model.Unet
    assert detection_model.model.training is False
    assert next(detection_model.model.parameters()).device == detection_model.device


def test_model_inference(detection_model):
    """ test model inference pipeline works correctly"""
    img = np.zeros((1024, 1536, 1), dtype=np.uint8)

    rgb_mask = detection_model.model_inference(img)

    assert rgb_mask.shape == (256, 384, 3)


def test_decode_segmap(detection_model):

    zeros = np.zeros((256, 384, 1), dtype=np.uint8)

    rgb_mask = detection_model.decode_segmap(zeros)
    rgb_mask = rgb_mask.reshape(256, 384, 3)  # TODO: this should be decode_output

    assert np.all(rgb_mask == [0, 0, 0])  # assert black

    ones = np.ones_like(zeros) 
    rgb_mask = detection_model.decode_segmap(ones)
    rgb_mask = rgb_mask.reshape(256, 384, 3)  # TODO: this should be decode_output

    assert np.all(rgb_mask == [255, 0, 0])  # assert red

    twos = np.ones_like(zeros) * 2
    rgb_mask = detection_model.decode_segmap(twos)
    rgb_mask = rgb_mask.reshape(256, 384, 3)  # TODO: this should be decode_output

    assert np.all(rgb_mask == [0, 255, 0])  # assert green
