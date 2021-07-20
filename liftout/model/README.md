# Liftout Model

# TODO
- create base dataset - DONE
- write dataset documentation
- write labelling documentation
- improve data scripts
- fix app.py for use with new utils
- remove unused or consolidate files (exp_summary, mpl_click)
- fix / refactor train.py (maybe refactor dataset)
    - add a validation set to train.py? - DONE
- requirements.txt / environment.yml
- write proper readme for the model section
- convert the dataset directly from the json? maybe

# Labelling

LabelMe was used to label the training images. 
https://github.com/wkentaro/labelme 


The dataset is available here: [TODO: LINK]

Guide for labelling new images


# Data Preparation
Once the data has been labelled, there are a few steps to prepare the data for training.

To convert the labelled polygons into the training format please run label_to_dataset.sh

the default path to the labelled data is: "data/train/raw*.json". Edit t

`$ bash label_to_dataset.sh`


# Training
To train the model run the following 

`$ python3 train.py `
--data: the path to the training data
--epochs: the number of epochs to train for
--checkpoint: start training from a model checkpoint
--debug: show visualisations for debugging during training

The training will also evaluate the data on a randomised hold out validatation set during training.

# Evaluation
The streamlit app can be used to evaluate the model quality, and calculated liftout features (e.g. lamella centre, needle tip). You can provide a path to a dataset (.tif images) as well as select different models. 

To run the streamlit app:
`$ streamlit run app.py `