# Liftout Model

# TODO
- create base dataset
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


To convert the labelled polygons into the training format please run label_to_dataset.sh


`$ bash label_to_dataset.sh`



# Training
To train the model run the following 

`$ python3 train.py `
--epochs: the number of epochs to train for
--checkpoint: start training from a model checkpoint
--debug: show visualisations for debugging during training


# Evaluation


`$ streamlit run app.py `