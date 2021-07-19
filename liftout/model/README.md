# Liftout Model

# TODO
- create base dataset
- write dataset documentation
- write labelling documentation
- improve data scripts
- fix app.py for use with new utils
- remove unused or consolidate files (exp_summary, mpl_click)
- fix / refactor train.py (maybe refactor dataset)
    - add a validation set to train.py?
- requirements.txt / environment.yml
- write proper readme for the model section


# Labelling

LabelMe was used to label the training images. 
https://github.com/wkentaro/labelme 


The dataset is available here: [TODO: LINK]

Guide for labelling new images


To convert the labelled polygons into the training format please run label_to_dataset.sh


`$ bash label_to_dataset.sh`



# Training


`$ python3 train.py `



# Evaluation


`$ streamlit run app.py `