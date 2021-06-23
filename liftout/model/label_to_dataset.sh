#!/bin/bash

echo "Running convert labels to dataset script..."


# TODO: this doesn't work on windows

# loop through each label

LABEL_DIR="data/train/raw*.json"

for FILE in $LABEL_DIR; 
    do echo $FILE; 

    # get basename of file
    fbname=$(basename "$FILE" .json)
    echo "$fbname"

    # save into separate folder
    OUT_DIR="data/train/"$fbname
    echo "$OUT_DIR"

    # # run labelme_json_to_dataset
    labelme_json_to_dataset $FILE -o $OUT_DIR
done





