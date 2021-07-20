#!/bin/bash

echo "Running convert labels to dataset script..."

# TODO: this doesn't work on windows
# loop through each label

# set label directory
if [ -z $1 ]
then
    LABEL_PATH="data/train/raw/*.json"
else
    LABEL_PATH=$1"/*.json"
fi

for FILE in $LABEL_PATH; 
    do echo $FILE; 

    # get basename of file
    fbname=$(basename "$FILE" .json)
    echo "$fbname"

    # save into separate folder
    OUT_PATH="data/train/"$fbname
    echo "$OUT_PATH"

    # # run labelme_json_to_dataset
    labelme_json_to_dataset $FILE -o $OUT_PATH
done





