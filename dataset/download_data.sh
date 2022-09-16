#!/bin/bash

# Download, Unzip, and Remove zip
if [ "$1" = "D" ]
then

    echo "Downloading task_D_D ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
    unzip task_D_D.zip && rm task_D_D.zip
    echo "saved folder: task_D_D"
elif [ "$1" = "ABC" ]
then

    echo "Downloading task_ABC_D ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip
    unzip task_ABC_D.zip && rm task_ABC_D.zip
    echo "saved folder: task_ABC_D"

elif [ "$1" = "ABCD" ]
then

    echo "Downloading task_ABCD_D ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_D.zip
    unzip task_ABCD_D.zip && rm task_ABCD_D.zip
    echo "saved folder: task_ABCD_D"

elif [ "$1" = "debug" ]
then

    echo "Downloading debug dataset ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip
    unzip calvin_debug_dataset.zip && rm calvin_debug_dataset.zip
    echo "saved folder: calvin_debug_dataset"


else
    echo "Failed: Usage download_data.sh D | ABC | ABCD | debug"
    exit 1
fi
