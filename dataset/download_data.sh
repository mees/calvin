#!/bin/bash

# Download, Unzip, and Remove zip
if [ "$1" = "D" ]
then

    echo "Downloading task_D_D ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
    unzip task_D_D.zip && rm task_D_D.zip
    mv task_A_A task_D_D
    echo "saved folder: task_D_D"
elif [ "$1" = "ABC" ]
then

    echo "Downloading task_ABC_D ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip
    unzip task_ABC_D.zip && rm task_ABC_D.zip
    mv task_BCD_A task_ABC_D
    echo "saved folder: task_ABC_D"

elif [ "$1" = "ABCD" ]
then

    echo "Downloading task_ABCD_D ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_D.zip
    unzip task_ABCD_D.zip && rm task_ABCD_D.zip
    mv ABCD_A ABCD_D
    echo "saved folder: task_ABCD_D"

else
    echo "Failed: Usage download_data.sh D | ABC | ABCD"
    exit 1
fi
