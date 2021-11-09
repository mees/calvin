#!/bin/bash

# Download, Unzip, and Remove zip
if [ "$1" = "A" ]
then

    echo "Downloading task_A_A ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_A_A.zip
    unzip task_A_A.zip && rm task_A_A.zip
    echo "saved folder: task_A_A.1.0"
elif [ "$1" = "B" ]
then

    echo "Downloading task_BCD_A ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_BCD_A.zip
    unzip task_BCD_A.zip && rm task_BCD_A.zip
    echo "saved folder: task_BCD_A"

elif [ "$1" = "full" ]
then

    echo "Downloading task_ABCD_A ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_A.zip
    unzip task_ABCD_A.zip&& rm task_ABCD_A.zip
    echo "saved folder: task_ABCD_A"

else
    echo "Failed: Usage download_data.sh A | B | full"
    exit 1
fi
