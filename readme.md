Table of Contents
        About The Project
            Built With
        Getting Started
            Prerequisites
            Installation
        Usage
        Contributing
        License
        Contact


About The Project
The project has been done as a part of Phase 1 of CSE 515: Multimedia and Web Database.
In this project, you will experiment with
• image features,
• vector models, and
• similarity/distance measures


Prerequisites:
* Numpy
* Opencv
* Sklearn
* Skimage
* Json
* Python3.5 or later


Installation:


On ubuntu/linux:
* Pip install -r requirement.txt


Usage:
To run:
Step1:
# To create data folder and download the data. Run this command.
Python3 dataset_create.py
Step 2:
Python3  main.py --path [input_folder] --output [output_folder for task1 and task2] --test[test folder for task3 and task4] --task [task no.] --k [top k results]


Task1: 
For task1 place the image in input folder select task1 and model as required. The output json will be present in output folder.
For Color Moments:
python3 main.py --path input --output output --task task1 --model color_moment
For Extended Local Binary Pattern:
        python3 main.py --path input --output output --task task1 --model elbp
For HOG:
        python3 main.py --path input --output output --task task1 --model hog


Task 2:
For task2 place the image in input folder select task2. The output json will be present in output folder.
For Color Moments:
python3 main.py --path input --output output --task task2


Task 3:
For task 3 place the image_id image in input folder and the collection of images in output folder.Select the modeland the k value. The top k value will be displayed on the terminal.
For Color Moments:
        python3 main.py --path input --test test --task task3 --model color_moment --k 4
For Extended Local Binary Pattern:
python3 main.py --path input --test test --task task3 --model elbp --k 3
For HOG:
        python3 main.py --path input --test test --task task3 --model hog --k 3


Task 4:
For task 4 place the image_id image in input folder and the collection of images in output. The top k value will be displayed.
        python3 main.py  --path input --test test --task task4 --k 3


License
Distributed under the MIT License. 
Contact
Your Name: Mohd Zaid
ASU ID: 1222301444