# MWDB

**Table of Contents**
=======
﻿Table of Contents
        About The Project
            Built With
        Getting Started
            Prerequisites
            Installation
        Usage
        Contributing
        License
        Contact


**About The Project**
The project has been done as a part of Phase 2 of CSE 515: Multimedia and Web Database.
In this project, you will experiment with
• image features,
• vector models, 
• dimensionality curse,
• graph analysis


**Prerequisites:**
=======
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
<<<<<<< HEAD
* PIL 
* PyMongo Library (Mongo Client) 
* Mongo DB 


**Installation:**
=======


Installation:

On ubuntu/linux:
* Pip install -r requirement.txt


<<<<<<< HEAD
**Usage**:
To run:

**Task1:**

▪ Input: Feature model (HOG, ELBP...), Image Type (cc, con), k value (total semantic features), and dimensionality reduction technique (PCA, LDA...)  
▪ Output: Retrieves type weights latent semantic features. 

For Example:	
	Input:-
	   Welcome to Task 1 Demo. Enter the feature model (color_moment, elbp, hog):color_moment
	   Enter Type Id:cc
	   Select Dimension reduction technique: (1. PCA 2.SVD 3.LDA 4.k-means): 1
	   Image_vector_matrix dimension:  399 192
	   Enter K Value for Dimensionality Reduction:10
	   
**Task2:**

▪ Give Feature model (HOG, ELBP...), Subject Image ID (1, 2), k value (total semantic features), and dimensionality reduction technique (PCA, LDA...) as input accordingly.
▪ Run Command  python main_task_2.py 

For Example:
	Input:-
	  Enter the feature model (color_moment, elbp, hog):color_moment
	  Enter Subject Id:4
      Enter K Value for Dimensionality Reduction:10
      Select Dimension reduction technique: (1. PCA 2.SVD 3.LDA 4.k-means): 2
      Image_vector_matrix dimension:  120 192


**Task 3:** 

▪ Type feature latent semantic path, k value (total semantic features), dimensionality reduction technique (PCA, LDA...), and similarity function method as input accordingly.
▪ Run Command  python main_task_3.py 

For Example:
	Input:-
		Enter latent semantic path: ../Outputs/task_2_SVD_color_moment_4.json
		Calculating Similarity Matrix
		Enter k value: 10
		Enter dimensionality reduction technique (1. PCA 2.SVD 3.LDA 4.k-means): 1
		Calculating COV matrix
		Finding eigenvalues and eigenvectors
		../Outputs/task_3_PCA.json

**Task 4:**

▪ Subject feature latent semantic path, k value (total semantic features), dimensionality reduction technique (PCA, LDA...), and similarity function method as input accordingly.
▪ Run Command  python main_task_4.py 

For Example:
	Input:-
		Enter subject weight features path: ../Outputs/task_1_PCA_color_moment_cc.json
		Calculating Similarity Matrix
		Enter k value: 10
		Enter dimensionality reduction technique (1. PCA 2.SVD 3.LDA 4.k-means): 1
		Calculating COV matrix
		Finding eigenvalues and eigenvectors
		../Outputs/task_4_PCA.json
		
**Task 5: **

▪ Latent Semantic File path(.JSON) and query image are given as inputs accordingly
▪ Run Command  python main_task_5.py 

For Example:
	Input:-
		Input image used for all outputs : sample_images/jitter-image-184.png

**Task 6 & 7:**

▪ Latent Semantic File path(.JSON) and query image are given as inputs accordingly
▪ Run Command python main_task_6.py and  main_task_7.py accordingly.

For Example:
	Input:-
		Input image used for all outputs : sample_images/jitter-image-184.png

			
**Task 8:**

▪ Similarity Matrix retrieved from task 4, n value, m value as inputs accordingly
▪ Run Command  python main_task_8.py 


For Example:
	Input:-
		Enter similarity matrix name: ../Outputs/task_4_PCA.json
		Enter value n: 10
		Enter value m: 5
		
		
**Task 9:**

▪  Similarity Matrix retrieved from task 4, n value, m value as inputs accordingly
▪ Run Command  python main_task_9.py 

For Example:
	Input:-
		Welcome to Task 9 Demo. Enter the file which contains similarity matrix: ../Outputs/task_4_PCA.json
		Enter value of n:10
		Enter value of m:5
		Subject Id1:5
		Subject Id2:9
		Subject Id3:3












=======
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
>>>>>>> f08f75f90b69efda5255ef3425cdbb6b0b17648f


License
Distributed under the MIT License. 
Contact
<<<<<<< HEAD
=======
Your Name: Mohd Zaid
ASU ID: 1222301444
>>>>>>> f08f75f90b69efda5255ef3425cdbb6b0b17648f
