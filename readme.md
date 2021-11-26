# MWDB

**Table of Contents**
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
* Numpy
* Opencv
* Sklearn
* Skimage
* Json
* Python3.5 or later
* PIL 
* PyMongo Library (Mongo Client) 
* Mongo DB 


**Installation:**

On ubuntu/linux:
* Pip install -r requirement.txt


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














License
Distributed under the MIT License. 
Contact
