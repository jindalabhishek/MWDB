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

o Task 1 (Type Classifier)
▪ Given image folder as an input and their corresponding labels image type (for example: cc,con,jitter):
▪ Run Command python Task_1.py
▪ Associates Type Id to the test images


o Task 2 (Subject Classifier)

▪ Given image folder as an input and their corresponding labels subject id (for example: subject id 1-40):

▪ Run Command python Task_2.py

▪ Associates Subject Id to the test images


o Task 3 (Sample Classifier)

▪ Given image folder as an input and their corresponding labels image type, subject id,image sample id (for example: image-X-Y-Z):

▪ Run Command python Task_3.py

▪ Associates Sample Id to the test images


o Task 4 (Locality-Sensitive Hashing)

▪ Give input query image

▪ Number of Hash Functions

▪ Number of Hash Families

▪ Run Command python Task_4.py

▪ Outputs N similar images using LSH algorithm


o Task 5 (VA-Files)

▪ Give input query image

▪ Number of bits

▪ Number of similar images accordingly

▪ Run Command python Task_5.py

▪ Outputs N similar images using VA files algorithm

o Task 6 (Decision-tree-based relevance feedback)

▪  Give input query image

▪ Give feature model

▪ Give Hash function

▪  Hash family

▪ Run Command python Task_6.py

▪ Outputs N similar images and re-ranks the results as per the user's feedback using Decision Tree system.

o Task 7 (SVM-classifier-based relevance feedback)

▪ Give input query image

▪ Give number  of bit

▪ Give number of similar image

▪ Give number of feature model

▪ Run Command python Task_7.py

▪ Outputs N similar images and re-ranks the results as per the user's feedback using SVM system.

o Task 8 (Query and feedback interface)

▪ Run Command python Task_8.py

▪ This is used as an interface to execute Task 4,5,6,7
