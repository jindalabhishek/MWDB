## Offline Image based Search Engine
This is an Offline Image based Search Engine project developed as 
part of Course Project CSE 515 Multimedia and Web Databases. 
This is the first phase of the project which retrieves
K similar greyscale images for the query image based on models
such as Color Moment, Extended Local Binary Patterns (ELBP)
and Histogram Over Gradients (HOG).

#Dependencies
The following packages are required to run this project
* Python 3
* Pip 3
* Sklearn Library
* Scikit-image
* Numpy
* PIL
* PyMongo Library(Mongo Client)
* Mongo DB

#Getting Started
### Install Python & Pip
Please refer to https://www.python.org/downloads/ for downloading Python 3.x

### Install Libraries Sklearn, Scikit-image, Pymongo
Use the following commands to install these libraries
* pip3 install sklearn
* pip3 install scikit-image
* pip3 install pymongo

###Install Mongo DB
Please refer to https://www.mongodb.com/try/download/community 
for installing Mongo DB on your machine

#Execution
The project is divided into the following 5 tasks.

## Task 0
This tasks retrieves Olivetti Faces from Sklearn dataset and 
stores them into the Mongo Database.
### Run Command
python main_task_0.py

## Task 1
This task takes image id as the input and retrieves the corresponding 
image pixels from MongoDB (which were saved in task-0). The task also takes
one of the model names (Color Moment, ELBP, HOG) as the input and 
outputs the corresponding feature descriptor.
### Dependencies
Successful Execution of Task-0
### Run Command
python main_task_1.py

## Task 2
This task takes folder path as the input and loads all the images 
inside the folder. It further computes the color moment, 
ELBP and HOG feature descriptor for each image and stores it into the
MongoDB.
### Run Command
python main_task_2.py

## Task 3 & 4
This task takes folder base path, image name and one of the model names 
(Color Moment, ELBP, HOG, All) as the input. It queries the MongoDB for the
corresponding folder base path and loads all the image feature
descriptors for that base path. It further computes the similarity measure
based on the input model and outputs K similar images based on the
similarity measure.
* Please note that task 3 and 4 are combined in one file. If the user inputs
'All' as the model name, the code will combine all the models and outputs k similar images.
### Dependencies
Successful Execution of Task-2
### Run Command
python main_task_3_4.py


