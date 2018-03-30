# Speech-Recognition-using-Tensor-Flow-
The repository contains files and codes used for tensorflow speech recognition project. 
The files include text documents containing the test, train and validation lists. The repository also includes the folder 
containing the test files for which the actual prediction was performed using the model trained. The codes include a number
of distinct trials which we execute to achieve better accuracy. The trials include execution of different models 
such as CNN, RNN and DNN for speech recognition. Moreover, the trials also included the results
obtained by varying the hyperparameters. 

Contents:
1. File_List
	a. Train_list.txt
	b. Val_list.txt
	c. Test_list.txt
	d. All_list.txt
	e. Testing_list.txt
	f. Validation_list.txt
2. Codes
	a. DNN_Speech.py
	b. CNN_Speech.py
	c. RNN_Speech.py

3. Train
4. Test
5. Readme.md

Description:

1. File_List:
The folder contains a set of text files containing the list of training files,
validation files and testing (train_list.txt, val_list.txt, test_list.txt), files split
from the list of all the files in the dataset. The list of all files is also included in the all_list.txt. 

a. Train_List.txt:
Contains the list of files used for training the model. The list was extracted from the training files provided by Kaggle. 

b. Val_List.txt:
Contains the list of files used for validating the model. The list was obtained from the validation_list.txt file provided by Kaggle.

c. Test_List.txt:
Contains the list of files used for running the primary testing of the model. 
The list is extracted from the testing_list.txt file provided by Kaggle.

d. All_List.txt:
Contains the list of all files excluding the test files required for submission.

e. Testing_List.txt:
The list of testing files in the training set provided by Kaggle.
f. Validation_List.txt:
The list of Validation files in the training set provided by Kaggle.

2. Codes:
a. vect.py:
i. The Code to extract MFCC features from a .wav file.
ii. The code performs the basic preprocessing task which includes pre-emphasis, filtering, windowing, filter banks and MFCC.
iii. The MFCC features are extracted from each frame of the filter bank.
iv. The obtained features are stored in a 2-dimensional numpy array.

b. DNN_Speech.py:
i. The Code to perform speech recognition using Deep Neural Networks using the tensorflow library.
ii. Performs the task with 3 hidden neural layers and an output layer.
iii. The input features are extracted using the get_vect function imported from the vect.py file
iv. The label are encoded as one hot vectors for easing the weight calculation

c. CNN_Speech.py
i. The code implements a Convolutional Neural Network model using the tflearn API for tensorflow.
ii. It implements a two convolutional layer and a fully connected layer neural network model to perform the task.
iii. The code validates the model with the validation set and provides the accuracy score.
