#**Traffic Sign Recognition**

##Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/dataset_image.png "Datset Image"
[image2]: ./examples/visualization.png "Visualization"
[image3]: ./examples/original1.png "Original Image"
[image4]: ./examples/gray_norm.png "Grayscale Normalized Image"
[image5]: ./examples/original2.png "Traffic Sign 1"
[image6]: ./examples/translated.png "Traffic Sign 2"
[image7]: ./examples/scaled.png "Traffic Sign 3"
[image8]: ./examples/warped.png "Augumented Visualization"
[image9]: ./examples/brightness.png "Traffic Sign 3"
[image10]: ./examples/visualization_augumented.png "Augumented Visualization"
[image11]: ./test_images/Turn_Left_Ahead_No34.png "Traffic Sign 8"
[image12]: ./test_images/Speed_Limit_60kmph_No3.png "Traffic Sign 8"
[image13]: ./test_images/No_Entry_No17.png "Traffic Sign 8"
[image14]: ./test_images/Roadwork_No25.png "Traffic Sign 8"
[image15]: ./test_images/Keep_Right_No38.png "Traffic Sign 8"
[image16]: ./test_images/Speed_Limit_30kmph_No1.png "Traffic Sign 8"
[image17]: ./test_images/Pedestrians_No27.png "Traffic Sign 8"
[image18]: ./test_images/General_Caution_No18.png "Traffic Sign 8"
[image19]: ./examples/lenet.png "LeNet architecture"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Writeup / README

####Q1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/abhardwajnv/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Step 1: Dataset Summary & Exploration

####Q1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Load the Data: I used the pickle library to load the given dataset. The code for this step is contained in first cell of IPython notebook.

Summary of Dataset: I used Python Numpy library to calculate the summary statistics of the traffic sign data set. The code for this step is contained in the second code cell of the IPython notebook.  

Below are the statistics i got:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####Q2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third to fifth code cell of the IPython notebook.  
For this step i used matplotlib & random libraries of python.

Plotting Traffic Sign Images: Here i took 10 random images from the training dataset with the help of random.randint function and plotted them using matplotlib.pyplot library.

![alt text][image1]

Plotting the count of each sign: Here i used the pyplot library again to plot a histogram of Frequncy of Data Points for 43 unique classes identified in Q1 with the count of each of them. I plotted this chart for Training and Test Datasets.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribution available for each unique class. As the graph shows currently data is not even.

![alt text][image2]


### Step 2: Design and Test a Model Architecture

####Q1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sixth and seventh code cell of the IPython notebook.

Initially i shuffled the dataset using shuffle function from sklearn library as linear dataset affects in training.
With shuffled dataset first i converted the images to grayscale using opencv library as it reduces the computation time and takes less memory, helps in absence of GPU's.
Post this i normalized the images as it helps in achieving consitency with the ranges of dataset. As also explained in the sessions it helped achieving a relatively lower mean value as well.
Post this i took the mean of the normalized dataset images and then plotted the difference between original images and grayscale normalized images.

Here is an example of traffic sign image before and after this step:

![alt text][image3]  ![alt text][image4]


####Q2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Here i decided to apply augumentation to the dataset first as the given data was not sufficient for achiving higher accuracy rates. As we saw already that the given dataset was uneven hence chances were that network might get biased towards the higher datapoint classes. So to tackle this i made the datapoints atleast 800 for each class by putting up each image to a pipeline that applies translation, scaling, warping and brightness adjustments to the image. These images were then added back to the dataset.

This helped with 2 most importent factors, one increasing the size of the dataset and second creating more datapoints for the lower size classes making them more efficient while training.
The code for augumentation & data generation is contained in 8th to 13th code cells of the IPython notebook.

Here are the examples of each augumentation function w.r.t the original image fed to the function.

Original Image
![alt text][image5]

Translation
![alt text][image6]

Scaling
![alt text][image7]

Warp
![alt text][image8]

Brightness adjustments
![alt text][image9]

The difference between the original data set and the augmented data set is shown by below histogram.

![alt text][image10]

Post Augumentation and data generation in training dataset i shuffled the data again to break the linear pattern of datasets since i had to split and reassign the data to validation. Then i splitted the training dataset and assigned 20% to validation dataset due to low size of validation dataset. With this splitting i achieved roughly 60-20-20 ratio for training-validation-test datasets respectively.
The code for splitting and assignment is contained in 14th & 15th code cells of IPython notebook.

Below is the summary of change in dataset:

* The size of original training set : 34799
* The size of training set post data generation : 46480
* The size of new training set post splitting : 37184
* The size of 10% splitted dataset : 9296
* The size of old validation set : 4410
* The size of new validation set post splitting : 13706
* The size of test dataset (Not Changed) : 12630

####Q3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in 17th & 18th code cells of the IPython notebook.
I used the same LeNet model we used in class with some modification required for current dataset.

My final model consisted of the following layers:

| Layer         		    |     Description	        					            |
|:---------------------:|:---------------------------------------------:|
| Input         		    | 32x32x1 grayscale image   							      |
| Convolution1 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	  |
| RELU1					        |	Outputs 28x28x6											          |
| Max pooling1	     	  | 2x2 stride,  outputs 14x14x6 				          |
| Convolution2 5x5	    | 1x1 stride, same padding, outputs 10x10x16 	  |
| RELU2					        |	Outputs 10x10x16											        |
| Max pooling2	     	  | 2x2 stride,  outputs 5x5x16 				          |
| Fully connected	      | Shape 400 (Flatten)    									      |
| Fully connected1	    | Inputs 400, outputs 120									      |
| Fully connected2	    | Inputs 120, outputs 84									      |
| Fully connected3	    | Inputs 84, outputs 43 									      |


####Q4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The Code for training the model is contained in 16th & 19th to 21st code cell of the IPython notebook.
To train the network model i used Adam optimizer which we implemented in LeNet class.
I changed few hyperparamters which are as follows:

Epochs: 25
batch size: 64
learning rate: 0.001
mu: 0
sigma: 0.1

####Q5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 22nd cell of the Ipython notebook.

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 96.8%
* test set accuracy of 92.5%

To start with i used the same LeNet architecture we learnt in sessions.
Here's how the architecture looks like:

![alt text][image19]

First change was to include more number of classes i.e., 43 instead of 10.

Then i tried with this architecture and my validation accuracy was reaching ~94% with initial set 10 Epochs.
I tried with changing different learning rates and Epochs.
With 25 Epochs and 0.001 learning rate i was able to achieve 99.6% accuracy.
I tried to further reduce the overfitting by chaging the convolution layer parameters but i was not getting desired results hence i decided to keep the architecture as it is.


### Step 3: Test a Model on New Images

####Q1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image11] ![alt text][image12] ![alt text][image13] ![alt text][image14]
![alt text][image15] ![alt text][image16] ![alt text][image17] ![alt text][image18]

The first & second image might be difficult to classify because the images are not parallel to camera capture angle. Images are a bit tilted. In addition first image is very bright and second image has watermark on the sign board.

The third image also has a watermark on the sign board along with a bright background which gives contrast to the sign board.

The fourth image is having bright sun reflection on top left side of the sign board making it hard to identify.

The sixth, seventh & eighth image are having noise in background in terms of trees and other objects.


Because of the above mentioned reasons i chose these images to cover different aspects of real life detection scenario's.
The code for loading these images is contained in 23rd code cell of the IPython notebook.

####Q2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 24th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			            |     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| Turn Left Ahead      	| Turn Left Ahead   									          |
| Speed Limit (60km/h)  | Go straight or left 										      |
| No Entry					    | Stop											                    |
| Roadwork	      		  | Double curve				 				                  |
| Keep Right			      | Keep Right      							                |
| Speed Limit (30km/h)	| Speed limit (50km/h)											    |
| Pedestrians	      		| Pedestrians					 				                  |
| General Caution			  | General Caution      							            |

The model was able to correctly guess 4 of the 8 traffic signs, which gives an accuracy of 50%.

As the second & third images had watermarks on the sign boards which confused the network to properly identify the images.
4th image had bright sunlight reflection which again blinded a part of signboard to be identified.
In the sixth image since the noise in the background was high due to which identification was not appropriate.

Although apart from second image all the other image predictions were in Top 3 count.
This can be improved with adding more such images to the training dataset.

####Q3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| 1            			    | Turn Left Ahead   									          |
| 0        				      | Speed Limit (60km/h) 										      |
| 0.17  					      | No Entry											                |
| 0.000006	   			    | Roadwork					 				                    |
| 1				              | Keep Right      							                |
| 0.03					        | Speed Limit (30km/h)											    |
| 0.97	      			    | Pedestrians					 				                  |
| 1 				            | General Caution      							            |


For 1st, 5th & 8th images probability is 100% so no doubt in identification of these images.
For 7th image probability is 97% and being the highest probability sign is identified with this which is correct.
For 2nd image network does not identify it in top 5 probabilities.
For rest for 3rd, 4th and 6th probablities are low due to which signs are not identified properly.

### (OPTIONAL) Step 4: Visualize the Neural Network's State with Test Images

Not Implemented Yet.
