# **Traffic Sign Classifier**

## Writeup Submit

---

**Build a Traffic Sign Classifier Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./SingleExample.png "Visualization"
[image2]: ./ClassDistribution.png "Class Distribution"
[image3]: ./PreprocessedExample.png "Prepocessed Images"
[image4]: ./TrainValidTestAccuracy.png "Performance"
[image5]: ./ResizedFiveNewImages.png "Five New Images"
[image6]: ./PreprocessedFiveNewImages.png "Preprocessed Five New Images"
[image7]: ./Predict1.png "Predict1"
[image8]: ./Predict2.png "Predict2"
[image9]: ./Predict3.png "Predict3"
[image10]: ./Predict4.png "Predict4"
[image11]: ./Predict5.png "Predict5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/RuiyeNi/CarND-Term1-Project2-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier_Submit.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy and pandas libraries to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Each image is 32x32x3 in RGB, and one examplar image is displayed below with its sign labeled. 

![Visualization][image1]

There are totally 43 signs and their distribuiton in the training dataset is visualized in a histogram. As shown in the figure, training dataset is not very balanced across different labels. 
![Class Distribution][image2]

Ten random samples are also visualized for each type of sign in the [Jupyter Notebook](https://github.com/RuiyeNi/CarND-Term1-Project2-Traffic-Sign-Classifier/master/Traffic_Sign_Classifier_Submit.ipynb)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As suggested in [Sermanet and LeCun's publication](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), color information is not that helpful to improve classifier's performance. Therefore, my first step was to convert the images to grayscale, which reduced the datasize and led to faster training. By looking closer into the transformed images, I discovered that some images with low exposure were not well handeled by grayscale conversion, and contrast equalization process were applied to enchance images' contrast.

Here is an example of ten samples of traffic sign 'speed limit 60 (km/h)' before and after grayscaling, and after contrast equalization processing.

![Prepocessed Images][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale equalized image 			|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU 	                |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x16                    |
| Fully connected		| output 120      								|
| RELU                  |                                               |
| Dropout               | keep_prob = 0.8                               |
| Fully connected		| output 84        								|
| RELU					| 												|
| Dropout				| keep_prob = 0.9								|
| Fully connected       | output 43                                     |
| Softmax               | output 43                                     |


 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with learning rate 0.001, batch size 128 and epochs 100. Early stopping was also applied to prevent overfitting. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.3%
* validation set accuracy of 95.0%
* test set accuracy of 92.7%

I started with the well-known LeNet architecture since it demonsrtated good performance for classifying digits' images in the previous project. Convolution network was also expected to work well on traffic sign images with slightly distortion. With the default LeNet architecture, training set accuracy could achieve nearly 100% accuracy, but validation accuracy was barely around 90%, which indicated the model wasn't generalized well enough and there was an overfitting problem. To tackle the overfitting, I tried mainly two ways. On one hand, I added dropout layers after two fully connected layers so that the model weights would be optmized for more generalization. On the other hand, I tuned the batch size and number of epochs, and applied early stopping to obtain model trained with least validation loss. Dropout rates were exprimented between 0.5 and 0.9, batch size was tested between 30 and 256, epoches were tried from 10 to 300, and learning rate was chosen between 0.0005 and 0.001. Finally, I came up with a model with improved validaiton accuray of 95% by using learning rate 0.001, epoches 100, batch size 128, dropout rate 0.8 for the first fully connected layer, and dropout rate 0.9 for the second fully connected layer. Though it seemed there was still room to improve the model's performance, dropout was proved to be able to alleviate overfitting. Given more time, data augmentation, such as adding noise to training set and distort training samples, could be further experimented to build a more robust model. 

Here is a image shows the training process:
![Train valid Accuracy][image4]


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I googled five german traffic signs and resized them as 32x32x3 as shown below:

![Five New Test Images][image5]

These five images were preprocessed as the training set into grayscale and exposure equalized images as displayed below:
![Preprocessed New Test Images][image6]


The second image and third image might be difficult to classify. Because the second sign's pattern is not exactly the same as the one in training dataset and the sign was also more frame full-filled. Regarding the third image, its distortion might increase the recognition difficulty. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| No Entry      		| No Entry   									|
| Road Work     		| Road Work										|
| Stop  				| Stop											|
| Yield 	      		| Yield      					 				|
| 30 km/h   			| 30 km/h            							|


The model was able to correctly guess all of the 5 traffic signs, which gives an accuracy of 100%. This result is better than the test set accuracy 92.7%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 24th ~ 26th cells of the [Jupyter Notebook](https://github.com/RuiyeNi/CarND-Term1-Project2-Traffic-Sign-Classifier/master/Traffic_Sign_Classifier_Submit.ipynb)

As it was expected, classifier was pretty sure about the first, fourth, and fifth images, but relatively less for the second and third images. For the second image, the top five predicted signs all have triangle frames but with different contents inside, which further proves my previous guess. While for the third image, because of the hexagon‘s distortion, the classifier was not so sure about its frame shape. 

For the first images:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| No Entry   									|
| 0.00     				| No Passing 									|
| 0.00					| Turn Left Ahead								|
| 0.00	      			| Roundabout Mandatory			 				|
| 0.00				    | Go Straight or Left   						|

![Predict1][image7]



For the second image:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.99         			| Road Work  									|
| 0.00     				| Road Narrows on the Right						|
| 0.00					| Pedestrians									|
| 0.00	      			| Right-of-way at the Next Intersection			|
| 0.00				    | Slippery Road      							|

![Predict2][image8]



For the third image :

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.95         			| Stop       									|
| 0.05     				| No Vehicles 									|
| 0.00					| Speed Limit (60km/h)							|
| 0.00	      			| Yield				 			   				|
| 0.00				    | Turn Right Ahead     							|

![Predict3][image9]



For the fourth image :

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Yield  										|
| 0.00     				| Ahead Only 									|
| 0.00					| No Vehicles									|
| 0.00	      			| Speed Limit (50km/h)							|
| 0.00				    | Speed Limit (60km/h)   						|

![Predict4][image10]



For the fifth image :

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Speed Limit (30km/h) 							|
| 0.00     				| End of Speed Limit (80km/h)					|
| 0.00					| Speed Limit (80km/h)							|
| 0.00	      			| Keep Right					 				|
| 0.00				    | Speed Limit (20km/h)   						|

![Predict5][image11]



