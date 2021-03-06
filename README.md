# **Traffic Sign Recognition**

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)
[image1]: ./Writeup_images/training.png "Visualization training"
[image2]: ./Writeup_images/validation.png "Visualization validation"
[image3]: ./Writeup_images/test.png "Visualization test"
[image4]: ./Writeup_images/augment.png "Augment"
[image5]: ./Writeup_images/grayscale.png "Grayscaling"
[image6]: ./Writeup_images/test_img.png "Traffic Signs"
[image7]: ./Writeup_images/1.png "Traffic Sign 1"
[image8]: ./Writeup_images/2.png "Traffic Sign 2"
[image9]: ./Writeup_images/3.png "Traffic Sign 3"
[image10]: ./Writeup_images/4.png "Traffic Sign 4"
[image11]: ./Writeup_images/5.png "Traffic Sign 5"
[image12]: ./Writeup_images/6.png "Traffic Sign 6"
[image13]: ./Writeup_images/7.png "Traffic Sign 7"
[image14]: ./Writeup_images/8.png "Traffic Sign 8"
[image15]: ./Writeup_images/9.png "Traffic Sign 9"
[image16]: ./Writeup_images/10.png "Traffic Sign 10"
[image17]: ./Writeup_images/conv1.png "conv1"
[image18]: ./Writeup_images/conv1_relu.png "conv1_relu"
[image19]: ./Writeup_images/conv1_pool.png "conv1_pool"
[image20]: ./Writeup_images/conv2.png "conv2"
[image21]: ./Writeup_images/conv2_relu.png "conv2_relu"
[image22]: ./Writeup_images/conv2_pool.png "conv2_pool"
[image23]: ./Writeup_images/conv1_relu.png "conv1_reli"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Writeup / README

#### 1. Writeup / README includes all the rubric points.

Here is a link to my [project code](https://github.com/xushengyao/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here are exploratory visualizations of the data set. They are bar charts showing how the data is distributed in training set, validation set and test set separately.

<center>

![alt text][image1]


![alt text][image2]


![alt text][image3]
</center>

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I decided to generate additional data in order to get more data to train and also make the classifier more robust to handle different images. With the help of OpenCV functions, I firstly rotate the image with a random angle between -10 to 10 degree. Then, I transformed the image within +-5 pixels. I applied the augment functions to the training set, for the class which has less than 1500 samples, I augmented its data set twice in order to generate more images. For the rest part of the classes, I only generated the augmented image once. After the augmentation, the total number of the training set becomes: 89517. The difference between the original data set and the augmented data set is as follow:

<center>

![alt text][image4]
</center>


Then, I decided to convert the images to grayscale using OpenCV tool. The reason to grayscale the images is to simplify the image and reduce the computation.
Here is an example of a traffic sign image before and after grayscaling.
<center>

![alt text][image5]
</center>
As a last step, I normalized the image data to reduce the mean of the training set from all the data per each dimension. The code for making pre-processing on my training data set is located in the 8th cell of the Jupyter notebook.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

<center>

|      Layer      |                 Description                 |
|:---------------:|:-------------------------------------------:|
|      Input      |           32x32x1 grayscale image           |
| Convolution 5x5 | 5x5 stride, valid padding, outputs 28x28x6  |
|      RELU       |                                             |
|   Max pooling   | 2x2 stride, valid padding,  outputs 14x14x6 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
|      RELU       |                                             |
|   Max pooling   |  2x2 stride, valid padding, outputs 5x5x16  |
|     Flatten     |                 output 400                  |
| Fully connected |                 output 120                  |
|      RELU       |                                             |
|     Dropout     |               keep_prob = 0.7               |
| Fully connected |                  output 84                  |
|      RELU       |                                             |
|     Dropout     |               keep_prob = 0.7               |
| Fully connected |                  output 43                  |
</center>

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, initially I tried 30 epochs, but I found that more epochs could get a higher accuracy. So I switched to 80 epochs and get a better performance.

Regarding the optimizer, the batch size, and the learning rate, I just used the default values. To reduce the overfitting, I applied dropout, and set the keep_prob as 0.7 when training,

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

* What architecture was chosen?
  I used the LeNet architecture as the baseline. While, I made some changes on the original one to meet the required accuracy specifications, including changing the length of output from 10 to 43 and adding dropout layers to reduce the overfitting.
* Why did you believe it would be relevant to the traffic sign application?
  LeNet works very good with the grayscale  MINST training set. Since all traffic sign images are 32*32 size and have been converted to grayscale, I think it's very close to the MINST classification problem and LeNet should perform well on classifying traffic sign images.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  In general, the validation and test accuracy are very close, but sill less than the training accuracy, which is a little bit of overfitting.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.961
* test set accuracy of 0.951

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:
<center>

![alt text][image6]
</center>

The first image might be difficult to classify since the image I pick is not a standard German traffic sign. The rest of images should be recognized easily.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

<center>

|        Image         |      Prediction      |
|:--------------------:|:--------------------:|
|   General Caution    |   General Caution    |
|  Children crossing   |  Children crossing   |
|      Keep right      |      Keep right      |
|       No entry       |       No entry       |
|     Right of way     |     Right of way     |
|      Road work       |      Road work       |
| Roundabout mandatory | Roundabout mandatory |
| Speed limit (30km/h) | Speed limit (20km/h) |
|         Stop         |         Stop         |
|        Yield         |        Yield         |
</center>

The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This compares favorably to the accuracy on the test set of 95.1%.

Generally speaking, the model works well on most images and the accuracy is close to the test accuracy. I am a little bit of surprised that it can recognize the image one, which is not a standard traffic sign. While, I am also surprised it failed for the speed limit sign, and mixed the speed limit(30km/h) with the speed limit(20km/h). So, looks like the model still has some trouble predicting different speed limit signs.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Below are the predicted probabilities distribution for ten tested images.

<center>

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
</center>

In general, according to the distribution comparison, the model is quite confident to get the predicted result in most cases.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

As the layer goes deeper, most noise are gone and only pixels with import features have been kept.

<center>
<center>conv1</center>

![alt text][image17]

<center>conv1_relu</center>

![alt text][image18]

<center>conv1_max_pool</center>

![alt text][image19]

<center>conv2</center>

![alt text][image20]

<center>conv2_relu</center>

![alt text][image21]

<center>conv2_max_pool</center>

![alt text][image22]
</center>
