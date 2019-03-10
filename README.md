## Behavioral Cloning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
This is the fourth project of [Udacity's Self Driving Car Nano Degree Program](https://www.udacity.com/drive). The goal is to use a neural network to mimic the behavior of a human driver. The network outputs a steering angle based on the photo captured by a camara mounted on the car; the steering angle is then fed into the simulator in autonomous mode for testing.


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./plots/nVidia_model.png "NVIDIA model"
[image2]: ./plots/original_image.png "Original image"
[image3]: ./plots/flipped_image.png "Flipped image"
[image4]: ./plots/brightness_image.png "Brightness changed"
[image5]: ./plots/original_distribution.png "Original distribution"
[image6]: ./plots/augment_distribution.png "Augmented distribution"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The instructions suggest starting from a known self-driving car model and provided a link to the [nVidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The below diagram shows each layer of the network:

![alt text][image1]

The conversion of color space from RGB to YUV is also added at the very beginning of the network so that no change is required in drive.py to make simulator work. As the community suggests, ELU activation function is added at each layer. When training the network, ADAM optimizer is used with learning rate = 1e-4. The model works quite well with only 20 epochs; to make it more like a human driver, the model is trained for 200 epochs.

#### 2. Attempts to reduce overfitting in the model

Contrary to the popular approach in the community (adding dropout layer), an l2 regularizer is used in each layer. The validation loss is always close to the training loss, which indicates the elimination of overfitting issue.

#### 3. Model parameter tuning

Two parameters have been changed. The learning rate of ADAM optimizer is set to 1e-4; the weight of l2 regularizer is set to 1e-3. All the other parameters are taking their default values. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. A few data augmentation techniques are used to flatten the distribution of steering angle.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The nvidia model presented above was implemented first without any modification. The car cannot complete a lap, which is a minimum requirement of this project.

Then I added ELU activation function based on suggestions from the community. The model now seems overfitting as the validation loss is significantly larger than the training loss. Then I tried both the dropout layer and l2 regularizer; l2 regularizer seems to work better in this case.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is very close to the original nvidia model. The only difference is adding a Lambda layer for the conversion of color space so that no modification is needed in drive.py. 

#### 3. Creation of the Training Set & Training Process

Below is the guidelines I followed in order to capture good driving behavior:

  1. two or three laps of center lane driving
  2. one lap of recovery driving from the sides
  3. one lap focusing on driving smoothly around curves

To increase the robustness of the model, the data set is augmented so that: 1) the distribution of steering angle is flattened and 2) the model becomes less sensitive to the brightness of the image. Flipping and changing brightness are the only two approaches used in data augmentation and their effect is shown below.

Original image:

![alt text][image2]

Flipped image:

![alt text][image3]

Brightness changed:

![alt text][image4]

To flatten the distribution, I first chose a number of bins (I decided upon 23) and produced a histogram of the turning angles using `numpy.histogram`. The distribution of original data looks like this:

![alt text][image5]

As expected, the bin with the smallest steering angle (absolute value) has the most number of samples. The images that belong to other bins are duplicated by using the augmentation strategy mentioned earlier. The distribution of augmented data looks like this:

![alt text][image6]

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The model works okay with just 20 epochs and 200 epochs are used in training for the final model used in the video.
