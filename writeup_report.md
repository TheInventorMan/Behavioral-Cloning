# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_resources/test.jpg "Raw front view"
[image2]: ./writeup_resources/cropped_scaled.jpg "Cropped and scaled"
[image3]: ./writeup_resources/flipped.jpg "Flipped"
[image4]: ./writeup_resources/nvidia.png "Architecture"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolutional neural network
* writeup_report.md summarizing the results
* video.mp4 containing a front-facing view of the autonomous run
* video_perspective.mp4 displaying the augmented view passed to the model
* video_chase.mp4 showing the entire simulator screen during the test run

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. It depends on preprocess.py, which is used for preprocessing and augmenting the dataset and saving it in a separate folder.  

Each of the files show the pipeline I used for training and validating the model, and also contain comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a normalization layer, followed by a series of convolutional layers with 5x5 or 3x3 filter sizes, in addition to three fully-connected layers.  

The model includes RELU layers to introduce nonlinearity (driveModel.py line 12), and the data is normalized in the model using a Keras BatchNormalization layer (driveModel.py line 8).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (driveModel.py line 30-37).  

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 78). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (driveModel.py line 4).

#### 4. Appropriate training data

When beginning the project, I attempted to manually drive the car around the track, but the lag from using an online virtual machine made it all but impossible. I finally resorted to using the sample data provided in the project repository.  

The sample data contained left and right images, in addition to the center view, allowing for training a lane-centering model.  

For details about how I processed and augmented the training data, see the next section.  

### Model Architecture and Training Strategy

#### 1. Design Strategy and Training Process

The overall strategy for deriving a model architecture was to first crop the image such that the background landscape and hood of the car are not visible, since that information is not useful to computing steering angle. The cropped image was then resized to 200x100, to reduce the number of pixels that would be passed through the network. This resizing process also changed the aspect ratio, stretching the image along the vertical axis. It can be somewhat compared to the perspective transformed applied to forward-facing images to determine lane curvature.

Here's a sample image, before and after the preprocessing:

![alt text][image1]
![alt text][image2]

Additionally, I augmented the dataset by flipping the images and reversing the steering angle that correspond to those images. Here is a preprocessed image, before and after flipping:

![alt text][image2]
![alt text][image3]

By the end of the data augmentation process, I had twice as many input pictures: originals + flipped. To ensure the car stays in the middle of the road, the steering angle associated with the left and right images would be adjusted by an offset, so that the car "seeing" those images from the center would add additional correction to the steering angle.

For training the network, the preprocessed image would first be normalized using the Keras BatchNormalization function to cause the mean and variance of the pixel values to be close to zero. Next, the image would be passed through several convolutional layers to extract key features of the road. This set of feature maps would then be flattened and sent through a dropout layer before the fully-connected layers. Dropout was added to prevent the model from overfitting to the training set. The fully-connected layers ended with a single output node, which issued the steering angle commands.

I had the initial idea of using Leaky ReLU activations for the entire network. My reasoning behind that was that this could potentially speed up training, since there is a gradient towards the non-linearity in the negative input regime. However, it seems that the online workspace instance does not support Leaky ReLU, so I reverted to standard ReLU activations and let the model train for longer.

Adding dropout when constructing the initial model resulted in low training and validation error from the start. I then experimented with various training batch sizes, and found that a batch_size of 256 was optimal, yielding a validation mean squared error of 0.012 after epoch 5. I used an Adam optimizer, so learning rate did not have to be tuned.

The final step was to run the simulator to see how well the car drove around track one. It exhibited some oscillatory behavior, essentially "ping-ponging" from side to side. The steering seemed a bit sluggish, which was not a problem for the gentler curves on the track, but the car ended up taking a swim following a sharp curve. I had initially thought my model was inadequately trained, but I revisited the drive.py code. It turned out that I was passing the entire field of view to the model, complete with the background landscape and hood. The model was seeing much less pronounced road features and failed as a result.

I added the crop and rescale code to the drive.py file and the drive went remarkably well. The car was almost "locked" in the center of the lane for most of the track.

At the end of the process, the vehicle was able to drive autonomously around the track while almost perfectly remaining centered in the lane.

Here's a [link to a first-person-view video of the run](./video.mp4)  

Here's a [link to a screen capture of the simulator during the run](./video_chase.mp4)  

#### 2. Final Model Architecture

The final model architecture (driveModel.py) is based on the end-to-end deep learning architecture from Nvidia.

Here is a diagram of the original architecture:

![alt text][image4]  

The modifications that I made included:
* Adjusting the inputs to accept a (200,100) image.
* Adding a dropout layer with a keep_rate of 0.5 before the fully connected layers
* Adding a dropout layer with a keep_rate of 0.5 after the 100-node fully-connected layer.
