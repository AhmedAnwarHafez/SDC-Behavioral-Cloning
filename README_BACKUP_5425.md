#**Behavioral Cloning** 


**Behavrioal Cloning Project**

The goals / steps of this project are the following:
- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Plot the training and validation losses
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cropping.png "Cropping"
[image2]: ./examples/cameras.png "cameras"
[image3]: ./examples/recover1.gif "Recovery Image"
[image4]: ./examples/recover2.gif "Recovery Image"
[image5]: ./examples/recover3.gif "Recovery Image"
[image6]: ./examples/uncropped.jpg "Uncropped Image"
[image7]: ./examples/cropped.jpg "Cropped Image"
[image8]: ./examples/loss.png "Training and Validation Loss"
[image9]: ./examples/1-lap.png "1 lap"
[image10]: ./examples/after-binning.png "After binning"
[image11]: ./examples/after-binning-and-flipping.png "After binning and flipping"

---
### I Files Submitted

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
- **model.py** containing the script to create and train the model
- **drive.py** for driving the car in autonomous mode
- **model.h5** containing a trained convolution neural network 
- **writeup_report.md** (you're reading it)

---
### II Quality of Code

####1. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####2. Submission code is usable and readable

The model.py file contains the code for training and saving the weights and model of the convolution neural network. The file shows the pipeline I used for training and it contains comments to explain how the code works.

---
###III Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

I have used [Comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py)  steering architecture.
The architecture has 3 convolutional layers, each conv layer is followed by a sub-sampling and ELU activation function. The output is then flattened and connected to a 512 dense layer and connected to dense layer with just one neuron to output a single continous number.

For more details, please see [Final Model Architecture] section down below.

####2. Attempts to reduce overfitting in the model

To prevent overfitting, two dropouts were added, one between the flattened layer and the 512 dense layer, the other one is between the 512 layer and the last last layer.

After each training trial, I plotted the training and validation losses to find if the model is either over or underditting and change the number of epochs if necessary.

####3. Model parameter tuning

I trained the model using Adam optimzer as an adaptive learning optimizer. The dafault learnig rate was set by keras.

For faster convergince, I introduced a lamda layer on top of my architecure as a normalizer. The normalization layer takes the input image and divides by 127.5 and subtracts 1.


####4. Appropriate training data

I carefully collected my own data by using the latest beta simulator and used the mouse to steer the car. The mouse was reasonably better than using a keyboard or a UBS joystick controller. I found that the mouse gives smoother control.

All three cameras were used in this project.

---
###IV Architecture and Training Documentation

####1. Solution Design Approach

In my opinion, the heart of this project is not about coding and training a model but it is about getting quality data. Deep neural networks are too sensitive to bad data and biased data. The majority of the first track is biased towards straight driving. I recorded a single lap drive and here is a plot of the angle distribution.

![][image9]

The plot shows a peak in the middle and this peak means that there are too many examples of zero or near-zero steering angles. If a model is trained on this recording, it would most likely go off track because the model learned enough about straight driving but not enough driving at sharp turns.

So the challenge here is to balance the data so that the model sees all sitauations without bias.

My strategy was to use binning by allocating all the training examples into 200 bins ranging from -1.0 and +1.0, where -1.0 is lowest steering angle and +1.0 is the highest steering angle. 

	# create bins
	bins = np.linspace(-1.0, 1.0, 200)
	
	# allocate angles to each bin
	digitized = np.digitize(y_train, bins)
	
Then I take 100 examples from each bin. Here is a plot after binning the training examples.

![][image10]

The last step was to flip the images to augment the data. Here is the final distribution.

![][image11]

This approach proved to be very effective because now the model sees all driving siutations and not biased.

The downside of this strategy is that all training examples should be loaded into memory in order to apply the binning approach.

After I reached to a good balanced data I trained my comma.ai model.
I chose the comma.ai because it has been proven to work and also smaller comapring to the nvidia architecture.

After the model is trained, I used the simulator and observed the car's behaviour on the track.
For many times, the car struggled at the sharp left and right turns after the bridge. The problem is solved by collecting more training data at these two spots in particular by recovering the car.  Please see the animations below.

I repeated this process until the car is able to drive around the track.


####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers:

1. Conv layer 8x8 @ 16
2. Subsampling 4x4
3. ELU
4. Conv layer 5x5 @ 32
5. Subsampling 2x2
6. ELU
7. Conv layer 5x5 @ 64
8. Subsampling 2x2
9. ELU 
10. Flatten
11. ELU
12. 512 Dense
13. ELU
14. 1 Dense


####3. Creation of the Training Set & Training Process

##### A. Creating Training Set
My first step is to record a good 2 foward laps and 2 backword laps and also recorded some recovery data for sharp turns.

My code reads the three cameras from the driving_log.csv

![alt text][image2]

The total number of examples including all cameras is 21k.

##### B. Car Recovery 
The car struggled passing sharp turns after the bridge. So, I recorded the car recovering from the left side and right sides of the road back to center. These images show what a recovery looks like starting from edge of the road and steering back to the center:

![alt text][image3]
![alt text][image4]
![alt text][image5]

##### C. Preprocessing

All images were cropped from the bottom and the top leaving the middle part.



```sh
img = img[70:140,:,:]
```
![alt text][image6]
![alt text][image7]

After cropping, the images were resized to 16x64.


	new_col, new_row = 16, 64
	img = cv2.resize(img, (new_row, new_col), interpolation = cv2.INTER_AREA)


Then converted all images from RGB to HSV color space and extract the V channel for training.

    colors.rgb_to_hsv(img)
    img = img[:,:,2]



To augment the data sat, I also flipped the images and angles thinking that this prevents from overfitting and generlizes to second track.

	X_train = np.append(X_train,X_train[:,:,::-1],axis=0)
	y_train = np.append(y_train,-y_train,axis=0)


The final of total training examples after binning, preprocessing and augmenting is around 11k examples.


I set the `validation_split` parameter in the `model.fit` function to split the training into training and validation sets by `0.1`.


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by the plot below.

![Loss][image8]

The validation loss incremently decreased after each epoch until it reached the training loss at the 7th epoch.

---
### V Simulation

<<<<<<< HEAD


The car comfortably drives itself around the first track safely on the asphalt and goes around soft and sharp turns without touching edges or going to other unsafe surfaces.

<iframe width="560" height="315" src="https://www.youtube.com/embed/374smLAMcjc" frameborder="0" allowfullscreen></iframe>
=======
The car comfortably drives itself around the first track safely on the asphalt and goes around soft and sharp turns without touching edges or going to other unsafe surfaces.

>>>>>>> 5b5d3cc3dfc6b21d0fbba3b018de40cac35d59c9
