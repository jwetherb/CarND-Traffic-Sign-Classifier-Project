# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/DataVis.png "Visualization"
[image2rgb]: ./examples/RGBSamples.png "RGB (Original)"
[image2]: ./examples/GrayscaleSamples.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./samples_from_web/30km.png "Speed limit (30km/h)"
[image5]: ./samples_from_web/70km.png "Speed limit (70km/h)"
[image6]: ./samples_from_web/Children_Crossing_32.png "Children crossing"
[image7]: ./samples_from_web/Men_Working.png "Road work"
[image8]: ./samples_from_web/Stop.png "Stop"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! And here is a link to a [jupyter project with all of my project code](https://github.com/jwetherb/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_Final.ipynb) along with an [html download of my jupiter project](http://htmlpreview.github.io/?https://github.com/jwetherb/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_Final.html), both hosted in my GitHub repo.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799** images
* The size of the validation set is **4410** images
* The size of test set is **12630** images
* The shape of a traffic sign image is **(32, 32, 3)** (Note that after the images were reduced to grayscale, the shape became **(32, 32, 1)**)
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the input data set sizes for the Training, Validation, and Testing stages:

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. 

[What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)]

### Reduced RGB images to Grayscale

As a first step, I decided to convert the images to grayscale because, after comparing my model's results when using RGB vs. grayscale images, I found the learning was largely unaffected, and was significantly faster. Color as an input factor was likely both an aid and a hindrance in this case, and apparently the net result was not significant.

Here are three examples of Training traffic sign images, before and after grayscaling:

Original (RGB) images:

![Original (RGB) images][image2rgb]

And after gray scaling:

![Grayscaled images][image2]

### Data normalizing

As a second step, I normalized the image data because resetting the values around a 0 axis and using a range of -1..0..1 generally performs best in the algorithms used by this model.

### Data rebalancing

I decided to generate additional data because I found certain traffic signs were significantly overrepresented in the Training set. 

To add more data to the the data set, I used the following techniques to ensure that all traffic signs were equally represented in the Training set:

1) I calculated the number of times each traffic sign was shown in the Training set, and kept track of the sign that was represented the most

2) I doubled the most-represented sign's training set size to produce a target number for all signs to be represented (equally)

3) I then duplicated the Training set for each sign as many times as necessary to approximate this target number, and added each duplicate to the Training set

The result is that, following this rebalancing process, all signs were more or less equally represented in the final Training set.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|Layer|Operation|Description| 
|-----|---------|-----------| 
| 1 | Input         			| 32x32x1 grayscale image  					| 
| | Convolution 			| 1x1 stride, same padding, Output = 28x28x6. 		|
| | RELU				|								|
| | Dropout				| Keep rate = 0.75.						|
| | Max pooling 			| 2x2 stride, Input = 28x28x6. Output = 14x14x6. 	|
| 2 | Convolution 			| 1x1 stride, same padding, outputs 32x32x64 		|
| | RELU				|								|
| | Dropout				| Keep rate = 0.75.						|
| | Max pooling	      		| 2x2 stride, Input = 10x10x16. Output = 5x5x16. 	|
| | Flatten				| Input = 5x5x16. Output = 400.				|
| 3 | Fully connected		| Input = 400. Output = 200.        			|
| | RELU				|								|
| | Dropout				| Keep rate = 0.75.						|
| 4 | Fully connected		| Input = 200. Output = 120.        			|
| | RELU				|								|
| | Dropout				| Keep rate = 0.75.						|
| 5 | Fully connected		| Input = 120. Output = 84.        			|
| | RELU				|								|
| | Dropout				| Keep rate = 0.75.						|
| 6 | Fully connected (Logits)	| Input = 84. Output = 43.        				|
<br>
My model was based on the LeNet model. To improve upon the default behavior, I added Dropout steps at each of the hidden layers, and also added an extra Fully Connected layer.

The model began with a forward pass using the LeNet-based model described above. I passed the X_train input into a Convolution layer that produced output which I sent to a RELU activation layer, then to a Dropout step which arbitrarily removed 1/4 of the results, and then averaged out 2x2 sections of the resulting matrix using a Max pooling step.

I followed this procedure again, sending the result of the previous layer to a second Convolution layer with the same activation and refinement steps.

At this point I flattened the data to produce a single array of 400 intermediate output values. From this point on, I processed the data through a sequence of three Fully Connected layers, again employing RELU activation and refining the results through Dropout.

The results of the last hidden layer were sent to a final Fully Connected layer that produced the model output. This output was an array of 43 values, matching the 43 distinct signals we were teaching the model.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I experimented with a variety of learning rates, but found that the LeNet default hyperparameter learning rate of **.001** was the most effective value.

The learning steps I used can be summarized at each stage through these TensorFlow operations:

| Output | Description |
|----------|---------------------------------------------------------|
| logits | LeNet(x) |
| cross_entropy | tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits) |
| loss_operation | tf.reduce_mean(cross_entropy) |
| optimizer | tf.train.AdamOptimizer(learning_rate = rate) |
| training_operation | optimizer.minimize(loss_operation) |  
<br>
The results of the forward pass described in the Model section above were 43 values, with the desired result being the single index in this array that was associated with the input image holding the highest value in the array.

The learning stage involved taking these actual outputs from the forward pass, comparing them with the desired output, and then updating the weights in the model so that the next time that image was presented to the model, it would produce output that more closely matched the desired output. This process updating the weights in the model is called back propagation, and uses the principle of gradient descent to tweak the weights, moving one layer at a time backward through the model.

After calculating the loss (the difference between the actual output and the expected output), I used the TensorFlow AdamOptimizer to derive the amount by which to adjust ('teach') the weights at each layer. This process effectively updated the weights by a fixed, minute amount at each layer. The optimizer was regulated by the learning rate hyperparameter, which I set to the LeNet default of 0.001. I experimented with slightly higher and lower values, but this default seemed to work the best overall. The weights were updated at each layer by the tf.reduce_mean() operation which operated on the results of the tf.nn.softmax_cross_entropy_with_logits() function.

The training process at a macro level involved taking the training images, batching them up into sets of 128 images, sending each batch through the model and then employing the learning back pass to update the weights in the model. A single pass through the entire input training set is known as an EPOCH, and in my training phase I underwent 10 epochs, so the model was taught the entire training set 10 times. In my case, because I balanced the data and so a lot of underrepresented traffic sign images were duplicated, many images in the training set were taught to the model many more than 10 times.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Beginning with a straight LeNet model, I added several refinements:

1. Converted the RGB images in the data sets to grayscale
2. Rebalanced the data in the training set, duplicating under-represented signals so that all traffic sign images were more-or-less evenly represented
3. Added Pooling to each the Convolutional layers in the model
4. Added Dropout after each hidden layer in the model
5. Experimented with tweaking the learning rate (though eventually settled back on the initial value of 0.001)
6. Introduced a new Fully Connected layer in the model, to perform a more gradual step-down of the initially flattened array of width 400 to the even eventual output width of 43

Initially my model performed at about 88-89% accuracy on the validation set. The addition of each of the above steps increased the model to the current 95-96% validation set accuracy.

My final model results were:
* training set accuracy of ? (I didn't check the validation of my training set. (That would have been a somewhat disingenuous validation test, since the training data was used during the learning process.) Instead I tested it on the validation and test sets, which were never "taught" to the model.)
* validation set accuracy of 96.3 (in final epoch)%
* test set accuracy of 93.9%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I began with a straight LeNet model, adjusting the input shape to reflect my switch to grayscale images, and my output shape to match the # of classes in the input data.

* What were some problems with the initial architecture?

The principal problem with the initial architecture was the training data did not balance the representation of each signal very evenly. Adding this balancing step alone improved accuracy in my model by 3-4%.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I did not attempt to use a wholly different model architecture from LeNet, but I did observe that the RGB image data did not perform quite as well, the training data did not balance the representation of each signal very evenly, it was missing the valuable Dropout and Pooling adjustments during the forward pass, and it was in need of an additional layer in the model.

* Which parameters were tuned? How were they adjusted and why?

I converted all images in the input sets to grayscale. I also experimented with adjusting the learning rate, though I eventually settled back on the default value of 0.001 because it appeared to perform best.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Convolution was essential for its ability to identify sub-features. The first convolution layer was able to decompose the images into primitive features like lines and curves, and the second convolution layer was instrumental in assembling these into intermediate structures like numbers and shapes. I experimented with introducing a third convolution layer, but this did not appear to improve accuracy.

I also employed Dropout to avoid overfitting, and Pooling to generalize feature detection. Both improved accuracy of my model.

If a well known architecture was chosen:
* What architecture was chosen?

My model is based on the LeNet architecture.

* Why did you believe it would be relevant to the traffic sign application?

LeNet uses both Convolutional and Fully Connected layers, and flattening, offering feature detection and high-performance learning. It was also easy to introduce Pooling and Dropout as beneficial refinements.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I validated the model at the end of each training epoch using the validation set, and accuracy was seen improving with each epoch. This demonstrated the model learning gradually. After 10 epochs or so, the model learning appeared to slow, indicating it was experiencing some over-fitting. But the overall test set validation accuracy of 94% indicates that the model was well trained.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I initially cropped all of my images so that the sign took up the entire frame, and that turned out to be a mistake, since the signs generally appeared smaller (farther away) in the training images. My model had learned to detect features of a certain size, and it appeared to be less good at detecting those same features when they took up a larger portion of the frame. After I re-cropped my images to be similar in scale to the training images, accuracy results improved.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image | Prediction | 
|------------------------|---------------------------------------------| 
| Speed limit (30km/h) | Speed limit (30km/h) | 
| Speed limit (70km/h) | Speed limit (70km/h) |
| Children crossing | Children crossing |
| Road work | Road work |
| Stop | Stop |
<br>
The model was able to correctly guess all 5 traffic signs, an accuracy of 100%. (I ran this step numerous times, with different tweaks to the model, and before re-cropping my web samples the model guessed only between 1 and 3 signs correctly). This was quite favorable to the test set validation, which showed only 93.9% accuracy.

I also believe both my model accuracy in general, and specifically the accuracy of the test images, could have been improved further by pre-processing all images to enhance line detection.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
~~~~
TopKV2(values=array([[  9.98715878e-01,   1.01069361e-03,   1.90324965e-04,
          5.39426364e-05,   1.54346162e-05],
       [  9.99998450e-01,   1.51756456e-06,   1.51179513e-08,
          8.95066776e-09,   2.51880765e-13],
       [  9.99764025e-01,   2.33762796e-04,   2.06936124e-06,
          1.04471880e-07,   6.36141309e-08],
       [  9.99274909e-01,   3.66700493e-04,   1.46640057e-04,
          9.49749228e-05,   3.22316300e-05],
       [  9.99964356e-01,   1.68471415e-05,   1.28504826e-05,
          4.10028315e-06,   1.13315980e-06]], dtype=float32), indices=array([[ 1,  2,  0,  5,  4],
       [ 4, 18,  1,  0, 26],
       [28, 30, 11, 23,  3],
       [25, 22, 20, 30, 26],
       [14,  1, 38,  3, 33]], dtype=int32))
Expected values were: [ 1  4 28 25 14]
~~~~
For all images, the model's accuracy in predicting the correct traffic sign was 99.87% or higher.

Image 1 (Speed limit (30km/h)):

| Probability |     Prediction | 
|---------------------|---------------------------------------------| 
| .9987 | Speed limit (30km/h) | 
| .0010 | Speed limit (50km/h) |
| .0002 | Speed limit (20km/h) |
| .0001 | Speed limit (80km/h) |
| .0000 | Speed limit (70km/h) |


Image 2 (Speed limit (70km/h)):

| Probability |     Prediction | 
|---------------------|---------------------------------------------| 
| 1.0000 | Speed limit (70km/h) | 
| .0000 | General caution |
| .0000 | Speed limit (30km/h) |
| .0000 | Speed limit (20km/h) |
| .0000 | Traffic signals |

Image 3 (Children crossing):

| Probability |     Prediction | 
|---------------------|---------------------------------------------| 
| .9998 | Children crossing | 
| .0002 | Beware of ice/snow |
| .0000 | Right-of-way at the next intersection |
| .0000 | Slippery road |
| .0000 | Speed limit (60km/h) |

Image 4 (Road work):

| Probability |     Prediction | 
|---------------------|---------------------------------------------| 
| .9993 | Road work | 
| .0004 | Bumpy road |
| .0001 | Dangerous curve to the right |
| .0001 | Beware of ice/snow |
| .0000 | Traffic signals |

Image 5 (Stop):

| Probability |     Prediction | 
|---------------------|---------------------------------------------| 
| 1.0000 | Stop | 
| .0000 | Speed limit (30km/h) |
| .0000 | Keep right |
| .0000 | Speed limit (60km/h) |
| .0000 | Turn right ahead |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I simply ran out of time working this project and wasn't able to pursue this step. I would be very interested in viewing what each layer "learns", and how these steps converge on accurately recognizing each pattern as a whole...
