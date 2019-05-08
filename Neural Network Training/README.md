**Training, Validating, Evaluating, Debugging and Testing a Neural network Model:** 
 
1.	Make sure the dataset is in proper format. Divide the dataset into training, validation and test set. Start with the ratios 80:10:10 then proceed to 90:5:5 and finally to 98:1:1.  Make sure the model we are training, we should pre-process the inputs if necessary. 

2.	**Training without Validation:** We have to make sure that the model first trains correctly rather than validating it. Doing training and validation side by side at the beginning is a misstep. To start the training, please see the below points. 

3.	**The Golden Rule:** First rule is to check whether our model is structured properly, meaning our model works if at all. To check that, we train it on a single data point (single image). Now if our model is correct and sensible no matter how inefficient it is, it should at least overfit that single data point without taking much time. If not, proceed to step 5 else proceed to step 4. 

4.	Train the model for 5 images and then proceed to 10 images without any validation. In both these cases, the model should overfit. It should train these small datasets. 
If no: We have to try to debug the network. Please see the below point for that. 
If yes: Skip to point 7. 

5.	**Debugging a Neural Network:** If these steps don’t work after point 6, then scroll down and follow more steps to debugging – point 8 or so. 
  
    •	Is the input data not making sense? Careless mistakes like replacing height with width and repeating same batches repeatedly will       make the network model to not learn. 

    •	**Try passing some random value:** If the error behaves the same way, then it is likely that the model is turning these random  values to garbage at some point. Now this problem is very difficult to eliminate. We need to debug layer by layer or module by module. Please see the below point 6 for that. 

    •	**Check data loader/generator:** We need to check that this code is actually loading the inputs to the network ResNet. Try printing the input to the ResNet to see if it is making any sense. 

    •	Is the setup of modules okay? Please check the module code once again to see that they are actually connected with each other and are making sense. 

    •	**Buggy Training Algorithm:** Is there in any calculations operations like division by 0 or logarithm of negative numbers? Eliminate it. Are the dataset pixel values integers instead of float32? Please change it. 

    •	**Check the initial loss:** If it is very large, then this probably means that the weight initialization is bad. Change the weight initialization. If unsure, set it to Xavier or He Initialization. 

    •	Is backpropagation working? To check if it is, implement the gradient checking algorithm. We can also output gradients using keras backend and see if it matches. 

    •	**Try to change hyperparameters:** Try grid search algorithm or manually tune it. 

    •	Did you standardize the features? Centred towards zero mean and unit variance. Please fix if not. 

    •	Check the learning rate. Start high and go low. Follow exponential decrease with each epoch or use KerasReduceLRonPlateau callback to monitor training loss. 

    •	**Try switching the optimizers:** Adam is fast but it’s generalization error is higher than SGD with momentum. We will have to trial and error with these optimizers and generally either of these two can do the needful. So start with these two first. Then move to RMSProp. 

    •	Shuffle the dataset. Remember, shuffle outputs in same order as the 	inputs. 
    Use train_test_split with shuffle=True or zip function. 

    •	Batch Normalization makes the training really fast. Please try to add it after each layer. 

    •	Again, check the consistency of dimensions for each module. Keras model.summary() should help in this regard. 
  
6.	**Debugging Tips:** 
  
    •	If there is memory crash due to large data, we can use tensorflow TfRecords or even read the data in batches if using Keras by creating a separate function/loader. 

    •	Try to print images being read or output after each module using print(x.eval(sess..)) if in tensorflow. 

    •	Try printing out the statistics of the tensors. 

    •	Wrapping tensorflow in keras is a friendly idea using engineered_features in Keras. 

    •	Always try to print outputs at each layer using keras.layers.lambda. 

    •	Add tf.print and tf.summary wherever possible to keep track. 

    •	Use 	Tensorflow 	debugger 
    (https://github.com/Createdd/Writing/blob/master/2018/articles /DebugTFBasics.md#5-use-the-tensorflow-debugger). 

    •	Use keras backend to print values after each layer: 

    •	Consider a visualization library like Tensorboard and Crayon. In a pinch, we can also display weights/biases/activations. 

    •	Layer updates should have approximately Gaussian distribution. 

    •	Try to follow this rule: For weights, the histograms should have an approximately Gaussian (normal) distribution, after some time. For biases, these histograms will generally start at 0, and will usually end up being approximately Gaussian. 
  
7.	**Try out the opposite golden rule:** Keep the full training set along with the validation set this time. Now just shuffle the outputs only (of training not validation) and then train the model on this set. We should have training loss decreasing very slowly and validation loss extremely high and random. If this does not happen, repeat steps 5,6. If still not, move to more debugging steps – step 8. If this does happen, skip to step 9. 
  
8.	**Debugging a Neural Network II:** If this does not work, either move to step 9, or try these steps 5,6,8 again and look more closely. If they work, go to 9. 
  
    •	**Unit Testing:** Again, please break the code into modules and check again if it gives expected output. I will follow this guide: https://medium.com/@keeper6928/how-to-unit-testmachine-learning-code-57cf6fd81765 

    •	We can follow this library for unit testing but it is only in TensorFlow for now: https://github.com/Thenerdstation/mltest 

    •	**Eliminate Exploding/Vanishing Gradients:** To overcome exploding gradients, we can use gradient clipping or we can decrease learning rate. For latter case, we can use Relu or Leaky Relu or we can increase learning rate. 

    •	**Overcoming NaNs:** Try to decrease learning rate, or check for any calculation errors. Finally, we need to follow step 6 to debug layer by layer. 
  
9.	Take the whole training set this time along with the validation. Make sure that they are shuffled correctly and pre-processing is done on all the sets in the same manner. For now, do not use callbacks/checkpoints. Now train the model on the dataset. We should have either of these cases: 
Training loss is not decreasing: This is clearly the case of underfitting. Go to step 10 to eliminate this problem. 
Training loss is low but the validation loss is high/random: This is clearly the case of overfitting. Skip to step 11 to eliminate this problem. If no issues: Skip to step 12. 
  
10.	**Debugging a Neural Network III – eliminate Underfitting:** 
  
    •	Is the dataset imbalanced? If yes, then replace the current loss function with the weighted loss function. Then run the training process all over again. 

    •	Are training examples sufficient? Try to change 80:10:10 ratio to 90:5:5 and finally to 98:1:1 and see if it improves anything. 

    •	**Reduce batch size:** This can generalize the model better but then there will be risk of overfitting. 

    •	Reduce too much Data Augmentation: See if it helps in any way. 

    •	**Make the model complex:** This we can do by increasing the number of hidden layers and test it on each module (see module based debugging). Try toincrease number of hidden units as well although former is more effective. 

    •	**Reduce Regularization if it is too much:** It decreases overfitting but too much can cause underfitting. Try to reduce dropout, L1/L2 regularization. 

    •	Verify the metric as well as loss function and see if it makes any sense. 

    •	Eliminate the ‘dying relu’ problem by replacing it with ‘leaky relu’ or Prelu. 

    •	Manually stop the training, change the learning rate, and then continue it just to avoid local minima. 
  
11.	**Debugging a Neural Network IV – eliminate Overfitting:** 
  
    •	**Increase batch size:** Less generalization. 

    •	**Increase data augmentation:** It has regularization effect. 

    •	**Increase dropout/L1/L2 regularization:** Again, it prevents weights and biases from behaving erratically. 

    •	**Make the model simpler:** Remove some layers and hidden units. 

    •	Debug layer by layer to check the behaviour of weights and biases and then manually tune them (most difficult). 

  
12.	**Final Tips:** Finally train and validate the model on the whole dataset and making use of callbacks such as early stopping, model checkpoint, Reduce LR on Plateau by monitoring validation loss. We can use a cyclic learning rate, which is said to be the best scheduler. 
  
13.	If none of the above steps work, try the above debugging steps once again or ask doubts on stackoverflow xD. Get further help or search for the help on communities like stackoverflow or github. 
  
14.	Try to push the metric of training set and the validation set. 
  
15.	Now run the model on the testing set and see if it predicts properly. The prediction accuracy should be close to validation set as of now. It should. 
