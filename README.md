# Final Project Group
Class Project

Data Download Scripts:	
To download the images from one drive and google drive using CURL.

DataProcessing.py:		
To split the downloaded images into train-validation-holdout and organize the files for ImageFolder.

MaskDetector_Final.py:	
Training of the final resnet50 model.

Inference.py:			
To view the inference probability of each class based on input image passed.

OcclusionExperiment.py:	
To intrepret and visualize the CNN based on occlusion experiment

Order to run the files:
1. Download Dataset from Kaggle: https://www.kaggle.com/sheldonsebastian/maskednet-flicker-faces
2. Run MaskDetector_Final.py to create Model
3. Inference.py
4. OcclusionExperiment.py
