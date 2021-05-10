## README:

### Final Report:
https://sheldonsebastian.github.io/face_mask_detector/

### Directory Structure:

| path | Description |
|------|-------------|
|Code/Data Download Scripts| Folder containing scripts to download the images from one drive and google drive using CURL.|
|Code/DataPreprocessing.py| To split the downloaded images into train-validation-holdout and organize the files for ImageFolder.|
|Code/Train.py| Training of the final resnet50 model.  |
|Code/Inference.py| To view the inference probability of each class based on input image passed. |
|Code/OcclusionExperiment.py| To intrepret and visualize the CNN based on occlusion experiment |
|docs/| Files related to github website |
|saved_model| Folder containing the trained model |
|Final-Group-Presentation.pdf| presentation slides as pdf |
|Final-Group-Project-Report.pdf| final report as pdf |

### Steps to replicate project:

1. Download data using Data Download Scripts
2. Train model using Train.py
3. Perform Inference using Inference.py
4. To interpret the model use OcclusionExperiment.py
