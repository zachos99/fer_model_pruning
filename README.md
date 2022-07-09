# Facial Expression Recognition using DAN models 
## Project purposes and details
This repository was created to present the results of a project. In this project we used the [DAN repository](https://github.com/yaoing/dan) and its pre-trained models. You can get the models from that repository. See also the [paper](https://paperswithcode.com/paper/distract-your-attention-multi-head-cross).

The purpose of this project was to prune the AffectNet-8 pre-trained model and evaluate depending on the percentage of pruning. We wanted to make the model lighter without losing much accuracy. The evaluation was implemented on [AffectNet-HQ](https://www.kaggle.com/datasets/tom99763/affectnethq) dataset, part of the [AffectNet dataset](https://paperswithcode.com/dataset/affectnet), a state-of-the-art database for Facial Expression. The AffectNet-HQ dataset was nevertheless not properly separated into folders, so, after downloading, we made the [fix_dataset.py](https://github.com/zachos99/fer_model_pruning/blob/main/fix_dataset.py) which rearrange images according to the labels.csv file.

## Demo
You can run the [demo_cam.py](https://github.com/zachos99/fer_model_pruning/blob/main/demo_cam.py) to detect real-time emotion from user's camera.

You can also run [multiple_images_clasification.py](https://github.com/zachos99/fer_model_pruning/blob/main/multiple_images_clasification.py) to classify multiple images into classes-emotions. You can set the path of a folder and then save the labels to a .csv file. 


## Pruning and Evaluation
We created the [pruning.py](https://github.com/zachos99/fer_model_pruning/blob/main/pruning.py) which is used for model pruning. You set the path of an existing model, the method and the percentage of pruning and then you save the pruned model. Then you can evaluate the pruned model with [model_evaluation.py](https://github.com/zachos99/fer_model_pruning/blob/main/model_evaluation.py) and calculate its accuracy. There are instructions inside the file.

## Testing and results
We used these scripts also to make some tests. We pruned the AffectNet-8 pre-trained model for different percentages and calculated the accuracies. Our goal was to make the model lighter so we checked the required resources. We calculated the accuracy, the time needed for the whole dataset, the GPU-memory usage and the RAM usage. The results are below:
![alt text](https://github.com/zachos99/fer_model_pruning/blob/main/pruned_models_comparison.png?raw=true)

For pruning percentage >30% the model was destroyed. The tests were made in Ubuntu 20.04 with CUDA Version: 11.6.

With minor changes we can evaluate on different datasets or prune different models or use different pruning methods.
