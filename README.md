# RNN as models of human cognition in mental health and disorder

## Objective:
1. Analyze individual differences of longitudinal smartphone based data (multi-task cognitive behavioral battery) focusing on performance, reaction time and errors 
2. Train individual RNNs on datasets and find optimal hyperparameter sets (grid search)
3. Evaluate internal representations in terms of robustness, topological structure and representational similarity over training and data collection time
4. Investigate sensible environmental factors (task complexity, task quantity, model hyperparameters) on internal representations
5. Compare individuals within and between, derive mechanistic understanding of computations and highlight differences between cognitive profiles
6. Compare the internal representational metrics to brain embedded graphs (fMRI based)

## Hierarchy of actions
1. preprocessing.py & dataAugmentation.py for data preperation
2. training.py & hyperparameterTuning.py for training models
3. hyperparameterOverview.py & hyperparameterOverview_overlays.py for performance and topological marker visualization
4. singleNetworkAnalysis.py & multipleNetworkAnalysis.py - analyzing models hidden structure and task representation + saving representational network metrics
5. cognitiveErrorAnalysis.py & cognitiveTaskAccuracy.py - analyzing individual cognitive behavior

## Docker Flow
- Dockerfile defines dependencies
- docker-build.yaml builds the .dockerignore reduced image whenever repo is pushed to github
- Latest image is pushed to docker hub (actions with secrets)
- From VM manually pull image from hub
- Initalize container with fine-tuned hps defined by config files w. different shell scripts


