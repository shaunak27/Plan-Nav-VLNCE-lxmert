# VLN-CE Hierarchical Planner

## Installation

Follow the instructions in [VLN-CE](https://github.com/jacobkrantz/VLN-CE)
- This code assumes that the scene data is in:
``` 
    data/datasets
    data/scene_datasets
```

## Datasets 

All datasets used are [here](https://drive.google.com/drive/folders/14w0surrcop_0s0yA3_XUgm1SgwTqEiBy).

## Models 

All models used are [here](https://drive.google.com/drive/folders/1gEzHLehaHyJYn8wX9uTc4gO4PPNtf3Zs).


## Tests 

NOTE: for all tests, the YAML file indicates the output directory where the data was saved. 

### Feature Extractor
To test the feature extractor model, run:
``` 
   cd test
   python feature_extractor.py --exp-config feature_extractor.yaml
```
Verify the models load correctly and the feature extractor successfully extracts the features from the panorama. 

### Candidate Proposal
To test the candidate proposal model, run:
``` 
   cd test
   python candidate_proposal.py --exp-config candidate_proposal.yaml
```
Verify the models load correctly and the candidate proposal generates an occupancy map and the candidate waypoints. Outputs should be generated to ``` data/out/candidate_proposal```.

### Projection Module
To test the projection module , run:
``` 
   cd test
   python backward_projection.py --exp-config backward_projection.yaml
```
Verify the models load correctly and the candidate proposal generates an occupancy map and the candidate waypoints. Outputs should be generated to ``` data/out/backward_projection```.


### Global Planner 
To test the global planner, run:
``` 
   cd test
   python global_planner.py --exp-config global_planer.yaml
```
Verify the planner generates the panoramas and predicts a heading and distance. Outputs should be generated to ``` data/out/global_planner```.
 

### Local Planner 
To test the local planner, run:
``` 
   cd test
   python local_planner.py --exp-config local_planner.yaml
```
Verify the planner generates a local path. Outputs should be generated to ``` data/out/local_planner```.

## Trainers

### Local Planner Trainer

To train or evaluate the local planner run:
``` 
   export MAGNUM_LOG=quiet
   export GLOG_minloglevel=2
   python run.py --run-type [train | eval] --exp-config vlnce_baselines/config/trainers/local_ppo.yaml
```
The YAML file specifies all the information required to train the agent (dataset and model locations, training parameters, etc...).

### Global Planner Trainer

To train or evaluate the global planner using DAgger and Seq2Seq, run:
``` 
   export MAGNUM_LOG=quiet
   export GLOG_minloglevel=2
   python run.py --run-type [train | eval] --exp-config vlnce_baselines/config/trainers/global_dagger_seq2seq.yaml
```

To train or evaluate the global planner using DAgger and CMA, run:
``` 
   export MAGNUM_LOG=quiet
   export GLOG_minloglevel=2
   python run.py --run-type [train | eval] --exp-config vlnce_baselines/config/trainers/global_dagger_cma.yaml
```
The YAML file specifies all the information required to train the agent (dataset and model locations, training parameters, etc...). 
