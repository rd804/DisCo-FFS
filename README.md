# DisCo-FFS
Feature Selection using Distance Correlation (arXiv:2212.00046v1)

# Environments

Includes 2 environment files for the different servers used in the pipeline.
After activating environments, run: 

`python -m pip install -e .`

# Data

Default data locations:
the scripts use the labels from `y_[train,val,test].txt` in `./data`
the efps are stored in `./data/tops/efp/[train,val,test]`
the initial features m,pt,mw are stored in `./data/tops/initial/`
The features are loaded using `feature_loader` class defined in `src/feature_loader.py`

# Pipeline
The full pipeline includes submission to condor, for computing scores, and submitting to slurm job scheduler for training variance. 

* Run full pipeline: `nohup bash selection_variance.sh &>./logs/pipeline_log.out &`

* Inside `selection_variance.sh` the scripts used:
  * `scripts/create_training_data.py` : creates .npy files with already selected/initial features
  * `scripts/classifier_training.py` : trains classifier to obtain classifier output
  * `scripts/compute_scores,py` : computes disco score using the confusion set using condor
  * `scripts/sort_and_add_new_feature.py` : finds feature with the highest score, and adds to the list of known features
  * `scripts/training_variance.py` : script submitted to slurm, used to train on a set of features 10 times

* `job_scripts/training_variance.sh` is used to submit the slurm submission script `job_scripts/job_array_training.sh`
* `job_scripts/compute_scores.jdl.base` is used to submit the condor submission script `job_scripts/compute_scores.sh`




  
