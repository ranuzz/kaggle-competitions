# README #


### What is this repository for? ###

This code base is for attempting kaggle competitions. The code is structured in a way that it can be used to analyse
other datasets as well. Modular approach has been used to share model architectures and ETL libraries across different datasets

### How do I get set up? ###

#### Getting code
* create [bitbucket](https://bitbucket.org/product) account to access the private repository
* git clone https://ranuzz@bitbucket.org/noobranu/kaggle-competitions.git
* or create a fork using bitbucket UI and clone your fork

#### Getting latest copy of the code (master branch)
* make sure that there are no changes done by you in local branch
* make sure you are in the master branch
> git branch <br />
> master should have a (*) next to it
* git pull
* git merge
* resolve conflicts if any

#### Contributing
* Never check-in to the master branch using : git push origin master
* create a new git branch
> git branch branchname
* switch to new branch
> git checkout branchname
* make sure current active branch is your new branch
* make changes
> git status : check your changes
* add all edited/created files
> git add -A
* commit locally
> git commit -m "message"
* switch to master and pull the latest changes
* switch back to branchname and merge with master
> git merge master : resolve conflict, if any, and commit again, if required
* push changes to your branch
> git push origin branchname
* goto bitbucket UI and check your push
* create push request

### Code structure ###

This is the directory structure of this code base
 ```
kaggle-competitions (BASE_DIR)
│   README.md
│   settings.py (all global setting)
|       |__LOG_DIR/DATA_DIR
|   requirements.txt (for pip)    
│
└───competitions
│   │
│   └───competitions folder
│       │   preprocess.py (download and prepares the data)
│       │   models.py (different models to applt)
│       │__ verification.py (data verifiacation)
│   
└───fetch
|    │   download code
|
|___utils
|   |__  AppsHelper.py (kaggle cli wrapper)
|    
```

#### creating development environment (Linux)
* install python 3.6+
* install virtualenv
> pip install virtualenv
* create a new virtual environment
> virtualenv dirname <br />
> or python -m virtualenv dirname
* activate the virtualenv
> source ./dirname/bin/activate
* locate synced code and requirements.txt file and install all required packages
> pip install -r requirements.txt
* test the setup
> python competition/tensorflow_mnist/mnist.py
* verify that training was complete and the script has creted **MNIST-data** and **tensorflow_models** directories have been created.
* verify tensorboard setup
> tensorboard --logdir=tensorflow_models


#### Setup kaggle-cli
* install kaggle cli
* generate API key (kaggle.com)
* set-up key for kaggler cli

#### Running code
* go to imaterialist directory
* read and understand the code after referring to MNIST code
* open preprocess.py and adjust the directories if required, increase number (nprocs), if possible
* run preprocess.py
> python preprocess.py
* in case if error try adding PYTHONPATH
> PYTHONPATH=/[kaggle-competiion-dir] python preprocess.py
* run verification.py to make sure that all files are present
* use jupyter notebook to get a feel of data
> jupyter notebook <br/>
> locate scratchpad.ipynb
* run basic_model.py for training and validation, adjust num_epochs and steps as necessary
> python basic_model.py
* run basic.py in test mode to generate kaggle-submission
> python basic_model.py 1
* locate the csv file in kaggle_submission

### Who do I talk to? ###

* ranu