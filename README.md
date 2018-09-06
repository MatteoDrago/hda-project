# hda-report
Human Data Analytics project - Matteo Drago, Riccardo Lincetto

## Deep Learning Techniques for Gersute Recognition: Dealing with Inactivity

To run the code download the OPPORTUNITY activity recognition dataset at:
https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition
The position of the dataset then has to be provided to the code in preprocessing phase.

The repository is organised as follows:
- code: this folder contains our code to perform activity recognition. There are two matlab files for preprocessing, we suggest using 'preprocessing_full.m' (otherwise the code has to be updated with the correct files location). Please note that 'file.root' variable needs to be provided the location of the dataset. Once preprocessing has been done, one can decide to run 'main.py' and 'main_multiuser.py' to get all the results at once (it takes some time, since 120 different models are trained), or to execute the code for a single configuraion in 'HAR_system.ipynb'. Then there is also a notebook with the purpose of visualising results, 'Evaluation.ipynb': to run this it is not necessary to run the complete code, because a set of results is already provided in the repository;
- presentation: this folder contains a set of slides that we used to present our project;
- report: this folder contains our report, named 'HDA_MDRL.pdf'.
