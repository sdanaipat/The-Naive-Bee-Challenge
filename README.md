# The Metis Challenge: Naive Bees Classifier

This notebook implements one of a solution to the Naive Bees Classifier challenge hosted by [drivendata.org](http://www.drivendata.org/competitions/8/).

## Overview of the solution
The idea is to fine-tune a pretrained VGG16 network with augmented data (flip and rotate) and a modified cross-entropy loss function (intended to fix the class imbalance problem).

## Requirements
This solution was implemented in Python (in iPython notebook) with theano, lasagne, and nolearn. To run this solution, you will need:

- ipython==3.2.0
- Lasagne==0.1
- matplotlib==1.4.3
- numpy==1.10.2
- pandas==0.16.2
- scikit-image==0.11.3
- scikit-learn==0.15.2
- scipy==0.16.1
- Theano==0.7.0
- nolearn==0.5

To install all the required packages at once, execute
```bash
pip install -r requirements.txt
```

### Data
Train and test data can be obtained from the [competition page](http://www.drivendata.org/competitions/8/) on drivendata.org. All preprocessing steps required for the solution to work have been described in the notebook.

### Training
The model usually takes 190-200 epochs to converge (~10 hours on a Titan X). I was able to achive the test score of 0.9934 on private leaderboard with this solution. Trained parameters of the model I used in the competition can be downloaded [here](https://googledrive.com/host/0B8YMpG0XziP0TzA4aGhEUHFkckE).
