# Week 8 Task

Today we are continuing to use scikit learn for classification, however, this time around we will be using more than 
just accuracy to evaluate the model performance.

We are using the same libraries as last week:

```bash
pip install pandas scikit-learn
```

## Tasks

In a previous module, you may remember working with the famous Titanic dataset.
We are now going to use that same dataset to train a machine learning model to predict if somebody would survive based 
on:
* Class - First, second or third class passenger
* Sex
* Age
* Siblings/spouses - number of siblings and spouses joining them on the journey
* Parents/children - number of parents and children joining them on the jounrey
* Fare paid

You can find the dataset [here](https://github.com/darrened/526/tree/main/Week8/titanic_new.csv)

Note, I have made a few changes to this dataset to make it easier to work with.
Various columns are removed.
Sex is now 0 or 1 (with 0 being male and 1 being female)

1. Like last week, you need to train multiple models (using the various algorithms) on this dataset.
You should use the different metrics to see how each model performed and make a decision as to which one is preferable 
in this case. An example of a decision tree can be found [here](https://github.com/darrened/526/tree/main/Week8/dt.py)

   
2. Above is a binary classification, if you would like to see a multi-class confusion matrix then here is an example 
using the Iris dataset from last week: [click here](https://github.com/darrened/526/tree/main/Week8/iris.py)
