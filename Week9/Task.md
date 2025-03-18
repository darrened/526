# Week 9 Task

This session follows on from the previous session.

Before, we have been training ML models and we have also looked at evaluating them. Now lets put some more pieces in 
place

## Pre-processing

An example of both scaling and PCA can be found [here](https://github.com/darrened/526/tree/main/Week9/pca_demo.py)

This time around I am giving you the 
original [Titanic dataset](https://github.com/darrened/526/tree/main/Week9/titanic.csv)

1. Look at this dataset and consider what features you do not want. Drop unwanted columns - ensure you have pandas 
imported
```python
pd.drop(["col1", "col2"], axis=1)
```

   
2. From what's left, scale the features

3. Train various models with and without scaling and see if they perform better
4. Consider using PCA on some (or all) columns and again train to see if this has any impact.

## HyperParameters

1. Train your dataset using grid search and K-fold. Which hyper-parameters work best for each algorithm?
An example can be found [here](https://github.com/darrened/526/tree/main/Week9/k_fold_grid.py)
