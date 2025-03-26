# Week 10

In this session, we will simply put the last piece in to the puzzle - using the models we have trained

At the moment, every time you run your program, you are training up your ML model from scratch, which isn't the idea.  
As you may imagine, we just want to train the model once and then make use of that trained model on our software.

## Using a Model

Making use of a model is simple, you've already done it when testing! Take the decision tree below:

```python
# Imports go here

# Load and split the data here 

dt = DecisionTreeClassifier()
dt_model = dt.fit(X_train, y_train)

# Test the model against the test data
dt_pred = dt_model.predict(X_test)
```
That "predict" method on the last line is us using the trained model to make predictions.

So, going back to the [titanic_new](https://github.com/darrened/526/tree/main/Week9/titanic_new.csv) dataset. I might 
want to make a new prediction, such as:

|Pclass|Sex|Age|SibSp|Parch|Fare|
|------|---|---|-----|-----|----|
|2|1|29|0|1|16.0|

All you have to do is put these values in a list, observing the correct order of the dataset, and then you can 
make a prediction. Note multiple predictions can be made, so this is actually a 2D list:

```python
values = [[2, 1, 29, 0, 1, 16.0]]
```

From here we can use this list to make a prediction:

```python
dt_pred = dt_model.predict(values)
print(dt_pred)
```

Output:
```
[1]
```

It is worth noting that since we trained the dataset using data from pandas, SKLearn does not appreciate being presented
with unlabelled columns. If you receive a warning for this, the fix is simple - present the data as a pandas dataframe:

```python
values = [[2, 1, 29, 0, 1, 16.0]]
new_df = pd.DataFrame(values, columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])
dt_pred = dt_model.predict(new_df)
print(dt_pred)
```

Output:
```
[1]
```

## Saving and Loading a Model

Now on to the important bit. How do you make it so that you do not have to train the model each time you run the 
software?
The answer is, you save the trained model to your file system.

We can do this using pickle:

```python
import pickle
# Other imports here

# Load and split the data here 

dt = DecisionTreeClassifier()
dt_model = dt.fit(X_train, y_train)

# Test the model here

with open("titanic_model.pkl", "wb") as file:
    pickle.dump(dt_model, file)
```
This will save our trained model to the file system.

Now when we want to use the model in our software, we can load it in:

```python
import pickle
import pandas as pd
# SKLearn imports also go here

with open("titanic_model.pkl", "rb") as file:
    t_model = pickle.load(file)

values = [[2, 1, 29, 0, 1, 16.0]]
new_df = pd.DataFrame(values, columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])
dt_pred = t_model.predict(new_df)
print(dt_pred)
```

## Task

Now you've seen how to save a ready trained model and then load it in.

Your task is a follow-on from last week:
1. Using the training/testing techniques and the titanic dataset from previous weeks, train up the best model you can.
2. Save the best performing model using pickle
3. Now create a completely separate program that:
   1. Loads in the model
   2. Takes in the data it needs via user input
   3. Uses said input to make a prediction and display the result in a user friendly way

The above program should have a simple menu system and run until the user chooses 'quit':

```
Please select an option:
A: Make prediction
Q: Quit
```

### Bonus Task
If you want a bigger challenge, look in to the 'TKInter' library 

Using the TKInter library, create a simple GUI for this system that allows the user to input data (via text boxes) and 
then present the result to the user.