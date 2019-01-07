# ML Course Notes

## Framing

- **Supervised Machine Learning:** ML systems learn how to combine inputs to produce useful predictions on never before seen data.

- **Label:**  The thing we're predicting, *y* variable in simple linear regression. ( eg: future price gold, animal type in picture ).

- **Feature:** The input variable, *x* variable in linear regression. Simple ML system might use a single feature, while more sophisticated systems could use millions of features.

          x1, x2, ...xN

          eg: Email spam detector ML system
          features: - Words in email text
                    - Senders address

- **Example:** Example is particular instance of  the data, **x**. **x** is vector.
  - Labeled Examples: {features, label}: (x, y)
  - Unlabeled Examples: {features, ?}: (x, ?)

  - You train model using labeled examples, then you use the model to predict labels on unlabled examples.

- **Models**: Model defines relationship between features and label.
  - **Training** means creating or learning the model. Show the model label examples so it gradually learns relationship between features and label.
  - **Inference** means applying trained model to unlabled examples. That is, you use the trained model to make useful predictions (y').


- **Regression vs Classification**
  - Regression models predict continuous values
  ( eg: average house price Dublin ).
  - Classification models predict discrete values
  ( eg: Is email spam or not spam)

## Liner Regression

- **Overview:**
  - Drawing a line of best fit between two variables gives a relationship. ***y*** axis  the thing we want to predict, ***x*** axis our input feature value.
  - Linear relationship, can be represented using equation of a line ***y*** **=** ***mx*** **+** ***b***.
  - ML Convention: ***y'*** **=** ***b*** **+** ***w1x1*** where
    - *y'* is the predicted label ( desired output).
    - *b* is bias, y intercept sometimes w0
    - *w1* is weight feature 1. Same concept as *m*
    - *x1* is a feature (known input)
  - to infer y' for value x1, substitute in value in model.
  - more sophisticated model rely on more than one feature: ***y'*** **=** ***b*** **+** ***w1x1*** **+** ***w2x2*** **+** ***w3x3***

- **Training**:

  - Training a model simply means learning        (determining) good values for all the weights and the bias from labeled examples.
  - ML algorithm builds a model by examining many examples and attempting to find a model which minimises loss ( **empirical risk minimisation** )
  - **Loss** is a number indicating how bad the models prediction was on a single example.
  - The goal of training a model is to find a set of weights and biases that have low loss, on average, across all examples.
  - A *loss function* is a mathematical model which aggregates individual losses in a meaningful fashion.
  - LR models generally use squared loss (L2), the squared loss for a given example.

        L2 = sqaure diff between label and prediction.
           = (observation - prediction(x))^2
           = (y-y')^2
  - **Mean Squared Error (MSE)** is the average loss per example over the whole the data set. Calculated by summing losses for individual examples and divide by number of examples. 
