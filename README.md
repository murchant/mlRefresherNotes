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


## Reducing Loss##
- **Iterative Approach**


![alt text](https://developers.google.com/machine-learning/crash-course/images/GradientDescentDiagram.svg)

  - Explanation
      - Considering ***y'*** **=** ***b*** **+** ***w1x1*** we pick random starting values for b and w1.
      - The ***Compute loss*** part of the diagram is done using squared loss function, L2.
      - In the ***Compute parameter updates*** section of the diagram examines the value of the loss function and generates new values for b and w1. These values are chosen in a process called **gradient descent**.
      - The ML system re-evaluates those values against all features and labels, and a new loss function value is yielded.
      - This learning continues to iterate until the systems finds model parameters with the lowest loss, in other words when the model has **converged**.

- **Gradient Descent**
  - Instead of calculating the loss for ever value of w1 (computationally expensive), gradient descent is a better mechanism.

    ![alt text](https://developers.google.com/machine-learning/crash-course/images/GradientDescentStartingPoint.svg)
  - Steps:
    - Pick random starting value for w1.
    - Then calculate the gradient of the loss curve at w1, which is the derivative (slope) at w1.
    - When there are multiple weights, the gradient is a vector of partial derivatives with respect to the weights.
    - **Gradient** is a vector, has *magnitude* and *direction*.
    - Gradient always points in the direction of steepest increase in the loss function.
    - So, the ***gradient descent algorithm*** takes a step in the direction of the negative gradient in order to reduce loss.

    ![alt text](https://developers.google.com/machine-learning/crash-course/images/GradientDescentNegativeGradient.svg)

    - Adds some fraction of the gradient's magnitude to the starting point to determine the next point along the loss function curve.
    - The gradient descent then repeats this process, edging ever closer to the minimum.

       - *Note: When performing gradient descent, we generalize the above process to tune all the model parameters simultaneously. For example, to find the optimal values of both w1 and the bias , we calculate the gradients with respect to both w1 and b. Next, we modify the values of w1 and b based on their respective gradients. Then we repeat these steps until we reach minimum loss.*
