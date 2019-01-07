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
