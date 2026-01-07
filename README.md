# LinearRegressionModel

## Notebook Summary

This notebook demonstrates a simple linear regression model implemented from scratch to predict salary based on years of experience. The process involves several key steps: loading the dataset, implementing custom cost and gradient functions, and using gradient descent to train the model. 

The notebook begins by downloading the 'Salary_Data.csv' dataset using `kagglehub`. It then loads this data into a pandas DataFrame and extracts 'YearsExperience' as the independent variable (`x_train`) and 'Salary' as the dependent variable (`y_train`). A scatter plot is generated to visualize the relationship between these variables, showing a clear linear trend. The core of the model is built with custom Python functions:

*   `cost_function`: Calculates the mean squared error (MSE) based cost, scaled by `1/(2*m)`.
*   `gradient_function`: Computes the partial derivatives of the cost function with respect to `w` (weight) and `b` (bias), which are used to update the parameters during training.
*   `gradient_descent`: An iterative optimization algorithm that updates `w` and `b` using the calculated gradients and a specified learning rate (`alpha`) to minimize the cost function over a set number of `iterations`.

The model is trained using a `learning_rate` of 0.01 and 10,000 `iterations`. The training process logs the cost at each iteration, demonstrating the convergence of the model. Upon completion, the final learned parameters are `w: 9449.9623` and `b: 25792.2002`. These parameters represent the slope and y-intercept of the best-fit line. Finally, a plot is generated showing the original data points along with the fitted regression line derived from the learned `w` and `b` values, effectively visualizing the model's prediction.
