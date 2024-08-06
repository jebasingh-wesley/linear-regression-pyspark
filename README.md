# Linear Regression with PySpark

In statistics, linear regression is a linear approach for modeling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables). The case of one explanatory variable is called simple linear regression; for more than one, the process is called multiple linear regression.

## Linear Regression

### Simple Linear Regression
Simple linear regression involves predicting a linear relationship between two variables, \( x \) and \( y \).

#### Key Concepts:
- **Minimizing Vertical Distance**: The goal is to minimize the vertical distances (residuals) between the observed data points and the regression line.
- **Regression Line**: The line that best fits the data points in the least-squares sense.

#### Example:
- **Independent Variable (x)**: Month (We can predict sales using the month)
- **Dependent Variable (y)**: Sales (The sales that can be made in a given month)

The regression line formula is:
\[ y = mx + c \]
- \( c \): Intercept
- \( x \): Given value
- \( m \): Slope

For a visual explanation, check out [this video](https://www.youtube.com/watch?v=dwNBDG7pPqY).

### Multiple Linear Regression
Multiple linear regression involves multiple explanatory variables and their linear relationship with the response variable.

## Problem Statement
Build a predictive model for a shipping company to estimate how many crew members a ship requires. We will use linear regression to predict the number of crew members needed based on various factors.

---

Feel free to explore the code, contribute, and raise issues if you encounter any. Happy coding!
