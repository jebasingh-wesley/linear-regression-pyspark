# Problem Statement: Build a predictive Model for the shipping company, to find an estimate of how many Crew members a ship requires.
# Let’s make the Linear Regression Model, predicting Crew members
#linear regression
from pyspark.sql import SparkSession

# Create SparkSession
spark = SparkSession.builder.master("local[1]").appName("SparkByExamples.com").getOrCreate()

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer

#path of the csv folder
dfc = "/home/ubuntu/Documents/cruise_ship_info.csv"
dfc2 = "/home/ubuntu/Documents/Ecommerce_Customers.csv"

#read the csv file
data = spark.read.csv(dfc,header = True, inferSchema = True)

#creating linear-regression we need to convet text value into numberical value so we need to create two colume
indexer = StringIndexer(inputCols=["Ship_name","Cruise_line"], outputCols=["Ship_name_Index","Cruise_line_Index"])

#after converting string to number we need to fit in to df
indexed = indexer.fit(data).transform(data)
# indexed.show(10)

# after converting we need to assemble the table in to singel table
assembler = VectorAssembler(inputCols =['Age','Tonnage','passengers','length', 'cabins', 'passenger_density', 'Ship_name_Index', 'Cruise_line_Index'],outputCol='features')
output = assembler.transform(indexed)
# after two colume
final_data = output.select('features','crew')

# Train Test split building two data one is train_data 0.7 anothe one is test_data 0.3
train_data,test_data = final_data.randomSplit([0.7,0.3])
# train_data.describe().show()
# test_data.describe().show()

# Build Model for linear
# creating a liner varible for model
regressor = LinearRegression(labelCol='crew')
model = regressor.fit(train_data)

# Evaluate Model (is used to test the data type f0r preduction)
pred_data = model.evaluate(test_data)
# pred_data.residuals.show()

# rootMeanSquaredError The root mean squared error (RMSE) is a commonly used evaluation metric for regression models. It measures the average deviation between
# the predicted values and the actual values in a regression problem.
# pred_data.rootMeanSquaredError.show()

# RegressionEvaluator r2 The R-squared value ranges between 0 and 1.
# pred_data.r2.show()

#The mean square error is the average of the square of the difference between the observed and predicted values of a variable.
# In Python, the MSE can be calculated rather easily, especially with the use of lists.
# pred_data.meanSquaredError.show()

# pred_data.meanAbsoluteError.show()

from pyspark.sql import functions as f
#Correlation Coefficient
# data.select(f.corr('crew','passengers')).show()

# fit to data type for transformation
unlabeled_data = test_data.select('features')
test_predictions = model.transform(unlabeled_data)

# it show the Linear Regression in the modal
# test_predictions.show()

import matplotlib.pyplot as plt

# Data for features and predictions
features = test_predictions.select('features').rdd.flatMap(lambda x: x).collect()
predictions = test_predictions.select('prediction').rdd.flatMap(lambda x: x).collect()

# Create a bar chart
plt.plot(range(len(predictions)), predictions)

# Customize the chart
plt.xlabel('Features')
plt.ylabel('Prediction')
plt.title('Prediction Results')
# plt.xticks(range(len(predictions)), ['Sample 1', 'Sample 2', 'Sample 3'])

# Show the chart
plt.show()
