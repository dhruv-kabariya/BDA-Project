# Describe the problem

#### In bigger city ,number of house is in million number , so company like magic brick , 99 acres have millions of rows data in their database , so user want specific recommended housing price for their house given that number of rooms , carpet area of house and neighbourhood rating. So our task was to predict the house price based on the given attributes.

<br>
<br>

# Explain the solution

#### To solve this problem we used linear regression model to predict the house price based on the given attributes.And we used pyspark libray to create the model and handle the big data .

<br>
<br>

# Implementation

## PLATFORM USED FOR THE PROJECT

### google collab

<br>

## INSTALLATION NEEDED

<br>

#### !pip install pyspark

#### pip install unzip

#### !pip install pandas

#### !pip install numpy

<br>

## CLASS NEEDED

<br>

### SparkSession - SparkSession is the entry point to Spark SQL functionality.

<br>

### VectorAssembler - VectorAssembler is used to create a vector from a given list of columns.

<br>

### LinearRegression - LinearRegression is used to create a linear regression model.

<br>

### RegressionEvaluator - RegressionEvaluator is used to evaluate the model.

<br>
<br>

# IMPLEMENTATION STEPS

<br>

## 1 : Download the data from kaggle - schirmerchad/bostonhoustingmlnd

<br>

## 2: Unzip the data

<br>

## 3: Create a spark session

> spark = SparkSession.builder .appName("how to read csv file") .getOrCreate()

<br>

## 4: Read the data from the csv file

> df =spark.read.cs('housing.csv',inferSchema =True,header = True)

<br>

## 5: change data to parquet format

>df.repartition(1).write.mode('overwrite').parquet('housing')
<br>

>df = spark.read.parquet('housing')

<br>

## 6: Create a vector assembler

>vectorAssembler =  VectorAssembler(inputCols =['RM','LSTAT','PTRATIO' ],outputCol='features')'
<br>

>vhouse_df = vectorAssembler.transform(df)
<br>

>vhouse_df =vhouse_df.select(['features','MEDV'])

<br>

## 7: Create a linear regression model

>lr = LinearRegression(featuresCol = 'features', labelCol='MEDV', maxIter=10, regParam=0.3, elasticNetParam=0.8)
<br>

>lr_model = lr.fit(train_df)

<br>

## 8: Predict and evaluate the model

>lr_predictions = lr_model.transform(test_df)
<br>

>lr_predictions.select("prediction","MEDV","features").show(5)
<br>

>lr_evaluator = RegressionEvaluator>(predictionCol="prediction", \
                 labelCol="MEDV",metricName="r2")


<br>

![results](https://github.com/dhruv-kabariya/BDA-Project/blob/main/WhatsApp%20Image%202022-05-01%20at%2011.34.08%20AM.jpeg?raw=true)
