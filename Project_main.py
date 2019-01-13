from __future__ import print_function
from operator import add
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.evaluation import MulticlassMetrics
import sys
import numpy as np

# encodinng the delay
def encoding_delay(x):
    
    if x[3] < 0:
        a = "early"
    elif x[3] > 0:
        a = "late"
    elif x[3] == 0:
        a = "on time"
    else:
        a = "nothing"
    
    return (x[0],x[1],x[2],a,x[4])

#Confusion matrix
def confuse(x):
    actual_label = x[0]
    predicted_label = x[1]

    if actual_label == 0:
        if predicted_label ==0:
            return (1,np.array([1,0,0,0]))
        elif predicted_label==1:
            return (1,np.array([0,0,0,1]))
    elif actual_label==1:
        if predicted_label ==0:
            return (1,np.array([0,1,0,0]))
        elif predicted_label==1:
            return (1,np.array([0,0,1,0]))

if __name__ == "__main__":
    if len(sys.argv)!=3:
        print("\n")
        print("\n")
        print("Error in number of arguments")
        print("Usage: spark-submit file.py <input-file> <output-file>",file=sys.stderr)
        print("\n")
        print("\n")
        
        exit(-1)

    #Creating the spark session
    spark = SparkSession\
                .builder\
                .appName("Dublin_Congestion_prediction")\
                .getOrCreate()

    sc = spark.sparkContext

    # Reading in the data
    data = sc.textFile(sys.argv[1])

    # Filtering the data by converting the string into list of numbers
    rows = data.map(lambda x:x.split(','))
    
    #Dropping the columns 
    filteredRows1 = rows.map(lambda x:(int(x[2]),x[6],int(x[7]),int(x[10]),int(x[14])))
    
    filteredRows = filteredRows1.map(encoding_delay)
    # Creating the schema for the data frame
    schema = StructType([
            StructField('Direction',IntegerType(),True),
            StructField('Operator',StringType(),True),
            StructField('Congestion',IntegerType(),True),
            StructField('Delay',StringType(),True),
            StructField('Atstop',IntegerType(),True),
    ])

    #Creating the df using the schema
    filteredRows_df = spark.createDataFrame(filteredRows,schema)

    # Indexing the string using the string indexer
    indexer = StringIndexer(inputCol = "Operator",outputCol = 'Operator_int')
    indexed = indexer.fit(filteredRows_df).transform(filteredRows_df)
    indexed = indexed.drop("Operator")
    indexer1 = StringIndexer(inputCol = "Delay",outputCol = 'Delay_int')
    indexed = indexer1.fit(indexed).transform(indexed)
    indexed = indexed.drop("Delay")
    indexed = indexed.drop("Direction")

    # Creating the categorical variables
    encoder = OneHotEncoderEstimator(inputCols=["Operator_int", "Delay_int"],
                                 outputCols=["Operator", "Delay"])
    encoded = encoder.fit(indexed).transform(indexed)
    encoded = encoded.drop("Operator_int")
    encoded = encoded.drop("Delay_int")
    
    # Extracting the features
    feature_cols = ['Atstop','Operator','Delay']
    assembler = VectorAssembler(inputCols=feature_cols,outputCol="features")
    output = assembler.transform(encoded)

    # Converting a dataframe into rdd again
    encoded_rdd = output.select("Congestion","features").rdd.map(tuple).cache()
    
    # Splitting the training and testing data
    trainingData, testData = encoded_rdd.randomSplit([0.8,0.2], seed=0)
    
    # Training the model
    prepared_rdd = trainingData.map(lambda x:LabeledPoint(x[0],Vectors.dense(x[1].toArray()))).cache()

    model = LogisticRegressionWithLBFGS.train(prepared_rdd, iterations=100)

    # Evaluating the model on testing data
    prepared_rdd1 = testData.map(lambda x:LabeledPoint(x[0],Vectors.dense(x[1].toArray()))).cache()
    labelsAndPreds = prepared_rdd1.map(lambda p: (p.label, model.predict(p.features)))
    # Confusion matrix
    confusion_matrix = labelsAndPreds.map(confuse).reduceByKey(lambda x,y:x+y).collect()
    print("Confuse",confusion_matrix)
    
    # Calculating the f-measure
    tp = float(confusion_matrix[0][1][0])
    fp = float(confusion_matrix[0][1][1])
    fn = float(confusion_matrix[0][1][2])
    tn = float(confusion_matrix[0][1][3])

    # Recall and precision
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    
    #F1-measure
    f_measure = 2*recall*precision/(recall+precision)
    
    print("f-measure = ",f_measure)
    
    #printing to output file
    result = ["f_measure:" + str(f_measure) + "confusion matrix:" + str(confusion_matrix)]
    task_output = sc.parallelize(result, 1)
    task_output.saveAsTextFile(sys.argv[2])
    
    #with open(sys.argv[2]+'.txt','w') as f:
        #f.write("confusion matrix  = {}".format(confusion_matrix))
        #f.write("\n\n")
        #f.write("f_measure =  {}".format(f_measure))
    # Stopping the spark-context
    sc.stop()