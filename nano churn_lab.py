from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


spark = SparkSession.builder \
    .appName("Telco Customer Churn Prediction") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


print("\n========== Part 1: Data Exploration ==========")

df = spark.read.csv(
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    header=True,
    inferSchema=True
)

df.show(5)
df.printSchema()

print("Total Rows:", df.count())

df.groupBy("Churn").count().show()


df = df.drop("customerID")

df = df.withColumn(
    "TotalCharges",
    when(trim(col("TotalCharges")) == "", None)
    .otherwise(trim(col("TotalCharges")).cast("double"))
)

df = df.na.drop()

print("Rows After Cleaning:", df.count())


categorical_cols = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

numeric_cols = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

print("Categorical Columns:")
print(categorical_cols)

print("Numeric Columns:")
print(numeric_cols)


print("\n========== Part 2: Preprocessing ==========")

label_indexer = StringIndexer(
    inputCol="Churn",
    outputCol="label"
)

index_cols = [c + "_index" for c in categorical_cols]
ohe_cols = [c + "_ohe" for c in categorical_cols]

cat_indexer = StringIndexer(
    inputCols=categorical_cols,
    outputCols=index_cols,
    handleInvalid="keep"
)

encoder = OneHotEncoder(
    inputCols=index_cols,
    outputCols=ohe_cols
)

assembler = VectorAssembler(
    inputCols=numeric_cols + ohe_cols,
    outputCol="features",
    handleInvalid="keep"
)

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print("Train Rows:", train_df.count())
print("Test Rows:", test_df.count())


print("\n========== Part 3 and Part 4: Models and Evaluation ==========")

accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

auc_eval = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)


def evaluate_model(predictions, model_name):
    accuracy = accuracy_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)
    auc = auc_eval.evaluate(predictions)

    print("\nModel:", model_name)
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("AUC:", auc)

    return model_name, accuracy, f1, auc


dt = DecisionTreeClassifier(
    labelCol="label",
    featuresCol="features",
    maxDepth=5,
    seed=42
)

dt_pipeline = Pipeline(stages=[
    label_indexer,
    cat_indexer,
    encoder,
    assembler,
    dt
])

dt_model = dt_pipeline.fit(train_df)
dt_predictions = dt_model.transform(test_df)
dt_result = evaluate_model(dt_predictions, "Decision Tree")


rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=30,
    maxDepth=5,
    seed=42
)

rf_pipeline = Pipeline(stages=[
    label_indexer,
    cat_indexer,
    encoder,
    assembler,
    rf
])

rf_model = rf_pipeline.fit(train_df)
rf_predictions = rf_model.transform(test_df)
rf_result = evaluate_model(rf_predictions, "Random Forest")


lr = LogisticRegression(
    labelCol="label",
    featuresCol="features",
    maxIter=20,
    regParam=0.01
)

lr_pipeline = Pipeline(stages=[
    label_indexer,
    cat_indexer,
    encoder,
    assembler,
    lr
])

lr_model = lr_pipeline.fit(train_df)
lr_predictions = lr_model.transform(test_df)
lr_result = evaluate_model(lr_predictions, "Logistic Regression")


print("\n========== Model Comparison ==========")

results_df = spark.createDataFrame(
    [dt_result, rf_result, lr_result],
    ["Model", "Accuracy", "F1", "AUC"]
)

results_df.show(truncate=False)


print("\n========== Part 5: Tuning ==========")

rf_tuning = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    seed=42
)

rf_tuning_pipeline = Pipeline(stages=[
    label_indexer,
    cat_indexer,
    encoder,
    assembler,
    rf_tuning
])

param_grid = ParamGridBuilder() \
    .addGrid(rf_tuning.numTrees, [10, 20]) \
    .addGrid(rf_tuning.maxDepth, [3, 5]) \
    .build()

tvs = TrainValidationSplit(
    estimator=rf_tuning_pipeline,
    estimatorParamMaps=param_grid,
    evaluator=auc_eval,
    trainRatio=0.8,
    seed=42
)

tuned_model = tvs.fit(train_df)
tuned_predictions = tuned_model.transform(test_df)

tuned_result = evaluate_model(tuned_predictions, "Tuned Random Forest")

best_pipeline = tuned_model.bestModel
best_rf = best_pipeline.stages[-1]

print("Best numTrees:", best_rf.getNumTrees)
print("Best maxDepth:", best_rf.getOrDefault("maxDepth"))


print("\n========== Part 6: Feature Analysis ==========")

reduced_numeric_cols = [
    "SeniorCitizen",
    "tenure"
]

reduced_assembler = VectorAssembler(
    inputCols=reduced_numeric_cols + ohe_cols,
    outputCol="features",
    handleInvalid="keep"
)

rf_reduced_pipeline = Pipeline(stages=[
    label_indexer,
    cat_indexer,
    encoder,
    reduced_assembler,
    rf
])

rf_reduced_model = rf_reduced_pipeline.fit(train_df)
rf_reduced_predictions = rf_reduced_model.transform(test_df)

reduced_result = evaluate_model(
    rf_reduced_predictions,
    "Random Forest Without Charges"
)


print("\n========== Part 7: Prediction for New Customer ==========")

new_customer = spark.createDataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1026.0,
    "Churn": "No"
}])

new_prediction = tuned_model.transform(new_customer)

new_prediction.select(
    "prediction",
    "probability"
).show(truncate=False)


spark.stop()
