import mlflow

acc = 0.90

with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", acc)
    open("model_info.txt", "w").write(run.info.run_id)
    open("accuracy.txt", "w").write(str(acc))
