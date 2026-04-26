import mlflow

mlflow.set_experiment("mlops-quiz")

with mlflow.start_run() as run:
    accuracy = 0.90

    mlflow.log_metric("accuracy", accuracy)

    run_id = run.info.run_id

    with open("model_info.txt", "w") as f:
        f.write(run_id)

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy}")
