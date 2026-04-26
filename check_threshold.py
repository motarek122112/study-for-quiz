import sys
import mlflow

THRESHOLD = 0.85

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

run = mlflow.get_run(run_id)

accuracy = run.data.metrics.get("accuracy")

print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy}")
print(f"Threshold: {THRESHOLD}")

if accuracy is None:
    print("Accuracy metric not found")
    sys.exit(1)

if accuracy < THRESHOLD:
    print("Accuracy below threshold. Deployment stopped.")
    sys.exit(1)

print("Accuracy passed. Deployment allowed.")
