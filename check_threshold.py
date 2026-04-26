import sys

THRESHOLD = 0.85

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

with open("accuracy.txt", "r") as f:
    accuracy = float(f.read().strip())

print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy}")
print(f"Threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print("Accuracy below threshold. Deployment stopped.")
    sys.exit(1)

print("Accuracy passed. Deployment allowed.")
