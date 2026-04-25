from python:3.10-slim
ARG RUN_ID
run pip istall -r requirments.txt
copy . .
run echo "preparing container...."
run echo "download the model for run ID: ${RUN-ID}"
cmd ["sh","-c","echo model container is ready for run ID: ${RUN-ID}"]

