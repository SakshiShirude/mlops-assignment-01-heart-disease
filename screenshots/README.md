# Local evidence captures

- `mlflow_runs.png` shows the two completed experiment runs.
- `mlflow_logistic_regression.png` and `mlflow_random_forest.png` show each run's parameters and five logged metrics.
- `mlflow_artifacts.png` shows the logged model, classification report, confusion matrix, and ROC curve.
- `api_health.png` shows the local model health response.
- `api_docs.png` shows the FastAPI OpenAPI page.
- `api_predict.png` shows a successful local `/predict` request and response.

These captures show local execution and hosted CI evidence. The container response JSON and request logs were produced by the successful Docker smoke test on GitHub's Linux runner; Kubernetes was not deployed.

- `github_actions_success.png` shows the successful hosted CI run and uploaded artifact. [Open run #1](https://github.com/SakshiShirude/mlops-assignment-01-heart-disease/actions/runs/39112997090).
- `docker_container_ci_run.png` shows the successful workflow run that built and started the Docker image. [Open run #6](https://github.com/SakshiShirude/mlops-assignment-01-heart-disease/actions/runs/39113734951).
- `container_health.json` and `container_predict.json` contain the actual container endpoint responses; `container_logs.txt` records Uvicorn startup and HTTP 200 request logs.
