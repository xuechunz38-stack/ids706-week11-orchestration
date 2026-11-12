COMPOSE := docker compose -f infra/docker-compose.yml --env-file .env

.PHONY: ensure-env up airflow agents frontend trigger test loop-cli down logs ps

ensure-env:
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")

# Everything: warehouse + Airflow + agent gateway (UI on :8088 and API on :8000)
up: ensure-env
	$(COMPOSE) up --build -d
	@echo "Airflow UI:  http://localhost:8088  (admin / admin)"
	@echo "Gateway API: http://127.0.0.1:8000/healthz"
	@echo "Agent UI:    make frontend  ->  http://127.0.0.1:5173/solution-loop"

# Only the pipeline side (warehouse + Airflow)
airflow: ensure-env
	$(COMPOSE) up --build -d warehouse-db airflow-db airflow-init airflow-webserver airflow-scheduler

# Only the agent side (warehouse + gateway)
agents: ensure-env
	$(COMPOSE) up --build -d warehouse-db redis gateway

frontend:
	cd frontend && npm install && npm run dev

# Unpause + trigger the DAG from the CLI (same as the UI buttons)
trigger:
	$(COMPOSE) exec airflow-scheduler airflow dags unpause life_expectancy_pipeline
	$(COMPOSE) exec airflow-scheduler airflow dags trigger life_expectancy_pipeline

# Unit tests inside the Airflow image (pandas/sklearn already installed)
test:
	$(COMPOSE) run --rm --no-deps --entrypoint bash -w /repo airflow-scheduler -c \
	  "pip install -q pytest 'langgraph>=0.2' >/dev/null && python -m pytest tests/unit/test_pipeline_etl.py -q"

# Run the modified agent loop headless inside the gateway container, save metrics
loop-cli:
	$(COMPOSE) exec gateway python /packages/solution/langgraph/run_cli.py \
	  --dataset /bundled/life_expectancy_clean.csv --max-cycles 4 > reports/agent_loop_result.json
	docker compose -f infra/docker-compose.yml --env-file .env run --rm --no-deps --entrypoint python -w /repo airflow-scheduler scripts/plot_agent_eval.py
	@echo "wrote reports/agent_loop_result.json"

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f --tail=100

down:
	-@pids=$$(lsof -tiTCP:5173 -sTCP:LISTEN 2>/dev/null); [ -n "$$pids" ] && kill $$pids || true
	$(COMPOSE) down
