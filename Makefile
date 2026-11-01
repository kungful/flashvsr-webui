.PHONY: help bootstrap dev real smoke docker-up docker-down clean

help:
	@echo "make bootstrap   # 克隆官方代码 + 建 venv + 装依赖 + 拉权重"
	@echo "make dev         # 本地 mock 引擎（无需 GPU），http://127.0.0.1:7861"
	@echo "make real        # 真引擎，http://127.0.0.1:7860"
	@echo "make smoke       # 冒烟测试（mock）"
	@echo "make docker-up   # docker compose 起真引擎"

bootstrap:
	./scripts/bootstrap.sh

dev:
	FVSR_ENGINE=mock FVSR_PORT=7861 FVSR_DATA=./data python3 app.py

real:
	FVSR_ENGINE=real FVSR_PORT=7860 FVSR_DATA=./data ./deploy/entrypoint.sh

smoke:
	./scripts/smoke_test.sh

docker-up:
	docker compose -f deploy/docker/docker-compose.yml up -d --build

docker-down:
	docker compose -f deploy/docker/docker-compose.yml down

clean:
	rm -rf data/* __pycache__ templates/__pycache__
