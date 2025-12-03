# RF Forensics - Build & Deployment Makefile
.PHONY: help build up down dev prod logs clean test lint

# Default target
help:
	@echo "RF Forensics - Docker Deployment Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Build Targets:"
	@echo "  build         Build all Docker images"
	@echo "  build-backend Build backend image only"
	@echo "  build-frontend Build frontend image only"
	@echo "  build-holoscan Build holoscan image only"
	@echo ""
	@echo "Run Targets:"
	@echo "  up            Start all services"
	@echo "  dev           Start in development mode"
	@echo "  prod          Start in production mode"
	@echo "  down          Stop all services"
	@echo "  restart       Restart all services"
	@echo ""
	@echo "Monitoring:"
	@echo "  logs          Follow logs from all services"
	@echo "  logs-backend  Follow backend logs"
	@echo "  logs-frontend Follow frontend logs"
	@echo "  logs-holoscan Follow holoscan logs"
	@echo "  status        Show service status"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean         Remove containers and volumes"
	@echo "  clean-images  Remove all RF forensics images"
	@echo "  test          Run backend tests"
	@echo "  lint          Run linters"
	@echo ""

# =============================================================================
# Build Targets
# =============================================================================
build:
	docker-compose build

build-backend:
	docker-compose build backend

build-frontend:
	docker-compose build frontend

build-holoscan:
	docker-compose build holoscan

build-no-cache:
	docker-compose build --no-cache

# =============================================================================
# Run Targets
# =============================================================================
up:
	docker-compose up -d

dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

down:
	docker-compose down

restart:
	docker-compose restart

stop:
	docker-compose stop

# =============================================================================
# Logs & Monitoring
# =============================================================================
logs:
	docker-compose logs -f

logs-backend:
	docker-compose logs -f backend

logs-frontend:
	docker-compose logs -f frontend

logs-holoscan:
	docker-compose logs -f holoscan

status:
	docker-compose ps

health:
	@echo "=== Backend Health ==="
	@curl -sf http://localhost:8000/health || echo "Backend: DOWN"
	@echo ""
	@echo "=== Frontend Health ==="
	@curl -sf http://localhost:3000/health || echo "Frontend: DOWN"
	@echo ""

# =============================================================================
# Maintenance
# =============================================================================
clean:
	docker-compose down -v --remove-orphans

clean-images:
	docker rmi rf-forensics-backend rf-forensics-frontend rf-forensics-holoscan 2>/dev/null || true

prune:
	docker system prune -f

# =============================================================================
# Development
# =============================================================================
test:
	docker-compose exec backend pytest tests/ -v

test-unit:
	docker-compose exec backend pytest tests/unit/ -v

test-integration:
	docker-compose exec backend pytest tests/integration/ -v

lint:
	docker-compose exec backend ruff check src/

format:
	docker-compose exec backend ruff format src/

shell-backend:
	docker-compose exec backend /bin/bash

shell-frontend:
	docker-compose exec frontend /bin/sh

shell-holoscan:
	docker-compose exec holoscan /bin/bash

# =============================================================================
# Utilities
# =============================================================================
env:
	@test -f .env || cp .env.example .env
	@echo ".env file ready"

gpu-check:
	@nvidia-smi || echo "NVIDIA GPU not available"
	@docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi || echo "Docker GPU support not configured"
