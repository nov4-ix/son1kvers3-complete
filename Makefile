# Son1k v3.0 - Professional Makefile
# AI Music Generation Platform

SHELL := /bin/bash
.DEFAULT_GOAL := help

# === CONFIGURATION ===
PROJECT_NAME := son1k-v3
PYTHON_VERSION := 3.11
NODE_VERSION := 20
BACKEND_DIR := backend
FRONTEND_DIR := frontend
STORAGE_DIR := storage
LOGS_DIR := logs

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[1;37m
NC := \033[0m # No Color

# === HELP ===
.PHONY: help
help: ## Show this help message
	@echo "$(CYAN)🎵 Son1k v3.0 - AI Music Generation Platform$(NC)"
	@echo "$(WHITE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)📋 Available Commands:$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(PURPLE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BLUE)🚀 Quick Start:$(NC) make setup && make dev"
	@echo "$(BLUE)🧪 Run Tests:$(NC) make test"
	@echo "$(BLUE)🐳 Docker:$(NC) make docker-up"

##@ 🏗️  Setup & Installation
.PHONY: setup setup-backend setup-frontend setup-dirs setup-all install-deps check-deps
setup: check-deps setup-dirs setup-backend setup-frontend ## Complete initial setup
	@echo "$(GREEN)✅ Setup complete! Run 'make dev' to start development.$(NC)"

setup-dirs: ## Create required directories
	@echo "$(BLUE)📁 Creating directories...$(NC)"
	@mkdir -p $(STORAGE_DIR)/{uploads,output,models}/{,ghost}
	@mkdir -p $(LOGS_DIR)
	@mkdir -p $(BACKEND_DIR)/data
	@touch $(STORAGE_DIR)/uploads/.gitkeep
	@touch $(STORAGE_DIR)/output/.gitkeep
	@touch $(STORAGE_DIR)/models/.gitkeep
	@echo "$(GREEN)✅ Directories created$(NC)"

setup-backend: ## Setup Python backend
	@echo "$(BLUE)🐍 Setting up Python backend...$(NC)"
	@if [ ! -d ".venv" ]; then \
		python$(PYTHON_VERSION) -m venv .venv; \
	fi
	@source .venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r $(BACKEND_DIR)/requirements.txt
	@if [ ! -f "$(BACKEND_DIR)/.env" ]; then \
		cp $(BACKEND_DIR)/.env.example $(BACKEND_DIR)/.env; \
		echo "$(YELLOW)⚠️  Created $(BACKEND_DIR)/.env from example$(NC)"; \
	fi
	@echo "$(GREEN)✅ Backend setup complete$(NC)"

setup-frontend: ## Setup Node.js frontend
	@echo "$(BLUE)⚛️  Setting up React frontend...$(NC)"
	@cd $(FRONTEND_DIR) && npm ci
	@if [ ! -f "$(FRONTEND_DIR)/.env" ]; then \
		cp $(FRONTEND_DIR)/.env.example $(FRONTEND_DIR)/.env; \
		echo "$(YELLOW)⚠️  Created $(FRONTEND_DIR)/.env from example$(NC)"; \
	fi
	@echo "$(GREEN)✅ Frontend setup complete$(NC)"

install-deps: ## Install additional system dependencies
	@echo "$(BLUE)📦 Installing system dependencies...$(NC)"
	@if command -v brew &> /dev/null; then \
		echo "$(BLUE)🍺 Installing with Homebrew...$(NC)"; \
		brew install ffmpeg rubberband || true; \
	elif command -v apt-get &> /dev/null; then \
		echo "$(BLUE)🐧 Installing with apt...$(NC)"; \
		sudo apt-get update && sudo apt-get install -y ffmpeg librubberband-dev; \
	else \
		echo "$(YELLOW)⚠️  Please install ffmpeg and rubberband manually$(NC)"; \
	fi
	@npm install -g concurrently || true
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

check-deps: ## Check required dependencies
	@echo "$(BLUE)🔍 Checking dependencies...$(NC)"
	@command -v python$(PYTHON_VERSION) >/dev/null 2>&1 || { echo "$(RED)❌ Python $(PYTHON_VERSION) required$(NC)"; exit 1; }
	@command -v node >/dev/null 2>&1 || { echo "$(RED)❌ Node.js required$(NC)"; exit 1; }
	@echo "$(GREEN)✅ Dependencies check passed$(NC)"

##@ 🚀 Development
.PHONY: dev dev-backend dev-frontend dev-all start stop restart
dev: dev-all ## Start development servers (alias for dev-all)

dev-backend: ## Start backend only
	@echo "$(BLUE)🔧 Starting backend server...$(NC)"
	@source .venv/bin/activate && \
		cd $(BACKEND_DIR) && \
		uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

dev-frontend: ## Start frontend only
	@echo "$(BLUE)🌐 Starting frontend server...$(NC)"
	@cd $(FRONTEND_DIR) && npm run dev

dev-all: ## Start both backend and frontend with concurrently
	@echo "$(GREEN)🚀 Starting Son1k development servers...$(NC)"
	@npx concurrently -k -n "$(PURPLE)API$(NC),$(CYAN)WEB$(NC)" -c "blue,green" \
		"source .venv/bin/activate && cd $(BACKEND_DIR) && uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload" \
		"cd $(FRONTEND_DIR) && npm run dev"

start: dev-all ## Alias for dev-all

stop: ## Stop development servers
	@echo "$(YELLOW)🛑 To stop servers, use Ctrl+C in the terminal running 'make dev'$(NC)"

restart: stop dev-all ## Restart development servers

##@ 🧪 Testing
.PHONY: test test-backend test-frontend test-smoke test-integration test-coverage
test: test-backend test-frontend ## Run all tests

test-backend: ## Run backend tests
	@echo "$(BLUE)🧪 Running backend tests...$(NC)"
	@source .venv/bin/activate && \
		cd $(BACKEND_DIR) && \
		python -m pytest tests/ -v --tb=short

test-frontend: ## Run frontend tests
	@echo "$(BLUE)🧪 Running frontend tests...$(NC)"
	@cd $(FRONTEND_DIR) && npm test

test-smoke: ## Run smoke tests only
	@echo "$(BLUE)🔥 Running smoke tests...$(NC)"
	@source .venv/bin/activate && \
		cd $(BACKEND_DIR) && \
		python -m pytest tests/test_api.py::TestHealthAndBasics -v

test-integration: ## Run integration tests
	@echo "$(BLUE)🔗 Running integration tests...$(NC)"
	@source .venv/bin/activate && \
		cd $(BACKEND_DIR) && \
		python -m pytest tests/test_api.py::test_full_integration_workflow -v

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)📊 Running tests with coverage...$(NC)"
	@source .venv/bin/activate && \
		cd $(BACKEND_DIR) && \
		python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

##@ 🐳 Docker
.PHONY: docker-build docker-up docker-down docker-logs docker-clean docker-dev
docker-build: ## Build Docker images
	@echo "$(BLUE)🐳 Building Docker images...$(NC)"
	@docker-compose build

docker-up: ## Start services with Docker
	@echo "$(GREEN)🐳 Starting Son1k with Docker...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)✅ Services started:$(NC)"
	@echo "  $(CYAN)Frontend:$(NC) http://localhost:3000"
	@echo "  $(CYAN)API:$(NC) http://localhost:8000"
	@echo "  $(CYAN)Docs:$(NC) http://localhost:8000/docs"

docker-down: ## Stop Docker services
	@echo "$(YELLOW)🐳 Stopping Docker services...$(NC)"
	@docker-compose down

docker-logs: ## Show Docker logs
	@docker-compose logs -f

docker-clean: ## Clean Docker resources
	@echo "$(BLUE)🐳 Cleaning Docker resources...$(NC)"
	@docker-compose down -v --remove-orphans
	@docker system prune -f
	@echo "$(GREEN)✅ Docker cleanup complete$(NC)"

docker-dev: ## Start development environment with Docker
	@echo "$(GREEN)🐳 Starting development environment...$(NC)"
	@docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

##@ 🏗️  Build & Deploy
.PHONY: build build-backend build-frontend deploy deploy-staging deploy-prod
build: build-backend build-frontend ## Build for production

build-backend: ## Build backend for production
	@echo "$(BLUE)🏗️  Building backend...$(NC)"
	@source .venv/bin/activate && \
		cd $(BACKEND_DIR) && \
		python -m pip install --upgrade pip && \
		pip install -r requirements.txt

build-frontend: ## Build frontend for production
	@echo "$(BLUE)🏗️  Building frontend...$(NC)"
	@cd $(FRONTEND_DIR) && \
		npm ci --only=production && \
		npm run build

deploy: build ## Deploy to production (placeholder)
	@echo "$(YELLOW)🚀 Deployment script placeholder$(NC)"
	@echo "$(BLUE)Add your deployment commands here$(NC)"

deploy-staging: build ## Deploy to staging
	@echo "$(YELLOW)📋 Deploying to staging...$(NC)"
	@echo "$(BLUE)Add staging deployment commands$(NC)"

deploy-prod: build ## Deploy to production
	@echo "$(RED)🚨 Production deployment$(NC)"
	@echo "$(BLUE)Add production deployment commands$(NC)"

##@ 🧹 Maintenance
.PHONY: clean clean-cache clean-logs clean-storage clean-all format lint fix
clean: clean-cache ## Clean generated files

clean-cache: ## Clean cache files
	@echo "$(BLUE)🧹 Cleaning cache...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@cd $(FRONTEND_DIR) && rm -rf .next node_modules/.cache 2>/dev/null || true
	@source .venv/bin/activate && python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
	@echo "$(GREEN)✅ Cache cleaned$(NC)"

clean-logs: ## Clean log files
	@echo "$(BLUE)🧹 Cleaning logs...$(NC)"
	@rm -rf $(LOGS_DIR)/*.log
	@echo "$(GREEN)✅ Logs cleaned$(NC)"

clean-storage: ## Clean generated storage files
	@echo "$(YELLOW)⚠️  This will delete all generated music files!$(NC)"
	@read -p "Are you sure? (y/N) " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		rm -rf $(STORAGE_DIR)/output/*.wav; \
		rm -rf $(STORAGE_DIR)/uploads/ghost/*; \
		echo "$(GREEN)✅ Storage cleaned$(NC)"; \
	else \
		echo ""; \
		echo "$(BLUE)Cancelled$(NC)"; \
	fi

clean-all: clean-cache clean-logs ## Clean everything (except storage)
	@echo "$(GREEN)✅ Full cleanup complete$(NC)"

format: ## Format code
	@echo "$(BLUE)🎨 Formatting code...$(NC)"
	@source .venv/bin/activate && \
		cd $(BACKEND_DIR) && \
		black src/ tests/ && \
		ruff check src/ tests/ --fix
	@cd $(FRONTEND_DIR) && npm run format || true
	@echo "$(GREEN)✅ Code formatted$(NC)"

lint: ## Lint code
	@echo "$(BLUE)🔍 Linting code...$(NC)"
	@source .venv/bin/activate && \
		cd $(BACKEND_DIR) && \
		ruff check src/ tests/
	@cd $(FRONTEND_DIR) && npm run lint
	@echo "$(GREEN)✅ Linting complete$(NC)"

fix: format lint ## Format and lint code

##@ 📊 Monitoring
.PHONY: status health logs tail-logs ps
status: ## Show service status
	@echo "$(BLUE)📊 Service Status:$(NC)"
	@curl -s http://localhost:8000/health | jq . 2>/dev/null || echo "$(RED)❌ Backend not running$(NC)"
	@curl -s http://localhost:3000/health 2>/dev/null && echo "$(GREEN)✅ Frontend running$(NC)" || echo "$(YELLOW)⚠️  Frontend status unknown$(NC)"

health: ## Health check
	@echo "$(BLUE)🔍 Health Check:$(NC)"
	@curl -f http://localhost:8000/health >/dev/null 2>&1 && echo "$(GREEN)✅ API healthy$(NC)" || echo "$(RED)❌ API unhealthy$(NC)"

logs: ## Show recent logs
	@echo "$(BLUE)📋 Recent logs:$(NC)"
	@tail -n 50 $(LOGS_DIR)/*.log 2>/dev/null || echo "$(YELLOW)No log files found$(NC)"

tail-logs: ## Tail logs in real-time
	@echo "$(BLUE)📋 Tailing logs (Ctrl+C to stop):$(NC)"
	@tail -f $(LOGS_DIR)/*.log 2>/dev/null || echo "$(YELLOW)No log files found$(NC)"

ps: ## Show running processes
	@echo "$(BLUE)📋 Python processes:$(NC)"
	@ps aux | grep python | grep -v grep || echo "$(YELLOW)No Python processes$(NC)"
	@echo "$(BLUE)📋 Node processes:$(NC)"
	@ps aux | grep node | grep -v grep || echo "$(YELLOW)No Node processes$(NC)"

##@ 🔧 Utilities
.PHONY: install-dev update-deps backup restore env-check
install-dev: ## Install development tools
	@echo "$(BLUE)🔧 Installing development tools...$(NC)"
	@source .venv/bin/activate && \
		pip install black ruff pytest-cov pre-commit
	@cd $(FRONTEND_DIR) && npm install --save-dev prettier eslint
	@echo "$(GREEN)✅ Development tools installed$(NC)"

update-deps: ## Update dependencies
	@echo "$(BLUE)📦 Updating dependencies...$(NC)"
	@source .venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r $(BACKEND_DIR)/requirements.txt --upgrade
	@cd $(FRONTEND_DIR) && npm update
	@echo "$(GREEN)✅ Dependencies updated$(NC)"

backup: ## Backup important data
	@echo "$(BLUE)💾 Creating backup...$(NC)"
	@BACKUP_NAME="son1k_backup_$$(date +%Y%m%d_%H%M%S)"; \
	mkdir -p backups; \
	tar -czf "backups/$$BACKUP_NAME.tar.gz" \
		$(STORAGE_DIR) \
		$(BACKEND_DIR)/.env \
		$(FRONTEND_DIR)/.env \
		2>/dev/null || true
	@echo "$(GREEN)✅ Backup created in backups/$(NC)"

restore: ## Restore from backup (interactive)
	@echo "$(BLUE)📂 Available backups:$(NC)"
	@ls -la backups/*.tar.gz 2>/dev/null || echo "$(YELLOW)No backups found$(NC)"
	@echo "$(YELLOW)To restore: tar -xzf backups/[backup_name].tar.gz$(NC)"

env-check: ## Check environment configuration
	@echo "$(BLUE)🔍 Environment Check:$(NC)"
	@echo "  Python: $$(python$(PYTHON_VERSION) --version)"
	@echo "  Node: $$(node --version)"
	@echo "  FFmpeg: $$(ffmpeg -version 2>&1 | head -1 || echo 'Not installed')"
	@echo "  Rubberband: $$(command -v rubberband >/dev/null && echo 'Installed' || echo 'Not installed')"
	@echo "  Virtual env: $$([[ -n "$$VIRTUAL_ENV" ]] && echo 'Active' || echo 'Inactive')"

##@ 📚 Documentation
.PHONY: docs docs-api docs-serve
docs: ## Generate documentation
	@echo "$(BLUE)📚 Generating documentation...$(NC)"
	@echo "$(YELLOW)Documentation generation not implemented yet$(NC)"

docs-api: ## Generate API documentation
	@echo "$(BLUE)📚 API Documentation available at:$(NC)"
	@echo "  $(CYAN)http://localhost:8000/docs$(NC) (Swagger UI)"
	@echo "  $(CYAN)http://localhost:8000/redoc$(NC) (ReDoc)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)📚 Serving documentation...$(NC)"
	@echo "$(CYAN)http://localhost:8000/docs$(NC)"

# === ADVANCED TARGETS ===
##@ 🎛️  Advanced
.PHONY: benchmark profile monitor
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)⚡ Running benchmarks...$(NC)"
	@source .venv/bin/activate && \
		cd $(BACKEND_DIR) && \
		python -m pytest tests/test_performance.py -v 2>/dev/null || echo "$(YELLOW)No performance tests found$(NC)"

profile: ## Profile application performance
	@echo "$(BLUE)📊 Profiling application...$(NC)"
	@echo "$(YELLOW)Profiling not implemented yet$(NC)"

monitor: ## Monitor system resources
	@echo "$(BLUE)📊 System monitoring:$(NC)"
	@echo "  GPU: $$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo 'N/A')%"
	@echo "  Memory: $$(free -h | grep '^Mem:' | awk '{print $$3 "/" $$2}' 2>/dev/null || echo 'N/A')"
	@echo "  Disk: $$(df -h . | tail -1 | awk '{print $$3 "/" $$2 " (" $$5 " used)"}' 2>/dev/null || echo 'N/A')"

# === SECRET TARGETS ===
.PHONY: demo show-secrets
demo: ## Run demonstration
	@echo "$(PURPLE)🎭 Son1k Demo Mode$(NC)"
	@echo "$(CYAN)Starting demo with sample data...$(NC)"
	@make dev-all

show-secrets: ## Show hidden Easter eggs
	@echo "$(PURPLE)🎵 Son1k Easter Eggs:$(NC)"
	@echo "  make demo        - Demo mode"
	@echo "  make show-logo   - ASCII logo"
	@echo "  make motivate    - Random motivation"

motivate: ## Random motivation
	@QUOTES=("🎵 Music is the universal language of mankind" \
			"🚀 Every expert was once a beginner" \
			"💡 Innovation distinguishes leaders from followers" \
			"🎯 The only way to do great work is to love what you do" \
			"🌟 Your limitation—it's only your imagination"); \
	echo "$(CYAN)$${QUOTES[$$RANDOM % $${#QUOTES[@]}]}$(NC)"

show-logo: ## Show ASCII logo
	@echo "$(PURPLE)"
	@echo "  ____              _ _    __     _____  ___  "
	@echo " / ___|  ___  _ __ / | | __\ \   / /___|/ _ \ "
	@echo " \___ \ / _ \| '_ \| | |/ / \ \ / /|___ \ | | |"
	@echo "  ___) | (_) | | | | |   <   \ V /  ___) | |_| |"
	@echo " |____/ \___/|_| |_|_|_|\_\   \_/  |____/ \___/ "
	@echo ""
	@echo " 🎵 AI Music Generation Platform"
	@echo "$(NC)"

# === VALIDATION ===
.PHONY: validate validate-env validate-deps validate-config
validate: validate-env validate-deps validate-config ## Validate complete setup

validate-env: ## Validate environment files
	@echo "$(BLUE)🔍 Validating environment...$(NC)"
	@[ -f "$(BACKEND_DIR)/.env" ] && echo "$(GREEN)✅ Backend .env exists$(NC)" || echo "$(RED)❌ Backend .env missing$(NC)"
	@[ -f "$(FRONTEND_DIR)/.env" ] && echo "$(GREEN)✅ Frontend .env exists$(NC)" || echo "$(RED)❌ Frontend .env missing$(NC)"

validate-deps: ## Validate dependencies
	@echo "$(BLUE)🔍 Validating dependencies...$(NC)"
	@source .venv/bin/activate && python -c "import torch, transformers, librosa; print('$(GREEN)✅ Key Python packages installed$(NC)')" 2>/dev/null || echo "$(RED)❌ Missing Python packages$(NC)"
	@cd $(FRONTEND_DIR) && npm list react >/dev/null 2>&1 && echo "$(GREEN)✅ React installed$(NC)" || echo "$(RED)❌ React missing$(NC)"

validate-config: ## Validate configuration
	@echo "$(BLUE)🔍 Validating configuration...$(NC)"
	@[ -d "$(STORAGE_DIR)" ] && echo "$(GREEN)✅ Storage directory exists$(NC)" || echo "$(RED)❌ Storage directory missing$(NC)"
	@[ -d ".venv" ] && echo "$(GREEN)✅ Virtual environment exists$(NC)" || echo "$(RED)❌ Virtual environment missing$(NC)"

# Make sure intermediate files are not deleted
.PRECIOUS: %.log %.json %.wav

# Parallel execution settings
.NOTPARALLEL: setup clean

# Silent mode for some commands
%/clean:
	@$(MAKE) -s clean

# Include local customizations if present
-include Makefile.local