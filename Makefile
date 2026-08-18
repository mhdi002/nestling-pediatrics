# Nestling deploy automation wrapper.
#
# This Makefile always shells out to ./deploy.sh, since `make` itself only
# runs where a POSIX shell already exists (Git Bash/WSL on Windows, or
# natively on Linux/macOS). Windows users without Git Bash/WSL/make: run
# .\deploy.ps1 directly instead of `make`.
SHELL := /usr/bin/env bash

.PHONY: help deploy deploy-app down logs status health model clean

help: ## show this help
	@echo "Nestling deployment targets:"
	@echo "  make deploy       full stack (GPU LLM if available, else app-only)"
	@echo "  make deploy-app   app-only, force no LLM"
	@echo "  make down         stop the stack"
	@echo "  make logs         tail logs (all services)"
	@echo "  make status       health check + docker compose ps"
	@echo "  make model        download the LLM model only"
	@echo "  make clean        stop stack AND delete data volumes (destructive)"
	@echo ""
	@echo "Windows without Git Bash/WSL: run .\\deploy.ps1 directly instead of make."

deploy: ## full stack, GPU if available
	@./deploy.sh --mode full

deploy-app: ## app-only, no LLM
	@./deploy.sh --mode app

down: ## stop the stack
	@./deploy.sh --down

logs: ## tail logs
	@./deploy.sh --logs

status: ## health check + docker compose ps
	@./deploy.sh --status

health: status ## alias for status

model: ## download the LLM model only
	@./deploy.sh --model-only

clean: ## DESTROYS data volumes -- confirmation happens inside deploy.sh
	@echo "WARNING: this deletes all Nestling data (children DB, uploads, HF cache volume)."
	@./deploy.sh --clean
