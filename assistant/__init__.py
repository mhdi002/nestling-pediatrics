#!/usr/bin/env python3
"""
Pediatrics parent assistant package.

Dual RAG:
  - medical knowledge (translated educational + screening content)
  - per-child precision memory (scores, growth, questionnaires)

Tools (deterministic — no LLM math):
  - INTERGROWTH growth equations
  - ASQ / M-CHAT scoring
  - chart overlay plotting

Models (optional, loaded on demand):
  - Qwen/Qwen3.5-4B via local vLLM OpenAI-compatible sidecar → RAG + chat + vision
  - Salesforce/xLAM-1b-fc-r → tool/function calling (optional)
"""

__version__ = "0.1.0"
