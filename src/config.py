"""API 配置：从环境变量读取，避免密钥进 git。"""
import os

API_BASE = os.environ.get("LLM_API_BASE", "http://113.46.219.251/v1")
API_KEY = os.environ.get("LLM_API_KEY", "")
