"""
data_pipeline/storage/redis_pool.py
====================================
Module-level shared Redis connection pool.
All feature engines import `redis_client` from here — never instantiate
their own redis.Redis(). This caps total FD usage to max_connections=20.
"""
from __future__ import annotations
import os
import redis
from loguru import logger
from dotenv import load_dotenv

load_dotenv('/workspaces/Trading_bot/.env')

_pool = redis.ConnectionPool(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    password=os.getenv("REDIS_PASSWORD") or None,
    decode_responses=True,
    max_connections=20,          # hard cap on open sockets
    socket_keepalive=True,
    socket_keepalive_options={},
    health_check_interval=30,    # drops stale connections
)

redis_client = redis.Redis(connection_pool=_pool)
logger.info("Redis | Shared pool initialised (max_connections=20)")
