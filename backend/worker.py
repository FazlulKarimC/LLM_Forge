"""
RQ Worker Script

Starts RQ worker with proper environment loading and heartbeat support.
Run with: python worker.py

The worker is an *optional acceleration layer*. When it is not running,
the API falls back to inline execution via FastAPI BackgroundTasks.
A daemon heartbeat thread writes a row to the worker_heartbeats table
every 30s so the API can detect worker liveness.
"""

import os
import sys
import socket
import threading
import time
from urllib.parse import urlparse

from dotenv import load_dotenv

# Load .env before importing app modules
load_dotenv()

from redis import Redis
from rq import Worker, Queue


def mask_redis_url(url: str) -> str:
    """
    Mask credentials in Redis URL for safe logging.
    
    Args:
        url: Full Redis URL potentially containing credentials
        
    Returns:
        URL with password masked
    """
    parsed = urlparse(url)
    if parsed.password is not None:
        masked = url.replace(parsed.password, "***")
        return masked
    return url


def _heartbeat_loop(worker_id: str, interval: int) -> None:
    """
    Background thread that upserts a heartbeat row every *interval* seconds.

    Uses asyncio.run() per tick (via the sync wrapper) so we don't need
    a separate sync DB engine or psycopg2 dependency.
    """
    from app.core.worker_heartbeat import touch_worker_heartbeat_sync

    while True:
        try:
            touch_worker_heartbeat_sync(
                worker_id=worker_id,
                backend="rq",
                queue_name="experiments",
                hostname=socket.gethostname(),
            )
        except Exception as exc:
            print(f"⚠️  Heartbeat write failed: {exc}")
        time.sleep(interval)


def main():
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        print("❌ REDIS_URL not set in environment")
        sys.exit(1)
    
    # Log safely without exposing credentials
    safe_url = mask_redis_url(redis_url)
    print(f"🚀 Starting RQ worker...")
    print(f"   Redis: {safe_url}")
    
    # Connect to Redis
    conn = Redis.from_url(redis_url)
    
    # Create queue and worker
    queues = [Queue("experiments", connection=conn)]
    worker = Worker(queues, connection=conn)

    # --- Heartbeat ---
    from app.core.config import settings

    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    heartbeat_interval = settings.RQ_WORKER_HEARTBEAT_INTERVAL_SECONDS

    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(worker_id, heartbeat_interval),
        daemon=True,   # dies when the main process exits
    )
    heartbeat_thread.start()
    print(f"💓 Heartbeat registered: {worker_id} (every {heartbeat_interval}s)")
    
    print(f"✅ Worker ready, listening on 'experiments' queue")
    worker.work()


if __name__ == "__main__":
    main()
