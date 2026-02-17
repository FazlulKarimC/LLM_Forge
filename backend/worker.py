"""
RQ Worker Script

Starts RQ worker with proper environment loading.
Run with: python worker.py
"""

import os
import sys
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
    if parsed.password:
        # Mask the password
        masked = url.replace(parsed.password, "***")
        return masked
    return url


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
    
    print(f"✅ Worker ready, listening on 'experiments' queue")
    worker.work()


if __name__ == "__main__":
    main()
