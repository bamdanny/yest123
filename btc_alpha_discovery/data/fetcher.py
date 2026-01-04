"""
Base Data Fetcher
================

Handles rate limiting, retries, and common fetch logic.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, calls_per_minute: int):
        self.rate = calls_per_minute / 60.0  # calls per second
        self.tokens = calls_per_minute
        self.max_tokens = calls_per_minute
        self.last_update = time.time()
    
    def _update_tokens(self):
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
        self.last_update = now
    
    def acquire(self):
        """Blocking acquire for sync code"""
        while True:
            self._update_tokens()
            if self.tokens >= 1:
                self.tokens -= 1
                return
            sleep_time = (1 - self.tokens) / self.rate
            time.sleep(sleep_time)


def retry_with_backoff(max_retries: int = 5, base_delay: float = 1.0):
    """Decorator for retry with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, ConnectionError, TimeoutError) as e:
                    last_exception = e
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


class BaseFetcher(ABC):
    """Abstract base class for all data fetchers"""
    
    def __init__(self, base_url: str, rate_limit: int, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.rate_limiter = RateLimiter(rate_limit)
        
        # Setup session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Return headers for API requests"""
        pass
    
    @retry_with_backoff(max_retries=5)
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make GET request with rate limiting and retry"""
        self.rate_limiter.acquire()
        
        url = f"{self.base_url}{endpoint}"
        headers = self.get_headers()
        
        logger.debug(f"GET {url} params={params}")
        
        response = self.session.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        return response.json()
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test API connectivity"""
        pass
    
    def close(self):
        """Close session"""
        self.session.close()


@dataclass
class DataPoint:
    """Standard data point structure"""
    timestamp: datetime
    data: Dict[str, Any]
    source: str


def timestamp_to_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp"""
    return int(dt.timestamp() * 1000)


def ms_to_timestamp(ms: int) -> datetime:
    """Convert milliseconds timestamp to datetime"""
    return datetime.fromtimestamp(ms / 1000)


def align_timestamps(data: List[Dict], target_interval: str) -> List[Dict]:
    """
    Align data points to target interval boundaries.
    Handles data from different sources with different timestamps.
    """
    interval_seconds = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400,
    }
    
    if target_interval not in interval_seconds:
        raise ValueError(f"Unknown interval: {target_interval}")
    
    interval_s = interval_seconds[target_interval]
    
    aligned = []
    for point in data:
        ts = point.get('timestamp')
        if isinstance(ts, datetime):
            ts = ts.timestamp()
        
        # Floor to interval boundary
        aligned_ts = (ts // interval_s) * interval_s
        aligned_point = point.copy()
        aligned_point['timestamp'] = datetime.fromtimestamp(aligned_ts)
        aligned.append(aligned_point)
    
    return aligned
