"""
Rate Limiter
Implements rate limiting for API requests to prevent abuse.
"""

import time
import logging
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import threading

from config.security import SECURITY_CONFIG

logger = logging.getLogger(__name__)

class RateLimitResult(Enum):
    """Rate limit check results."""
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    WARNING = "warning"

@dataclass
class RateLimitInfo:
    """Rate limit information."""
    result: RateLimitResult
    remaining_requests: int
    reset_time: float
    message: str

class RateLimiter:
    """
    Token bucket rate limiter with multiple time windows.
    Implements per-user and global rate limiting.
    """
    
    def __init__(self):
        self._user_requests = defaultdict(lambda: {
            'minute': deque(),
            'hour': deque(),
            'day': deque()
        })
        self._global_requests = {
            'minute': deque(),
            'hour': deque(),
            'day': deque()
        }
        self._lock = threading.RLock()
        
        # Rate limits from configuration
        self.limits = {
            'minute': SECURITY_CONFIG['max_queries_per_minute'],
            'hour': SECURITY_CONFIG['max_queries_per_hour'],
            'day': SECURITY_CONFIG['max_queries_per_day']
        }
        
        # Time windows in seconds
        self.windows = {
            'minute': 60,
            'hour': 3600,
            'day': 86400
        }
    
    def check_rate_limit(self, user_id: str = "anonymous") -> RateLimitInfo:
        """
        Check if request is within rate limits.
        
        Args:
            user_id: User identifier for per-user limiting
            
        Returns:
            RateLimitInfo with rate limit status
        """
        current_time = time.time()
        
        with self._lock:
            # Clean old requests
            self._cleanup_old_requests(current_time)
            
            # Check each time window
            for window, limit in self.limits.items():
                if limit <= 0:  # Skip if limit is disabled
                    continue
                
                user_count = len(self._user_requests[user_id][window])
                global_count = len(self._global_requests[window])
                
                # Check user-specific limit
                if user_count >= limit:
                    reset_time = current_time + self.windows[window]
                    return RateLimitInfo(
                        result=RateLimitResult.BLOCKED,
                        remaining_requests=0,
                        reset_time=reset_time,
                        message=f"User rate limit exceeded for {window} ({user_count}/{limit})"
                    )
                
                # Check global limit (10x user limit)
                global_limit = limit * 10
                if global_count >= global_limit:
                    reset_time = current_time + self.windows[window]
                    return RateLimitInfo(
                        result=RateLimitResult.BLOCKED,
                        remaining_requests=0,
                        reset_time=reset_time,
                        message=f"Global rate limit exceeded for {window} ({global_count}/{global_limit})"
                    )
            
            # All checks passed - calculate remaining requests
            minute_remaining = self.limits['minute'] - len(self._user_requests[user_id]['minute'])
            
            return RateLimitInfo(
                result=RateLimitResult.ALLOWED,
                remaining_requests=minute_remaining,
                reset_time=current_time + 60,  # Next minute reset
                message="Request allowed"
            )
    
    def record_request(self, user_id: str = "anonymous") -> bool:
        """
        Record a request for rate limiting.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if request was recorded successfully
        """
        current_time = time.time()
        
        with self._lock:
            # Check rate limit first
            rate_limit_info = self.check_rate_limit(user_id)
            
            if rate_limit_info.result == RateLimitResult.BLOCKED:
                logger.warning(f"Rate limit exceeded for user {user_id}: {rate_limit_info.message}")
                return False
            
            # Record request in all time windows
            for window in self.windows:
                self._user_requests[user_id][window].append(current_time)
                self._global_requests[window].append(current_time)
            
            logger.debug(f"Request recorded for user {user_id}")
            return True
    
    def _cleanup_old_requests(self, current_time: float):
        """Remove old requests outside time windows."""
        # Clean user requests
        for user_id, windows in self._user_requests.items():
            for window, requests in windows.items():
                window_duration = self.windows[window]
                cutoff_time = current_time - window_duration
                
                while requests and requests[0] < cutoff_time:
                    requests.popleft()
        
        # Clean global requests
        for window, requests in self._global_requests.items():
            window_duration = self.windows[window]
            cutoff_time = current_time - window_duration
            
            while requests and requests[0] < cutoff_time:
                requests.popleft()
    
    def get_usage_stats(self, user_id: str = "anonymous") -> Dict[str, int]:
        """Get current usage statistics for a user."""
        current_time = time.time()
        
        with self._lock:
            self._cleanup_old_requests(current_time)
            
            stats = {}
            for window in self.windows:
                user_count = len(self._user_requests[user_id][window])
                global_count = len(self._global_requests[window])
                limit = self.limits[window]
                
                stats[f"{window}_user_requests"] = user_count
                stats[f"{window}_global_requests"] = global_count
                stats[f"{window}_limit"] = limit
                stats[f"{window}_remaining"] = max(0, limit - user_count)
        
        return stats
    
    def reset_user_limits(self, user_id: str):
        """Reset rate limits for a specific user."""
        with self._lock:
            if user_id in self._user_requests:
                del self._user_requests[user_id]
                logger.info(f"Rate limits reset for user {user_id}")
    
    def get_global_stats(self) -> Dict[str, int]:
        """Get global rate limiting statistics."""
        current_time = time.time()
        
        with self._lock:
            self._cleanup_old_requests(current_time)
            
            stats = {
                'total_users': len(self._user_requests),
                'active_users_minute': len([
                    uid for uid, windows in self._user_requests.items()
                    if len(windows['minute']) > 0
                ])
            }
            
            for window in self.windows:
                stats[f"global_requests_{window}"] = len(self._global_requests[window])
                stats[f"limit_{window}"] = self.limits[window]
        
        return stats

# Global rate limiter instance
rate_limiter = RateLimiter()
