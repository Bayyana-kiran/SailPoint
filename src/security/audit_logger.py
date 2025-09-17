"""
Audit Logger
Comprehensive audit logging for security and compliance.
"""

import json
import time
import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import threading
from pathlib import Path

from config.security import SECURITY_CONFIG

class AuditEventType(Enum):
    """Types of audit events."""
    QUERY_EXECUTED = "query_executed"
    QUERY_BLOCKED = "query_blocked"
    QUERY_FAILED = "query_failed"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    SCHEMA_ACCESSED = "schema_accessed"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_ERROR = "system_error"
    CONFIG_CHANGED = "config_changed"

@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_type: AuditEventType
    timestamp: float
    user_id: str
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    query: Optional[str]
    table_names: Optional[List[str]]
    result_count: Optional[int]
    execution_time: Optional[float]
    success: bool
    error_message: Optional[str]
    security_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    metadata: Optional[Dict[str, Any]]

class AuditLogger:
    """
    Comprehensive audit logging system with security event tracking.
    """
    
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        # Configure audit logger
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # File handler for audit logs
        audit_file = self.log_directory / "audit.log"
        file_handler = logging.FileHandler(audit_file)
        file_handler.setLevel(logging.INFO)
        
        # Security event handler (separate file)
        security_file = self.log_directory / "security.log"
        self.security_handler = logging.FileHandler(security_file)
        self.security_handler.setLevel(logging.WARNING)
        
        # JSON formatter
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        self.security_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(self.security_handler)
        
        self._lock = threading.RLock()
        
        # Event counters
        self.event_counters = {
            event_type: 0 for event_type in AuditEventType
        }
    
    def log_query_execution(
        self,
        user_id: str,
        query: str,
        table_names: List[str],
        success: bool,
        result_count: Optional[int] = None,
        execution_time: Optional[float] = None,
        error_message: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Log SQL query execution."""
        event_type = AuditEventType.QUERY_EXECUTED if success else AuditEventType.QUERY_FAILED
        security_level = "LOW" if success else "MEDIUM"
        
        event = AuditEvent(
            event_type=event_type,
            timestamp=time.time(),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            query=query,
            table_names=table_names,
            result_count=result_count,
            execution_time=execution_time,
            success=success,
            error_message=error_message,
            security_level=security_level,
            metadata=None
        )
        
        self._log_event(event)
    
    def log_query_blocked(
        self,
        user_id: str,
        query: str,
        reason: str,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Log blocked query attempt."""
        event = AuditEvent(
            event_type=AuditEventType.QUERY_BLOCKED,
            timestamp=time.time(),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            query=query,
            table_names=None,
            result_count=None,
            execution_time=None,
            success=False,
            error_message=reason,
            security_level="HIGH",
            metadata={"block_reason": reason}
        )
        
        self._log_event(event)
    
    def log_security_violation(
        self,
        user_id: str,
        violation_type: str,
        details: str,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log security violation."""
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_VIOLATION,
            timestamp=time.time(),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=None,
            query=None,
            table_names=None,
            result_count=None,
            execution_time=None,
            success=False,
            error_message=details,
            security_level="CRITICAL",
            metadata={
                "violation_type": violation_type,
                **(metadata or {})
            }
        )
        
        self._log_event(event)
    
    def log_rate_limit_exceeded(
        self,
        user_id: str,
        limit_type: str,
        current_count: int,
        limit: int,
        ip_address: Optional[str] = None
    ):
        """Log rate limit violation."""
        event = AuditEvent(
            event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
            timestamp=time.time(),
            user_id=user_id,
            session_id=None,
            ip_address=ip_address,
            user_agent=None,
            query=None,
            table_names=None,
            result_count=None,
            execution_time=None,
            success=False,
            error_message=f"Rate limit exceeded: {current_count}/{limit} for {limit_type}",
            security_level="HIGH",
            metadata={
                "limit_type": limit_type,
                "current_count": current_count,
                "limit": limit
            }
        )
        
        self._log_event(event)
    
    def log_schema_access(
        self,
        user_id: str,
        tables_accessed: List[str],
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ):
        """Log database schema access."""
        event = AuditEvent(
            event_type=AuditEventType.SCHEMA_ACCESSED,
            timestamp=time.time(),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=None,
            query=None,
            table_names=tables_accessed,
            result_count=None,
            execution_time=None,
            success=True,
            error_message=None,
            security_level="LOW",
            metadata={"access_type": "schema_metadata"}
        )
        
        self._log_event(event)
    
    def log_user_session(
        self,
        user_id: str,
        action: str,  # 'login' or 'logout'
        session_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True
    ):
        """Log user session events."""
        event_type = AuditEventType.USER_LOGIN if action == 'login' else AuditEventType.USER_LOGOUT
        
        event = AuditEvent(
            event_type=event_type,
            timestamp=time.time(),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            query=None,
            table_names=None,
            result_count=None,
            execution_time=None,
            success=success,
            error_message=None if success else f"Failed {action}",
            security_level="LOW" if success else "MEDIUM",
            metadata={"action": action}
        )
        
        self._log_event(event)
    
    def log_system_error(
        self,
        error_type: str,
        error_message: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log system errors."""
        event = AuditEvent(
            event_type=AuditEventType.SYSTEM_ERROR,
            timestamp=time.time(),
            user_id=user_id or "system",
            session_id=None,
            ip_address=None,
            user_agent=None,
            query=None,
            table_names=None,
            result_count=None,
            execution_time=None,
            success=False,
            error_message=error_message,
            security_level="MEDIUM",
            metadata={
                "error_type": error_type,
                **(metadata or {})
            }
        )
        
        self._log_event(event)
    
    def _log_event(self, event: AuditEvent):
        """Internal method to log audit event."""
        with self._lock:
            # Update counters
            self.event_counters[event.event_type] += 1
            
            # Convert to JSON
            event_dict = asdict(event)
            event_dict['event_type'] = event.event_type.value
            event_dict['timestamp_iso'] = datetime.fromtimestamp(event.timestamp).isoformat()
            
            event_json = json.dumps(event_dict, default=str)
            
            # Log to appropriate handler based on security level
            if event.security_level in ["HIGH", "CRITICAL"]:
                self.security_handler.handle(
                    logging.LogRecord(
                        name="security",
                        level=logging.WARNING,
                        pathname="",
                        lineno=0,
                        msg=event_json,
                        args=(),
                        exc_info=None
                    )
                )
            
            # Always log to main audit log
            self.logger.info(event_json)
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics."""
        with self._lock:
            total_events = sum(self.event_counters.values())
            
            stats = {
                "total_events": total_events,
                "event_counts": {
                    event_type.value: count
                    for event_type, count in self.event_counters.items()
                },
                "log_directory": str(self.log_directory),
                "audit_enabled": SECURITY_CONFIG.get('log_all_queries', True)
            }
            
            # File sizes
            try:
                audit_file = self.log_directory / "audit.log"
                security_file = self.log_directory / "security.log"
                
                if audit_file.exists():
                    stats["audit_log_size"] = audit_file.stat().st_size
                if security_file.exists():
                    stats["security_log_size"] = security_file.stat().st_size
            except Exception:
                pass
            
            return stats
    
    def search_audit_logs(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        security_level: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search audit logs with filters."""
        results = []
        
        try:
            audit_file = self.log_directory / "audit.log"
            if not audit_file.exists():
                return results
            
            with open(audit_file, 'r') as f:
                for line in f:
                    if len(results) >= limit:
                        break
                    
                    try:
                        event_data = json.loads(line.strip())
                        
                        # Apply filters
                        if start_time and event_data['timestamp'] < start_time:
                            continue
                        if end_time and event_data['timestamp'] > end_time:
                            continue
                        if user_id and event_data['user_id'] != user_id:
                            continue
                        if event_type and event_data['event_type'] != event_type.value:
                            continue
                        if security_level and event_data['security_level'] != security_level:
                            continue
                        
                        results.append(event_data)
                        
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            self.log_system_error("audit_search_error", str(e))
        
        return results

# Global audit logger instance
audit_logger = AuditLogger()
