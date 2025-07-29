"""
GPU monitoring utilities using NVML.
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from schemas.base import GPUStatus
from cli.utils.logging import gola_logger

@dataclass
class GPUInfo:
    """GPU information."""
    index: int
    name: str
    memory_total_mb: int
    compute_capability: str

class GPUMonitor:
    """GPU monitoring using NVML."""
    
    def __init__(self, gpu_indices: Optional[List[int]] = None):
        """
        Initialize GPU monitor.
        
        Args:
            gpu_indices: List of GPU indices to monitor (None for all)
        """
        self.gpu_indices = gpu_indices
        self.gpu_info: Dict[int, GPUInfo] = {}
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable[[Dict[int, GPUStatus]], None]] = []
        
        if not NVML_AVAILABLE:
            gola_logger.warning("NVML not available. GPU monitoring disabled.")
            return
        
        try:
            pynvml.nvmlInit()
            self._discover_gpus()
        except Exception as e:
            gola_logger.error(f"Failed to initialize NVML: {e}")
    
    def _discover_gpus(self) -> None:
        """Discover available GPUs."""
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            
            if self.gpu_indices is None:
                self.gpu_indices = list(range(device_count))
            
            for gpu_index in self.gpu_indices:
                if gpu_index >= device_count:
                    gola_logger.warning(f"GPU index {gpu_index} not available")
                    continue
                
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_cap_str = f"{compute_cap[0]}.{compute_cap[1]}"
                
                self.gpu_info[gpu_index] = GPUInfo(
                    index=gpu_index,
                    name=name,
                    memory_total_mb=memory_info.total // (1024 * 1024),
                    compute_capability=compute_cap_str
                )
                
                gola_logger.info(f"Discovered GPU {gpu_index}: {name} "
                               f"({memory_info.total // (1024 * 1024)} MB)")
        
        except Exception as e:
            gola_logger.error(f"Error discovering GPUs: {e}")
    
    def get_gpu_status(self, gpu_index: int) -> Optional[GPUStatus]:
        """
        Get current status of a specific GPU.
        
        Args:
            gpu_index: GPU index
            
        Returns:
            GPU status or None if not available
        """
        if not NVML_AVAILABLE or gpu_index not in self.gpu_info:
            return None
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            
            # Get utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = utilization.gpu
            
            # Get memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_mb = memory_info.used // (1024 * 1024)
            memory_total_mb = memory_info.total // (1024 * 1024)
            
            # Get temperature (if available)
            try:
                temperature_c = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except:
                temperature_c = None
            
            # Get power consumption (if available)
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except:
                power_w = None
            
            return GPUStatus(
                gpu_index=gpu_index,
                utilization=gpu_utilization,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                temperature_c=temperature_c,
                power_w=power_w,
                timestamp=datetime.utcnow()
            )
        
        except Exception as e:
            gola_logger.error(f"Error getting GPU {gpu_index} status: {e}")
            return None
    
    def get_all_gpu_status(self) -> Dict[int, GPUStatus]:
        """
        Get status of all monitored GPUs.
        
        Returns:
            Dictionary of GPU status by index
        """
        status_dict = {}
        
        for gpu_index in self.gpu_indices:
            status = self.get_gpu_status(gpu_index)
            if status:
                status_dict[gpu_index] = status
        
        return status_dict
    
    def add_callback(self, callback: Callable[[Dict[int, GPUStatus]], None]) -> None:
        """
        Add callback for GPU status updates.
        
        Args:
            callback: Function to call with GPU status updates
        """
        self.callbacks.append(callback)
    
    def start_monitoring(self, interval_seconds: float = 5.0) -> None:
        """
        Start continuous GPU monitoring.
        
        Args:
            interval_seconds: Monitoring interval in seconds
        """
        if self.is_monitoring:
            gola_logger.warning("GPU monitoring already started")
            return
        
        if not NVML_AVAILABLE:
            gola_logger.warning("Cannot start GPU monitoring: NVML not available")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        gola_logger.info("GPU monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous GPU monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        gola_logger.info("GPU monitoring stopped")
    
    def _monitor_loop(self, interval_seconds: float) -> None:
        """Monitoring loop."""
        while self.is_monitoring:
            try:
                status_dict = self.get_all_gpu_status()
                
                # Call callbacks
                for callback in self.callbacks:
                    try:
                        callback(status_dict)
                    except Exception as e:
                        gola_logger.error(f"Error in GPU monitoring callback: {e}")
                
                time.sleep(interval_seconds)
            
            except Exception as e:
                gola_logger.error(f"Error in GPU monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def get_best_gpu(self, min_memory_mb: int = 1000) -> Optional[int]:
        """
        Get the best available GPU for processing.
        
        Args:
            min_memory_mb: Minimum required memory in MB
            
        Returns:
            Best GPU index or None if none available
        """
        best_gpu = None
        best_score = -1
        
        for gpu_index in self.gpu_indices:
            status = self.get_gpu_status(gpu_index)
            if not status:
                continue
            
            # Check memory availability
            available_memory = status.memory_total_mb - status.memory_used_mb
            if available_memory < min_memory_mb:
                continue
            
            # Calculate score (lower utilization = better)
            score = available_memory - (status.utilization * 10)
            
            if score > best_score:
                best_score = score
                best_gpu = gpu_index
        
        return best_gpu
    
    def is_gpu_available(self, gpu_index: int, min_memory_mb: int = 1000) -> bool:
        """
        Check if a GPU is available for processing.
        
        Args:
            gpu_index: GPU index
            min_memory_mb: Minimum required memory in MB
            
        Returns:
            True if GPU is available
        """
        status = self.get_gpu_status(gpu_index)
        if not status:
            return False
        
        available_memory = status.memory_total_mb - status.memory_used_mb
        return available_memory >= min_memory_mb and not status.is_overloaded
    
    def get_gpu_summary(self) -> Dict[str, Any]:
        """
        Get summary of all GPUs.
        
        Returns:
            GPU summary dictionary
        """
        summary = {
            "total_gpus": len(self.gpu_indices),
            "available_gpus": 0,
            "overloaded_gpus": 0,
            "gpu_details": {}
        }
        
        for gpu_index in self.gpu_indices:
            status = self.get_gpu_status(gpu_index)
            if not status:
                continue
            
            gpu_info = self.gpu_info.get(gpu_index)
            gpu_detail = {
                "name": gpu_info.name if gpu_info else "Unknown",
                "utilization": status.utilization,
                "memory_used_mb": status.memory_used_mb,
                "memory_total_mb": status.memory_total_mb,
                "memory_utilization": status.memory_utilization,
                "temperature_c": status.temperature_c,
                "power_w": status.power_w,
                "is_overloaded": status.is_overloaded,
                "is_available": self.is_gpu_available(gpu_index)
            }
            
            summary["gpu_details"][gpu_index] = gpu_detail
            
            if gpu_detail["is_available"]:
                summary["available_gpus"] += 1
            
            if gpu_detail["is_overloaded"]:
                summary["overloaded_gpus"] += 1
        
        return summary
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'is_monitoring') and self.is_monitoring:
            self.stop_monitoring()

# Global GPU monitor instance
gpu_monitor = GPUMonitor()

def get_gpu_monitor() -> GPUMonitor:
    """Get the global GPU monitor instance."""
    return gpu_monitor

def init_gpu_monitor(gpu_indices: Optional[List[int]] = None) -> GPUMonitor:
    """
    Initialize the global GPU monitor.
    
    Args:
        gpu_indices: GPU indices to monitor
        
    Returns:
        GPU monitor instance
    """
    global gpu_monitor
    gpu_monitor = GPUMonitor(gpu_indices)
    return gpu_monitor 