import time
import torch
from typing import Optional, Dict, Union

class PerformanceTracker:
    """
    A context manager to track latency and peak GPU memory usage for a code block.
    
    This version uses PyTorch's internal memory tracker for higher accuracy.
    """
    def __init__(self, device: str):
        self.device = device
        self.start_time: Optional[float] = None
        self.device_index = None
        
        if self.device == "cuda" and torch.cuda.is_available():
            # Get the integer index of the current CUDA device
            self.device_index = torch.cuda.current_device()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def start(self):
        """Resets memory stats and starts the timer."""
        if self.device_index is not None:
            # Reset the peak memory counter for the current device
            torch.cuda.reset_peak_memory_stats(self.device_index)
        
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self, num_new_tokens: int = 0) -> Dict:
        """
        Stops the timer and calculates performance metrics.

        Args:
            num_new_tokens (int): The number of tokens generated in the call.
        
        Returns:
            A dictionary with latency, memory usage, and throughput.
        """

        if self.start_time is None:
            raise RuntimeError("Tracker was not started.")

        torch.cuda.synchronize()
        latency = (time.perf_counter() - self.start_time) * 1000
        
        peak_mem_used_mb = 0.0
        if self.device_index is not None:
            # Get the peak memory allocated since the last reset
            peak_mem_bytes = torch.cuda.max_memory_allocated(self.device_index)
            peak_mem_used_mb = peak_mem_bytes / (1024 ** 2)

        # Calculate throughput (tokens per second)
        if latency > 0 and num_new_tokens > 0:
            tokens_per_second = (num_new_tokens / latency) * 1000
        else:
            tokens_per_second = 0

        return {
            "latency_ms": latency,
            "peak_gpu_mem_used_mb": peak_mem_used_mb,
            "tokens_per_second": tokens_per_second,
        }


if __name__ == "__main__":
    
    def run_gpu_test():
        if not torch.cuda.is_available():
            print("CUDA is not available. Skipping GPU test.")
            return

        print("--- Running PerformanceTracker Test ---")
        
        with PerformanceTracker(device="cuda") as tracker:
            print("Simulating a workload...")
            tracker.start()

            # Simulate Work
            tensor = torch.randn((4096, 4096), device="cuda")
            time.sleep(1)

            metrics = tracker.stop()

            print("\n--- Test Results ---")
            print(f"Latency: {metrics['latency_ms']:.2f} ms")
            print(f"Peak GPU Memory Used: {metrics['gpu_mem_used_mb']:.2f} MB")

        print("--------------------")

    run_gpu_test()