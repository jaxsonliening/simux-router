import httpx
import asyncio
import time
from typing import Optional, Dict, Any

class RouteDecision:
    def __init__(self, provider: str, reason: str, model_id: str):
        self.provider = provider
        self.reason = reason
        self.model_id = model_id

class SiMuxRouter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Failure tracking: {provider_name: last_failure_timestamp}
        self.health_status = {}
        self.COOLDOWN_SECONDS = 60 

    def _is_healthy(self, provider: str) -> bool:
        """Check if a provider is in a cooling down period after a failure."""
        last_fail = self.health_status.get(provider, 0)
        if time.time() - last_fail < self.COOLDOWN_SECONDS:
            return False
        return True

    def mark_failure(self, provider: str):
        """Temporarily ban a provider from routing."""
        print(f"!! ALERT: Marking {provider} as unhealthy for {self.COOLDOWN_SECONDS}s")
        self.health_status[provider] = time.time()

    def select_route(self, model_slug: str, context_length: int, stream: bool) -> RouteDecision:
        """
        The core logic engine.
        Priority:
        1. Health (Is the provider up?)
        2. Capabilities (Can they handle this context window?)
        3. Performance (Latency vs Throughput)
        """
        
        # 1. Define Candidate Pool
        candidates = ["groq", "cerebras", "sambanova"]
        
        # Filter out unhealthy providers
        active_candidates = [c for c in candidates if self._is_healthy(c)]
        
        # FALLBACK: If everyone is down, try to reset the oldest failure
        if not active_candidates:
            active_candidates = candidates 

        # 2. Capability Filtering (Context Window)
        # SambaNova is the only one we trust for > 16k tokens right now
        if context_length > 16000:
            if "sambanova" in active_candidates:
                return RouteDecision("sambanova", "high_context_specialist", 
                                   self.config["models"][model_slug]["sambanova"])
            else:
                # Critical fallback: If Samba is down, Cerebras is next best for batch
                return RouteDecision("cerebras", "context_fallback", 
                                   self.config["models"][model_slug]["cerebras"])

        # 3. Performance Optimization (Small/Medium Context)
        
        # Case A: Streaming Chat (Low Latency required) -> Prefers Groq
        if stream and context_length < 4000:
            if "groq" in active_candidates:
                return RouteDecision("groq", "latency_optimization", 
                                   self.config["models"][model_slug]["groq"])
        
        # Case B: Batch / Throughput -> Prefers Cerebras
        if "cerebras" in active_candidates:
            return RouteDecision("cerebras", "throughput_optimization", 
                               self.config["models"][model_slug]["cerebras"])
        
        # Case C: Catch-all Fallback (e.g., if Cerebras is down, use Groq)
        fallback = active_candidates[0]
        return RouteDecision(fallback, "availability_fallback", 
                           self.config["models"][model_slug][fallback])