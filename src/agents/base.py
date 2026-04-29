"""Base protocol for agents."""

from typing import Protocol, Any, Awaitable


class AgentProtocol(Protocol):
    """Protocol defining the interface for agents."""
    
    async def run(self, *args: Any, **kwargs: Any) -> Awaitable[Any]:
        """Execute the agent and return a result."""
        ...

