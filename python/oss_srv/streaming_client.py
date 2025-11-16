"""
OSS_SRV Streaming Client
Support for streaming token-by-token responses

Note: Currently requires HTTP/SSE server implementation
      This is a client-side implementation ready for when server is deployed
"""

import subprocess
import threading
import queue
import time
from typing import Iterator, Optional, Callable
from dataclasses import dataclass
from .client import InferenceClient, InferenceConfig


@dataclass
class StreamingToken:
    """Single token in a streaming response"""
    token_id: int
    text: str
    is_final: bool
    timestamp: float


class StreamingInferenceClient:
    """
    Streaming inference client for token-by-token responses

    Example:
        >>> client = StreamingInferenceClient(model_path="model.engine")
        >>> for token in client.stream("Once upon a time"):
        ...     print(token.text, end='', flush=True)
    """

    def __init__(self, model_path: str, executable_path: Optional[str] = None):
        """Initialize streaming client"""
        self.base_client = InferenceClient(model_path, executable_path)
        self._token_queue = queue.Queue()
        self._stop_event = threading.Event()

    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        config: Optional[InferenceConfig] = None,
        callback: Optional[Callable[[StreamingToken], None]] = None
    ) -> Iterator[StreamingToken]:
        """
        Stream completion tokens as they're generated

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            config: InferenceConfig object
            callback: Optional callback function called for each token

        Yields:
            StreamingToken objects as they're generated
        """
        if config is None:
            config = InferenceConfig()

        # Override config with parameters
        final_max_tokens = max_tokens if max_tokens is not None else config.max_tokens
        final_temperature = temperature if temperature is not None else config.temperature
        final_top_p = top_p if top_p is not None else config.top_p

        # For now, we'll simulate streaming by running inference and yielding tokens
        # In production, this would connect to an SSE/WebSocket endpoint

        # TODO: When HTTP server is ready, use SSE endpoint instead
        # For now, use subprocess and parse output line-by-line

        result = self.base_client.complete(
            prompt=prompt,
            max_tokens=final_max_tokens,
            temperature=final_temperature,
            top_p=final_top_p,
            config=config
        )

        if not result.success:
            raise Exception(f"Streaming inference failed: {result.error}")

        # Simulate streaming by splitting the generated text
        # In a real implementation, this would come from SSE events
        words = result.text.split()
        for i, word in enumerate(words):
            token = StreamingToken(
                token_id=i,  # Simulated
                text=word + " ",
                is_final=(i == len(words) - 1),
                timestamp=time.time()
            )

            if callback:
                callback(token)

            yield token

    async def stream_async(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        config: Optional[InferenceConfig] = None
    ) -> Iterator[StreamingToken]:
        """
        Async version of stream()

        TODO: Implement true async streaming with aiohttp when server is ready
        """
        import asyncio

        # For now, run sync version in executor
        loop = asyncio.get_event_loop()

        for token in await loop.run_in_executor(
            None,
            lambda: list(self.stream(prompt, max_tokens, temperature, top_p, config))
        ):
            yield token


class SSEStreamingClient:
    """
    SSE (Server-Sent Events) streaming client

    Example:
        >>> client = SSEStreamingClient(base_url="http://localhost:8000")
        >>> for event in client.stream("/v1/completions/stream", data={"prompt": "Hello"}):
        ...     print(event.data)
    """

    def __init__(self, base_url: str):
        """
        Initialize SSE client

        Args:
            base_url: Base URL of the inference server
        """
        self.base_url = base_url.rstrip('/')

    def stream(
        self,
        endpoint: str,
        data: dict,
        timeout: int = 120
    ) -> Iterator[dict]:
        """
        Stream SSE events from endpoint

        Args:
            endpoint: API endpoint (e.g., "/v1/completions/stream")
            data: Request payload
            timeout: Request timeout in seconds

        Yields:
            Parsed SSE event dictionaries
        """
        # TODO: Implement when HTTP server is ready
        # This would use requests or aiohttp with streaming

        try:
            import requests
        except ImportError:
            raise ImportError("requests library required for SSE streaming")

        url = f"{self.base_url}{endpoint}"

        with requests.post(url, json=data, stream=True, timeout=timeout) as response:
            response.raise_for_status()

            # Parse SSE format
            event_type = None
            event_data = []

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    # Empty line signals end of event
                    if event_data:
                        yield {
                            'event': event_type or 'message',
                            'data': '\n'.join(event_data)
                        }
                        event_type = None
                        event_data = []
                    continue

                if line.startswith('event:'):
                    event_type = line[6:].strip()
                elif line.startswith('data:'):
                    event_data.append(line[5:].strip())


class WebSocketStreamingClient:
    """
    WebSocket streaming client for bidirectional communication

    Example:
        >>> client = WebSocketStreamingClient(url="ws://localhost:8000/ws")
        >>> client.connect()
        >>> for token in client.stream({"prompt": "Hello"}):
        ...     print(token)
    """

    def __init__(self, url: str):
        """
        Initialize WebSocket client

        Args:
            url: WebSocket URL
        """
        self.url = url
        self.ws = None

    def connect(self):
        """Connect to WebSocket server"""
        # TODO: Implement when WebSocket server is ready
        try:
            import websocket
        except ImportError:
            raise ImportError("websocket-client library required for WebSocket streaming")

        # self.ws = websocket.create_connection(self.url)

    def stream(self, data: dict) -> Iterator[dict]:
        """
        Stream messages via WebSocket

        Args:
            data: Request payload

        Yields:
            Response messages
        """
        # TODO: Implement when WebSocket server is ready
        import json

        if not self.ws:
            raise RuntimeError("Not connected. Call connect() first.")

        # self.ws.send(json.dumps(data))
        # while True:
        #     message = self.ws.recv()
        #     if not message:
        #         break
        #     yield json.loads(message)

    def close(self):
        """Close WebSocket connection"""
        if self.ws:
            self.ws.close()
            self.ws = None
