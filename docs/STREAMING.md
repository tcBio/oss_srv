# Streaming Inference Guide

OSS_SRV supports token-by-token streaming for real-time inference responses. This is critical for chat applications and interactive UIs.

## Overview

**Streaming inference** generates and returns tokens one at a time as they're produced, rather than waiting for the entire response to complete. This dramatically improves perceived latency and enables real-time user experiences.

### Benefits

- ✅ **Lower perceived latency** - Users see output immediately
- ✅ **Better UX** - Progressive text rendering feels more responsive
- ✅ **Interruptible** - Can stop generation early if needed
- ✅ **Real-time feedback** - Monitor generation progress

### Use Cases

- 🗨️ Chat applications (ChatGPT-style)
- 📝 Content generation tools
- 🤖 Interactive AI assistants
- 🎮 Game dialogue systems

---

## C++ API

### Basic Usage

```cpp
#include "engine_core.hpp"
#include "streaming_server.hpp"

using namespace oss_srv;

// Initialize engine
EngineCore engine;
engine.initialize(config);

// Create stream manager
StreamManager stream_manager;
std::string request_id = "req_001";
stream_manager.createStream(request_id);

// Set up streaming request
InferenceRequest request;
request.prompt = "Once upon a time";
request.max_tokens = 100;
request.stream = true;  // Enable streaming

// Set up callback for each token
request.stream_callback = [&](int32_t token_id, const std::string& token_text, bool is_final) {
    std::cout << token_text << std::flush;

    if (is_final) {
        std::cout << "\n[Done]" << std::endl;
    }
};

// Execute (tokens stream via callback)
auto result = engine.executeInference(request);
```

### Advanced: SSE (Server-Sent Events)

```cpp
#include "streaming_server.hpp"

// Create SSE callback
auto sse_callback = StreamingHelper::createSSECallback(stream_manager, request_id);
request.stream_callback = sse_callback;

// Execute inference
std::thread inference_thread([&]() {
    engine.executeInference(request);
});

// Read SSE stream
while (true) {
    StreamManager::StreamToken token;
    if (stream_manager.popToken(request_id, token)) {
        // Convert to SSE message
        std::string sse_msg = StreamingHelper::tokenToSSE(
            token.token_id,
            token.token_text,
            token.is_final
        );

        // Send to client (HTTP response)
        // response_stream << sse_msg;

        if (token.is_final) break;
    }
}
```

---

## Python SDK

### Synchronous Streaming

```python
from oss_srv import StreamingInferenceClient

client = StreamingInferenceClient(model_path="model.engine")

# Stream tokens as they're generated
for token in client.stream("Once upon a time", max_tokens=100):
    print(token.text, end='', flush=True)

print()  # Newline at end
```

### With Callback

```python
def on_token(token):
    """Called for each generated token"""
    print(f"[{token.token_id}] {token.text}", end='')
    if token.is_final:
        print(" [DONE]")

# Stream with callback
for token in client.stream("Hello world", callback=on_token):
    pass  # Callback handles display
```

### Async Streaming

```python
import asyncio
from oss_srv import StreamingInferenceClient

client = StreamingInferenceClient(model_path="model.engine")

async def stream_example():
    async for token in client.stream_async("Once upon a time"):
        print(token.text, end='', flush=True)

asyncio.run(stream_example())
```

---

## SSE HTTP API

When the HTTP server is deployed, use Server-Sent Events:

### Request

```bash
curl -N http://localhost:8000/v1/completions/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "stream": true
  }'
```

### Response (SSE Format)

```
event: token
data: {"token_id": 1234, "text": "Once"}

event: token
data: {"token_id": 5678, "text": " upon"}

event: token
data: {"token_id": 9012, "text": " a"}

event: done
data: {"finish_reason": "stop"}
```

### Python Client for SSE

```python
from oss_srv import SSEStreamingClient
import json

client = SSEStreamingClient(base_url="http://localhost:8000")

for event in client.stream("/v1/completions/stream", data={
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "stream": true
}):
    if event['event'] == 'token':
        data = json.loads(event['data'])
        print(data['text'], end='', flush=True)
    elif event['event'] == 'done':
        print()
        break
```

---

## WebSocket API (Future)

For bidirectional streaming:

```python
from oss_srv import WebSocketStreamingClient

client = WebSocketStreamingClient(url="ws://localhost:8000/ws")
client.connect()

for token in client.stream({"prompt": "Hello"}):
    print(token, end='', flush=True)

client.close()
```

---

## Performance Characteristics

### Metrics

| Metric | Standard | Streaming |
|--------|----------|-----------|
| **Time to First Token** | Same | Same |
| **Perceived Latency** | High | Low |
| **User Experience** | Waiting | Interactive |
| **Overhead** | ~0% | <5% (callback) |

### Throughput

Streaming adds minimal overhead (<5%) due to per-token callbacks. Total throughput remains the same, but UX is dramatically better.

---

## Best Practices

### ✅ Do

- Use streaming for chat and interactive UIs
- Buffer tokens for display smoothness
- Handle network errors gracefully
- Implement timeout mechanisms

### ❌ Don't

- Use streaming for batch processing (use batch API instead)
- Stream very short responses (<10 tokens)
- Forget to handle the `is_final` flag
- Block on callback execution (keep it fast)

---

## Example: Chat UI

```python
from oss_srv import StreamingInferenceClient

client = StreamingInferenceClient(model_path="model.engine")

def chat_loop():
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        print("Bot: ", end='', flush=True)

        # Stream response
        for token in client.stream(user_input, max_tokens=200):
            print(token.text, end='', flush=True)

        print()  # Newline

chat_loop()
```

---

## Troubleshooting

### Tokens arrive slowly

**Cause:** Model inference is the bottleneck, not streaming.

**Solution:** Use smaller model or speculative decoding for speedup.

### Buffering issues

**Cause:** Output buffering in terminal/HTTP.

**Solution:** Flush after each token:
```python
print(token.text, end='', flush=True)
```

### Missing tokens

**Cause:** Callback exception or network interruption.

**Solution:** Add error handling:
```cpp
request.stream_callback = [](int32_t token_id, const std::string& text, bool final) {
    try {
        // Your code
    } catch (const std::exception& e) {
        std::cerr << "Callback error: " << e.what() << std::endl;
    }
};
```

---

## Related Features

- **Speculative Decoding** - 2-3x speedup for faster token generation
- **Batch Inference** - Process multiple requests concurrently
- **Dynamic Batching** - Continuous request aggregation

---

## Further Reading

- [HTTP Server Setup](HTTP_SERVER.md) (coming soon)
- [WebSocket Guide](WEBSOCKET.md) (coming soon)
- [API Reference](API.md)
