# Diffusion Monitoring Platform - Quick Start Guide

Get started monitoring your diffusion models in 5 minutes!

## Prerequisites

- Python 3.9+
- Docker & Docker Compose (for Prometheus/Grafana)
- PyTorch 2.0+
- (Optional) A diffusion model: LLaDA, Open-dLLM, or custom

## Step 1: Installation

```bash
cd diffusion_monitor

# Install the monitoring library
pip install -e .

# Or install specific extras
pip install -e ".[dev,tensorrt]"
```

## Step 2: Start Observability Stack

```bash
# Start Prometheus + Grafana + AlertManager
docker-compose up -d

# Verify services are running
docker-compose ps

# Expected output:
# diffusion-prometheus    Up    0.0.0.0:9090->9090/tcp
# diffusion-grafana       Up    0.0.0.0:3000->3000/tcp
# diffusion-alertmanager  Up    0.0.0.0:9093->9093/tcp
```

## Step 3: Run Basic Example

```bash
cd examples
python basic_monitoring.py
```

You should see:

```
======================================================================
Diffusion Model Monitoring - Basic Example
======================================================================

[1/5] Starting Prometheus exporter...
✓ Metrics available at: http://localhost:8000/metrics

[2/5] Loading diffusion model...
✓ Model loaded (mock)

[3/5] Attaching monitoring hooks...
✓ Monitor attached with 2 hooks

[4/5] Running monitored inference...

======================================================================
Running monitored inference: mock-diffusion
======================================================================

Step  0: Latency=  5.23ms, Confidence=0.342, Flips=  0, Entropy=8.52
Step  1: Latency=  4.87ms, Confidence=0.456, Flips= 45, Entropy=8.21
Step  2: Latency=  5.01ms, Confidence=0.523, Flips= 32, Entropy=7.98
...
```

## Step 4: View Metrics

### Prometheus UI

Open http://localhost:9090

Try these queries:
```promql
# Average step latency
diffusion_step_latency_seconds

# Confidence per step
diffusion_step_confidence

# Inference rate
rate(diffusion_inference_total[5m])

# P95 latency
histogram_quantile(0.95, rate(diffusion_step_latency_seconds_bucket[5m]))
```

### Raw Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

### Grafana Dashboards

1. Open http://localhost:3000
2. Login: `admin` / `admin`
3. Navigate to Dashboards
4. Import dashboards from `visualization/grafana_dashboards/`

## Step 5: Monitor Real Diffusion Model

### With LLaDA

```python
from diffusion_monitor import DiffusionMonitor
from exporters import PrometheusExporter
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load LLaDA model
model = AutoModelForCausalLM.from_pretrained("microsoft/llada-7b")
tokenizer = AutoTokenizer.from_pretrained("microsoft/llada-7b")

# Start exporter
exporter = PrometheusExporter(port=8000)
exporter.start()

# Attach monitor
monitor = DiffusionMonitor(
    model_name="llada-7b",
    exporters=[exporter],
    track_attention=True
)
monitor.attach(model)

# Run inference
with monitor:
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, num_diffusion_steps=20)
    text = tokenizer.decode(outputs[0])

# Get metrics
metrics = monitor.compute_convergence_metrics()
print(f"Converged: {metrics['converged']}")
print(f"Efficiency: {metrics['efficiency']:.2%}")
print(f"Final confidence: {metrics['final_confidence']:.3f}")

# Visualize
from visualization import visualize_inference
trajectory = monitor.get_denoising_trajectory()
visualize_inference(trajectory, output_dir="./results")
```

### With Custom Model

```python
from diffusion_monitor import DiffusionMonitor

monitor = DiffusionMonitor(model_name="my-diffusion-model")

# Manual step recording
monitor.start_inference()

for step in range(num_diffusion_steps):
    logits = my_model.denoise_step(hidden_state, step)

    # Record metrics
    monitor.record_step(
        step=step,
        logits=logits,
        mask=mask,  # Optional
        attention_weights=attn  # Optional
    )

summary = monitor.end_inference()
```

## Step 6: Set Up Alerts

Edit `config/alerts.yml` to customize alert thresholds:

```yaml
- alert: HighPerplexity
  expr: diffusion_final_perplexity > 50
  for: 1m
  labels:
    severity: warning
```

Reload Prometheus:
```bash
curl -X POST http://localhost:9090/-/reload
```

Configure AlertManager webhooks in `config/alertmanager.yml`:
```yaml
receivers:
  - name: 'slack-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#ml-alerts'
```

## Step 7: Create Visualizations

```python
from visualization import (
    plot_convergence,
    plot_confidence_evolution,
    plot_denoising_animation
)

# Load metrics
trajectory = monitor.get_denoising_trajectory()

# Plot convergence
plot_convergence(trajectory, save_path="convergence.png")

# Confidence evolution with per-token heatmap
plot_confidence_evolution(trajectory, show_tokens=True)

# Animated denoising process (requires ffmpeg)
plot_denoising_animation(trajectory, output_path="denoising.mp4")
```

## Common Use Cases

### 1. Compare Two Model Versions

```python
# Monitor baseline
monitor_v1 = DiffusionMonitor(model_name="llada-v1")
monitor_v1.attach(model_v1)
# ... run inference ...
metrics_v1 = monitor_v1.compute_convergence_metrics()

# Monitor new version
monitor_v2 = DiffusionMonitor(model_name="llada-v2")
monitor_v2.attach(model_v2)
# ... run inference ...
metrics_v2 = monitor_v2.compute_convergence_metrics()

# Compare
print(f"V1 efficiency: {metrics_v1['efficiency']:.2%}")
print(f"V2 efficiency: {metrics_v2['efficiency']:.2%}")
print(f"Improvement: {(metrics_v2['efficiency'] - metrics_v1['efficiency']) * 100:.1f}%")
```

### 2. Production Monitoring

```python
from exporters import PrometheusExporter, DataDogExporter

# Export to multiple platforms
prom_exporter = PrometheusExporter(port=8000)
dd_exporter = DataDogExporter(api_key="YOUR_KEY")

monitor = DiffusionMonitor(
    model_name="prod-llada",
    exporters=[prom_exporter, dd_exporter]
)

# Deploy with your inference server
app = FastAPI()

@app.post("/generate")
async def generate(prompt: str):
    with monitor:
        output = model.generate(prompt)
    return {"text": output}
```

### 3. Debug Quality Issues

```python
monitor.start_inference()

# Run problematic inference
output = model.generate("Problematic prompt")

summary = monitor.end_inference()

# Check for anomalies
for anomaly in summary['anomalies']:
    print(f"Found {anomaly['type']} at step {anomaly['step']}")

# Inspect specific step
problem_step = 15
metrics = monitor.get_step_metrics(problem_step)
print(f"Step {problem_step} confidence: {metrics.avg_confidence}")
print(f"Token flip rate: {metrics.token_flip_rate}")

# Visualize trajectory
trajectory = monitor.get_denoising_trajectory()
plot_denoising_animation(trajectory, "debug_animation.mp4")
```

## Troubleshooting

### Issue: Hooks not capturing metrics

**Solution:** Ensure hook target layers exist in your model:

```python
# List model layers
for name, module in model.named_modules():
    print(name)

# Manually specify layers to hook
monitor.attach(model)
# Verify hooks attached
print(f"Attached {len(monitor.hooks)} hooks")
```

### Issue: Prometheus not scraping

**Solution:** Check network connectivity:

```bash
# From host
curl http://localhost:8000/metrics

# From Prometheus container
docker exec diffusion-prometheus wget -O- http://host.docker.internal:8000/metrics
```

Update `config/prometheus.yml` target if needed.

### Issue: No visualization output

**Solution:** Install visualization dependencies:

```bash
pip install matplotlib seaborn plotly

# For animations
sudo apt install ffmpeg  # Linux
brew install ffmpeg      # macOS
```

## Next Steps

1. **Integrate with CI/CD**: Add monitoring to your model evaluation pipeline
2. **Create Custom Dashboards**: Build Grafana dashboards for your specific metrics
3. **Tune Alert Thresholds**: Adjust alert rules based on your model's behavior
4. **Export to DataDog/New Relic**: Use custom exporters for your observability platform
5. **Contribute**: Add support for new model architectures!

## Resources

- [Architecture Document](../DIFFUSION_MONITORING_ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [LLaDA Paper](https://arxiv.org/abs/2304.xxxxx)
- [Prometheus Docs](https://prometheus.io/docs/)
- [Grafana Docs](https://grafana.com/docs/)

## Getting Help

- GitHub Issues: https://github.com/tcBio/oss_srv/issues
- Discussions: https://github.com/tcBio/oss_srv/discussions
- Email: support@example.com

---

Happy Monitoring! 🚀
