# Diffusion Model Monitoring Platform

Production-grade monitoring and visualization platform for diffusion-based language models (LLaDA, Open-dLLM, etc.).

## Features

- **Step-by-Step Denoising Visualization**: Watch token predictions evolve across diffusion steps
- **Convergence Tracking**: Monitor quality metrics and step efficiency in real-time
- **Intelligent Alerting**: Detect failures, quality degradation, and performance issues
- **Observability Integration**: Export to Prometheus, Grafana, DataDog, and more
- **Production-Ready**: Low overhead (<5%), scalable, and reliable

## Quick Start

### Installation

```bash
# Install from source
cd diffusion_monitor
pip install -e .

# Or install from PyPI (when published)
pip install diffusion-monitor
```

### Basic Usage

```python
from diffusion_monitor import DiffusionMonitor
from transformers import AutoModelForCausalLM

# Load your diffusion model (e.g., LLaDA)
model = AutoModelForCausalLM.from_pretrained("llada-7b")

# Attach monitor
monitor = DiffusionMonitor(model_name="llada-7b")
monitor.attach(model)

# Run inference
output = model.generate("Once upon a time", max_length=50)

# Get metrics
metrics = monitor.compute_convergence_metrics()
print(f"Convergence efficiency: {metrics['convergence_efficiency']:.2%}")
print(f"Average token flip rate: {metrics['avg_token_flip_rate']:.3f}")

# Visualize denoising process
from diffusion_monitor.visualization import plot_denoising_animation
trajectory = monitor.get_denoising_trajectory()
plot_denoising_animation(trajectory, output_path="denoising.mp4")

# Clean up
monitor.detach()
```

### Prometheus Integration

```python
from diffusion_monitor.exporters import PrometheusExporter

# Start Prometheus exporter
exporter = PrometheusExporter(port=8000)
exporter.start()  # Metrics available at http://localhost:8000/metrics

# Attach to monitor
monitor = DiffusionMonitor(
    model_name="llada-7b",
    exporters=[exporter]
)
```

### Grafana Dashboards

```bash
# Start local Prometheus + Grafana
docker-compose up -d

# Import dashboards
# Navigate to http://localhost:3000
# Import dashboards from visualization/grafana_dashboards/
```

## Project Structure

```
diffusion_monitor/
├── diffusion_monitor/          # Core instrumentation
│   ├── __init__.py
│   ├── hooks.py               # PyTorch hooks
│   ├── metrics_collector.py  # Metric extraction
│   ├── state_tracker.py      # State tracking
│   └── tensorrt_bridge.cpp   # TensorRT integration
├── monitoring_core/            # Monitoring engine
│   ├── aggregator.py          # Metric aggregation
│   ├── alert_engine.py        # Alerting system
│   ├── storage.py             # Storage backends
│   └── query_api.py           # Query API
├── exporters/                  # Platform integrations
│   ├── prometheus_exporter.py
│   ├── datadog_exporter.py
│   ├── grafana_datasource.py
│   └── webhook_exporter.py
├── visualization/              # Dashboards and viz
│   ├── dashboard/             # Web UI (Streamlit/React)
│   ├── grafana_dashboards/    # Grafana JSON
│   └── notebooks/             # Jupyter notebooks
├── tests/                      # Unit and integration tests
├── examples/                   # Example scripts
└── docs/                       # Documentation
```

## Architecture

See [DIFFUSION_MONITORING_ARCHITECTURE.md](../DIFFUSION_MONITORING_ARCHITECTURE.md) for detailed architecture.

## Supported Models

- **LLaDA** (Microsoft)
- **Open-dLLM**
- **LLaDA2.0** (MoE variant)
- **Custom diffusion transformers** (via plugin API)

## Metrics Tracked

### Per-Step Metrics
- Step latency
- Predicted tokens
- Mask coverage
- Confidence scores
- Attention entropy
- Logit variance

### Convergence Metrics
- Token flip rate
- Confidence delta
- Perplexity per step
- Early stop efficiency

### Quality Metrics
- Final perplexity
- Semantic coherence
- Grammatical score
- Output diversity

## POC Roadmap

- [x] Phase 1: Foundation (instrumentation, basic metrics)
- [ ] Phase 2: Export & Basic Visualization (Prometheus, Grafana)
- [ ] Phase 3: Advanced Visualization (attention maps, animations)
- [ ] Phase 4: Production Features (alerts, storage, integrations)

## Contributing

We welcome contributions! Areas of interest:

1. **New model support**: Add instrumentation for other diffusion LLMs
2. **Exporters**: Implement integrations with other platforms
3. **Visualizations**: Create new visualization types
4. **Optimization**: Reduce monitoring overhead
5. **Documentation**: Improve guides and tutorials

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE)

## Citation

```bibtex
@software{diffusion_monitor2025,
  title={Diffusion Model Monitoring Platform},
  author={Worthington, Brian},
  year={2025},
  url={https://github.com/tcBio/oss_srv}
}
```

## Resources

- [Architecture Document](../DIFFUSION_MONITORING_ARCHITECTURE.md)
- [API Documentation](docs/API.md)
- [Grafana Dashboard Guide](docs/GRAFANA_SETUP.md)
- [LLaDA Project](https://github.com/microsoft/LLaDA)
- [Open-dLLM Project](https://github.com/ML-Diffusion-Language/Open-dLLM)

## Support

- Issues: https://github.com/tcBio/oss_srv/issues
- Discussions: https://github.com/tcBio/oss_srv/discussions
- Email: support@example.com

---

**Status**: Early POC - Phase 1 in progress

**Target Release**: Q2 2025
