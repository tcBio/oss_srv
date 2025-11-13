# Contributing to Diffusion Monitor

Thank you for your interest in contributing! This guide will help you get started.

## Ways to Contribute

### 1. Add Support for New Models

We're actively seeking contributions to support more diffusion-based language models:

- **LLaDA variants**: LLaDA-13B, LLaDA-70B, LLaDA2.0 (MoE)
- **Open-dLLM**: Full integration with their training framework
- **Custom architectures**: Diffusion Transformers (DiT), UNet-based models
- **Multimodal**: LLaDA-V (vision), audio diffusion models

**How to add:**

1. Fork the repository
2. Create `diffusion_monitor/adapters/your_model.py`
3. Implement model-specific hooks:

```python
from diffusion_monitor import DiffusionMonitor

class YourModelMonitor(DiffusionMonitor):
    def attach(self, model):
        # Custom hook logic for your model
        for name, module in model.named_modules():
            if 'your_specific_layer' in name:
                hook = module.register_forward_hook(...)
                self.hooks.append(hook)
```

4. Add example in `examples/monitor_your_model.py`
5. Submit PR with tests

### 2. Create New Exporters

Help integrate with more observability platforms:

**Needed:**
- DataDog (DogStatsD + APM)
- New Relic
- Datadog Logs
- Splunk
- Elastic APM
- Custom webhook handlers (PagerDuty, Slack, Discord)

**Template:**

```python
# exporters/your_platform_exporter.py

class YourPlatformExporter:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def record_step(self, model: str, step: int, metrics: DiffusionMetrics):
        # Push metrics to your platform
        pass

    def record_inference(self, model: str, metrics: List, summary: Dict):
        # Push inference summary
        pass
```

### 3. Build Visualizations

Create new visualization types:

**Ideas:**
- Interactive 3D attention visualization
- Real-time Streamlit dashboard
- Token-level diff viewer (step-by-step changes)
- Attention flow animation
- Embedding space visualization (t-SNE of hidden states)

**Template:**

```python
# visualization/your_viz.py

def your_visualization(metrics_history: List[Dict], **kwargs):
    # Create visualization
    fig = ...
    return fig
```

### 4. Improve Documentation

- Add tutorials for specific models
- Create video walkthroughs
- Write blog posts about use cases
- Translate documentation
- Improve API docstrings

### 5. Performance Optimization

Help reduce monitoring overhead:

- Optimize tensor operations
- Reduce memory footprint
- Implement sampling (monitor every N steps)
- Add async metric export
- CUDA kernel for metric computation

**Target:** <2% overhead on inference

### 6. Testing & Quality

- Add unit tests for core components
- Integration tests with real models
- Benchmark suite
- CI/CD pipeline improvements
- Type hints and mypy compliance

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/tcBio/oss_srv.git
cd oss_srv/diffusion_monitor

# Install in development mode
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

### 2. Run Tests

```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=diffusion_monitor --cov-report=html

# Specific test
pytest tests/test_metrics_collector.py::test_collect_step_metrics
```

### 3. Code Style

We use:
- **black** for formatting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black diffusion_monitor/ exporters/ visualization/

# Lint
flake8 diffusion_monitor/

# Type check
mypy diffusion_monitor/
```

### 4. Run Examples

```bash
cd examples
python basic_monitoring.py
```

## Contribution Workflow

### 1. Create an Issue

Before starting work, create an issue describing:
- Problem or feature
- Proposed solution
- Expected behavior

**Labels:**
- `enhancement`: New feature
- `bug`: Bug fix
- `documentation`: Docs improvement
- `good-first-issue`: Great for newcomers
- `help-wanted`: We need community help

### 2. Fork and Branch

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/oss_srv.git
cd oss_srv

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Make Changes

- Write clean, documented code
- Follow existing code style
- Add tests for new features
- Update documentation

### 4. Test Thoroughly

```bash
# Run all tests
pytest tests/ -v

# Run specific integration test
pytest tests/integration/test_llada_monitor.py

# Manual testing
python examples/basic_monitoring.py
```

### 5. Commit

```bash
git add .
git commit -m "feat: Add support for LLaDA2.0 MoE variant"

# Follow conventional commits:
# feat: New feature
# fix: Bug fix
# docs: Documentation
# test: Tests
# refactor: Code refactoring
# perf: Performance improvement
```

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create Pull Request on GitHub with:
- Clear title and description
- Reference related issues (`Fixes #123`)
- Screenshots/videos if relevant
- Performance impact (if applicable)

### 7. Code Review

- Address reviewer feedback
- Keep PR scope focused
- Be responsive to comments
- Update based on suggestions

## PR Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages are clear
- [ ] PR description is complete

## Release Process

Maintainers will:
1. Review and merge PR
2. Update CHANGELOG.md
3. Tag release (semantic versioning)
4. Publish to PyPI
5. Update documentation

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on what's best for the project

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, showcases
- **Discord** (TBD): Real-time chat
- **Email**: support@example.com

## Areas Needing Help

### High Priority

1. **LLaDA Integration**: Test with real LLaDA model
2. **DataDog Exporter**: Implement DogStatsD integration
3. **Grafana Dashboards**: Create production-ready dashboards
4. **Documentation**: Video tutorials and blog posts
5. **Performance**: Reduce monitoring overhead

### Good First Issues

- Add more unit tests
- Improve error messages
- Add docstring examples
- Create simple visualization variants
- Fix typos in documentation

### Advanced Contributions

- CUDA kernels for metric computation
- Distributed monitoring (multi-GPU)
- Model comparison framework
- Automated anomaly detection (ML-based)
- Integration with Ray Serve / TorchServe

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Mentioned in documentation
- Acknowledged in papers/blog posts (if significant contribution)

## Getting Help

Stuck? Reach out:

1. Comment on your issue/PR
2. Ask in GitHub Discussions
3. Email: brian@example.com
4. Check [QUICKSTART.md](../QUICKSTART.md)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Diffusion Monitor! 🙏

Together we're making diffusion models more observable and debuggable for everyone.
