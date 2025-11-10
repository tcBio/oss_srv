"""
OSS_SRV Inference Client
Synchronous client for OSS-20B inference
"""

import subprocess
import json
import os
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class InferenceConfig:
    """Configuration for inference"""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    stop_sequences: List[str] = None

    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


@dataclass
class InferenceResult:
    """Result from inference"""
    text: str
    tokens: List[int]
    inference_time_ms: float
    total_time_ms: float
    throughput_tokens_per_sec: float
    success: bool = True
    error: Optional[str] = None


class InferenceClient:
    """
    Synchronous inference client for OSS_SRV

    Example:
        >>> client = InferenceClient(model_path="model.engine")
        >>> result = client.complete("Once upon a time", max_tokens=100)
        >>> print(result.text)
    """

    def __init__(self, model_path: str, executable_path: Optional[str] = None):
        """
        Initialize inference client

        Args:
            model_path: Path to TensorRT engine file (.engine)
            executable_path: Path to complete_inference executable
                           (default: searches in common locations)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Find executable
        if executable_path:
            self.executable = Path(executable_path)
        else:
            self.executable = self._find_executable()

        if not self.executable or not self.executable.exists():
            raise FileNotFoundError(
                "complete_inference executable not found. "
                "Please specify executable_path or ensure it's in your PATH"
            )

    def _find_executable(self) -> Optional[Path]:
        """Find complete_inference executable in common locations"""
        search_paths = [
            Path(".") / "complete_inference",
            Path("./build") / "complete_inference",
            Path("..") / "complete_inference",
            Path("/app") / "complete_inference",
        ]

        # Also check PATH
        import shutil
        path_exe = shutil.which("complete_inference")
        if path_exe:
            search_paths.insert(0, Path(path_exe))

        for path in search_paths:
            if path.exists() and os.access(path, os.X_OK):
                return path

        return None

    def complete(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        config: Optional[InferenceConfig] = None
    ) -> InferenceResult:
        """
        Generate completion for the given prompt

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate (overrides config)
            temperature: Sampling temperature (overrides config)
            top_p: Nucleus sampling parameter (overrides config)
            config: InferenceConfig object (optional)

        Returns:
            InferenceResult with generated text and metrics
        """
        if config is None:
            config = InferenceConfig()

        # Override config with parameters
        final_max_tokens = max_tokens if max_tokens is not None else config.max_tokens
        final_temperature = temperature if temperature is not None else config.temperature
        final_top_p = top_p if top_p is not None else config.top_p

        # Build command
        cmd = [
            str(self.executable),
            str(self.model_path),
            prompt,
            str(final_max_tokens),
            str(final_temperature),
            str(final_top_p)
        ]

        try:
            # Run inference
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            if result.returncode != 0:
                return InferenceResult(
                    text="",
                    tokens=[],
                    inference_time_ms=0,
                    total_time_ms=0,
                    throughput_tokens_per_sec=0,
                    success=False,
                    error=result.stderr
                )

            # Parse output
            return self._parse_output(result.stdout)

        except subprocess.TimeoutExpired:
            return InferenceResult(
                text="",
                tokens=[],
                inference_time_ms=0,
                total_time_ms=0,
                throughput_tokens_per_sec=0,
                success=False,
                error="Inference timeout (120s)"
            )
        except Exception as e:
            return InferenceResult(
                text="",
                tokens=[],
                inference_time_ms=0,
                total_time_ms=0,
                throughput_tokens_per_sec=0,
                success=False,
                error=str(e)
            )

    def _parse_output(self, output: str) -> InferenceResult:
        """Parse output from complete_inference"""
        lines = output.split('\n')

        generated_text = ""
        tokens_generated = 0
        inference_time = 0.0
        total_time = 0.0
        throughput = 0.0

        for line in lines:
            line = line.strip()
            if line.startswith("GENERATED_TEXT:"):
                generated_text = line.split("GENERATED_TEXT:", 1)[1].strip()
            elif line.startswith("TOKENS_GENERATED:"):
                try:
                    tokens_generated = int(line.split(":", 1)[1].strip())
                except:
                    pass
            elif line.startswith("INFERENCE_TIME_MS:"):
                try:
                    inference_time = float(line.split(":", 1)[1].strip())
                except:
                    pass
            elif line.startswith("TOTAL_TIME_MS:"):
                try:
                    total_time = float(line.split(":", 1)[1].strip())
                except:
                    pass
            elif line.startswith("THROUGHPUT_TOKENS_PER_SEC:"):
                try:
                    throughput = float(line.split(":", 1)[1].strip())
                except:
                    pass

        return InferenceResult(
            text=generated_text,
            tokens=[],  # TODO: parse actual tokens if available
            inference_time_ms=inference_time,
            total_time_ms=total_time,
            throughput_tokens_per_sec=throughput,
            success=True
        )

    def batch_complete(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        config: Optional[InferenceConfig] = None
    ) -> List[InferenceResult]:
        """
        Generate completions for multiple prompts

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per completion
            config: InferenceConfig object

        Returns:
            List of InferenceResult objects
        """
        results = []
        for prompt in prompts:
            result = self.complete(prompt, max_tokens=max_tokens, config=config)
            results.append(result)
        return results
