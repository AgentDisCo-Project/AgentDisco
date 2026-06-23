import os
import asyncio
import subprocess
import signal
import logging
import gin
import sys
sys.path.append('.')

import httpx
from typing import Optional
from openai import AsyncOpenAI
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')

_SERVER_PROCESS: Optional[subprocess.Popen] = None


@gin.configurable()
class QwenTTSDeploy:
    AVAILABLE_VOICES = [
        "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
        "Ryan", "Aiden", "Ono_Anna", "Sohee",
    ]

    def __init__(
        self,
        model_name: str = "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        voice: str = "Vivian",
        language: str = "Chinese",
        response_format: str = "wav",
        max_retries: int = 5,
        retry_delay: int = 3,
        timeout: int = 120,
        server_port: int = 8091,
        server_host: str = "0.0.0.0",
        server_base_url: str = "http://localhost:8091/v1",
        server_api_key: str = "EMPTY",
        server_python: str = "",
        gpu_memory_utilization: float = 0.9,
        startup_timeout: int = 300,
        auto_start: bool = True,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.voice = voice
        self.language = language
        self.response_format = response_format
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.server_port = server_port
        self.server_host = server_host
        self.server_python = server_python
        self.gpu_memory_utilization = gpu_memory_utilization
        self.startup_timeout = startup_timeout
        self.auto_start = auto_start

        self._base_url = server_base_url
        self.client = AsyncOpenAI(
            api_key=server_api_key,
            base_url=self._base_url,
            timeout=timeout,
        )
        self._server_ready = False

    async def ensure_server(self):
        if self._server_ready:
            return
        if await self._health_check():
            self._server_ready = True
            logging.info("[QwenTTSDeploy] server already running")
            return
        if not self.auto_start:
            raise RuntimeError(
                f"[QwenTTSDeploy] server not reachable at {self._base_url} "
                f"and auto_start=False"
            )
        self._start_server()
        await self._wait_until_ready()
        self._server_ready = True

    async def _health_check(self) -> bool:
        health_url = self._base_url.replace("/v1", "") + "/health"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(health_url)
                return resp.status_code == 200
        except Exception:
            return False

    def _get_python(self) -> str:
        return self.server_python or sys.executable

    def _ensure_dependencies(self):
        python_bin = self._get_python()
        check = subprocess.run(
            [python_bin, "-c", "import vllm; import vllm_omni; print('ok')"],
            capture_output=True, text=True,
        )
        if check.returncode == 0:
            logging.info("[QwenTTSDeploy] vllm/vllm_omni ready")
            return

        logging.warning(
            f"[QwenTTSDeploy] vllm/vllm_omni import failed: {check.stderr[-500:]}"
        )
        raise RuntimeError(
            "[QwenTTSDeploy] vllm and vllm-omni must be pre-installed. "
            "Run: pip install vllm vllm-omni"
        )

    def _start_server(self):
        global _SERVER_PROCESS
        if _SERVER_PROCESS and _SERVER_PROCESS.poll() is None:
            logging.info("[QwenTTSDeploy] server process already exists, skipping start")
            return

        self._ensure_dependencies()

        python_bin = self._get_python()
        cmd = [
            python_bin, "-m", "vllm_omni.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--port", str(self.server_port),
            "--host", self.server_host,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]
        logging.info(f"[QwenTTSDeploy] starting vLLM-Omni server: {' '.join(cmd)}")
        _SERVER_PROCESS = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    async def _wait_until_ready(self):
        logging.info(
            f"[QwenTTSDeploy] waiting for server to be ready "
            f"(timeout={self.startup_timeout}s)..."
        )
        elapsed = 0
        interval = 3
        while elapsed < self.startup_timeout:
            if _SERVER_PROCESS and _SERVER_PROCESS.poll() is not None:
                stdout_tail = ""
                if _SERVER_PROCESS.stdout:
                    try:
                        stdout_tail = _SERVER_PROCESS.stdout.read().decode(
                            "utf-8", errors="ignore"
                        )[-2000:]
                    except Exception:
                        pass
                logging.error(
                    f"[QwenTTSDeploy] server stdout/stderr:\n{stdout_tail}"
                )
                raise RuntimeError(
                    f"[QwenTTSDeploy] server process exited with code "
                    f"{_SERVER_PROCESS.returncode}"
                )
            if await self._health_check():
                logging.info(
                    f"[QwenTTSDeploy] server ready after {elapsed}s"
                )
                return
            await asyncio.sleep(interval)
            elapsed += interval
        raise RuntimeError(
            f"[QwenTTSDeploy] server did not become ready within "
            f"{self.startup_timeout}s"
        )

    async def synthesize(
        self,
        text: str,
        instructions: str = "",
    ) -> Optional[bytes]:
        await self.ensure_server()

        attempt = 0
        while attempt < self.max_retries:
            try:
                extra_body = {"language": self.language}
                if instructions:
                    extra_body["instructions"] = instructions

                response = await self.client.audio.speech.create(
                    model=self.model_name,
                    input=text,
                    voice=self.voice,
                    response_format=self.response_format,
                    extra_body=extra_body,
                )
                return response.read()

            except Exception as e:
                error_str = str(e)

                if ("429" in error_str or
                    "Too Many Requests" in error_str or
                    "exceed quota" in error_str):
                    tqdm.write(f"[QwenTTSDeploy] rate limited (not counted): {error_str}")
                    await asyncio.sleep(self.retry_delay * 3)
                    continue

                attempt += 1
                tqdm.write(f"[QwenTTSDeploy] attempt {attempt}/{self.max_retries} failed: {error_str}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    logging.error(f"[QwenTTSDeploy] exhausted retries for text: {text[:50]}...")
                    return None

    @staticmethod
    def shutdown_server():
        global _SERVER_PROCESS
        if _SERVER_PROCESS and _SERVER_PROCESS.poll() is None:
            logging.info("[QwenTTSDeploy] shutting down vLLM-Omni server...")
            os.killpg(os.getpgid(_SERVER_PROCESS.pid), signal.SIGTERM)
            try:
                _SERVER_PROCESS.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(_SERVER_PROCESS.pid), signal.SIGKILL)
            _SERVER_PROCESS = None
