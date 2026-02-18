import json
import httpx, requests, asyncio
from typing import AsyncIterator, List, Dict, Any, Optional
from loguru import logger
from .stateless_llm_interface import StatelessLLMInterface


class OllamaNativeLLM(StatelessLLMInterface):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.8,
        keep_alive: float = -1,
        **kwargs,
    ):
        # 忽略未使用的关键字参数
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.client = httpx.AsyncClient(
            timeout=60.0, http2=False
        )
        logger.info(
            f"Initialized OllamaNativeLLM with base_url: {self.base_url}, model: {self.model}"
        )

    import asyncio

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[str]:
        # 预处理消息（与之前相同）
        processed_messages = []
        if system:
            processed_messages.append(
                {"role": "system", "content": system}
            )
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                texts = []
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                    ):
                        texts.append(part.get("text", ""))
                msg["content"] = " ".join(texts)
            processed_messages.append(msg)

        payload = {
            "model": self.model,
            "messages": processed_messages,
            "stream": False,  # 保持非流式，以使用同步 requests
            "options": {
                "temperature": self.temperature,
                "num_keep_alive": self.keep_alive,
            },
        }
        logger.debug(f"Ollama request payload: {payload}")

        # 使用 asyncio.to_thread 在后台线程中执行同步 requests.post
        try:
            response = await asyncio.to_thread(
                requests.post,
                f"{self.base_url}/api/chat",
                json=payload,
                headers={
                    "Content-Type": "application/json"
                },
                timeout=60,
            )
            if response.status_code != 200:
                logger.error(
                    f"Ollama returned {response.status_code}: {response.text}"
                )
                yield f"Error: Ollama returned {response.status_code}"
                return
            data = response.json()
            if (
                "message" in data
                and "content" in data["message"]
            ):
                yield data["message"]["content"]
            else:
                logger.error(f"Unexpected response: {data}")
                yield f"Error: Unexpected response"
        except Exception as e:
            logger.exception(f"Ollama request failed: {e}")
            yield f"Error: {e}"

    async def close(self):
        await self.client.aclose()
