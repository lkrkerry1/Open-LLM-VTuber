import asyncio
import json
import re
import uuid
from datetime import datetime
from typing import List, Optional, Dict
from loguru import logger

from ..agent.output_types import DisplayText, Actions
from ..live2d_model import Live2dModel
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload
from .types import WebSocketSend


class TTSTaskManager:
    """Manages TTS tasks and ensures ordered delivery to frontend while allowing parallel TTS generation"""

    def __init__(self) -> None:
        self.task_list: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        self._payload_queue: asyncio.Queue[Dict] = (
            asyncio.Queue()
        )
        self._sender_task: Optional[asyncio.Task] = None
        self._sequence_counter = 0
        self._next_sequence_to_send = 0

    async def speak(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        websocket_send: WebSocketSend,
    ) -> None:
        """
        Queue a TTS task while maintaining order of delivery.
        """
        if not isinstance(tts_text, str):
            tts_text = str(tts_text)
        if (
            len(
                re.sub(
                    r'[\s.,!?，。！？\'"』」）】\s]+',
                    "",
                    tts_text,
                )
            )
            == 0
        ):
            logger.debug(
                "Empty TTS text, sending silent display payload"
            )
            current_sequence = self._sequence_counter
            self._sequence_counter += 1
            await self._ensure_sender_task(websocket_send)
            await self._send_silent_payload(
                display_text=(
                    display_text.to_dict()
                    if display_text
                    else None
                ),
                actions=(
                    actions.to_dict() if actions else None
                ),
                sequence_number=current_sequence,
            )
            return

        logger.debug(
            f"🏃Queuing TTS task for: '''{tts_text}''' (by {display_text.name})"
        )

        current_sequence = self._sequence_counter
        self._sequence_counter += 1

        await self._ensure_sender_task(websocket_send)

        task = asyncio.create_task(
            self._process_tts(
                tts_text=tts_text,
                display_text=display_text,
                actions=actions,
                live2d_model=live2d_model,
                tts_engine=tts_engine,
                sequence_number=current_sequence,
            )
        )
        self.task_list.append(task)

    async def _ensure_sender_task(
        self, websocket_send: WebSocketSend
    ):
        """Ensure the sender task is running."""
        if (
            not self._sender_task
            or self._sender_task.done()
        ):
            self._sender_task = asyncio.create_task(
                self._process_payload_queue(websocket_send)
            )
            logger.debug("Created new sender task")

    async def _process_payload_queue(
        self, websocket_send: WebSocketSend
    ) -> None:
        """
        Process and send payloads in correct order. Runs forever, handling exceptions.
        """
        buffered_payloads: Dict[int, Dict] = {}

        while True:
            try:
                payload, sequence_number = (
                    await self._payload_queue.get()
                )
                buffered_payloads[sequence_number] = payload

                # Send payloads in order
                while (
                    self._next_sequence_to_send
                    in buffered_payloads
                ):
                    next_payload = buffered_payloads.pop(
                        self._next_sequence_to_send
                    )
                    await websocket_send(
                        json.dumps(next_payload)
                    )
                    logger.debug(
                        f"Sent payload for sequence {self._next_sequence_to_send}"
                    )
                    self._next_sequence_to_send += 1

                self._payload_queue.task_done()

            except asyncio.CancelledError:
                logger.debug("Sender task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in sender task: {e}")
                # 短暂等待后继续，防止高频错误
                await asyncio.sleep(0.1)
                continue

    async def _send_silent_payload(
        self,
        display_text: dict,  # 改为接收字典
        actions: dict,  # 改为接收字典
        sequence_number: int,
    ) -> None:
        audio_payload = prepare_audio_payload(
            audio_path=None,
            display_text=display_text,
            actions=actions,
        )
        await self._payload_queue.put(
            (audio_payload, sequence_number)
        )
        logger.debug(
            f"Queued silent payload for sequence {sequence_number}"
        )

    async def _process_tts(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        sequence_number: int,
    ) -> None:
        # 提前转换为字典
        display_dict = (
            display_text.to_dict() if display_text else None
        )
        actions_dict = (
            actions.to_dict() if actions else None
        )
        audio_file_path = None
        try:
            audio_file_path = await self._generate_audio(
                tts_engine, tts_text
            )
            payload = prepare_audio_payload(
                audio_path=audio_file_path,
                display_text=display_dict,
                actions=actions_dict,
            )
            await self._payload_queue.put(
                (payload, sequence_number)
            )
            logger.debug(
                f"Queued audio payload for sequence {sequence_number}"
            )

        except Exception as e:
            logger.error(
                f"Error generating audio for sequence {sequence_number}: {e}"
            )
            # 发送静默 payload
            payload = prepare_audio_payload(
                audio_path=None,
                display_text=display_dict,
                actions=actions_dict,
            )
            await self._payload_queue.put(
                (payload, sequence_number)
            )
            logger.debug(
                f"Queued silent payload (fallback) for sequence {sequence_number}"
            )

        finally:
            if audio_file_path:
                try:
                    tts_engine.remove_file(audio_file_path)
                except Exception as e:
                    logger.warning(
                        f"Failed to remove audio file {audio_file_path}: {e}"
                    )

    async def _generate_audio(
        self, tts_engine: TTSInterface, text: str
    ) -> str:
        """Generate audio file from text"""
        logger.debug(
            f"🏃Generating audio for '''{text}'''..."
        )
        return await tts_engine.async_generate_audio(
            text=text,
            file_name_no_ext=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
        )

    def clear(self) -> None:
        """Clear all pending tasks and reset state"""
        self.task_list.clear()
        if self._sender_task:
            self._sender_task.cancel()
        self._sequence_counter = 0
        self._next_sequence_to_send = 0
        self._payload_queue = asyncio.Queue()
        logger.debug("TTSTaskManager cleared")
