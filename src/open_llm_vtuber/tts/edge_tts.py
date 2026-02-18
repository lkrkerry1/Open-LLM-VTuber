import sys
import os

import edge_tts
from loguru import logger
from .tts_interface import TTSInterface

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


class TTSEngine(TTSInterface):
    def __init__(self, voice="en-US-AvaMultilingualNeural"):
        self.voice = voice
        self.temp_audio_file = "temp"
        self.file_extension = "mp3"
        self.new_audio_dir = "cache"

        if not os.path.exists(self.new_audio_dir):
            os.makedirs(self.new_audio_dir)

    def generate_audio(self, text, file_name_no_ext=None):
        """
        Generate speech audio file using TTS.
        text: str
            the text to speak
        file_name_no_ext: str
            name of the file without extension

        Returns:
        str: the path to the generated audio file

        Raises:
        Exception: if TTS generation fails
        """
        file_name = self.generate_cache_file_name(
            file_name_no_ext, self.file_extension
        )

        try:
            communicate = edge_tts.Communicate(
                text, self.voice
            )
            communicate.save_sync(file_name)
        except Exception as e:
            logger.critical(
                f"\nError: edge-tts unable to generate audio: {e}"
            )
            logger.critical(
                "It's possible that edge-tts is blocked in your region."
            )
            # 抛出异常以便上层捕获并处理静默 payload
            raise e

        return file_name

    def remove_file(self, file_path: str) -> None:
        """Safely remove a file, ignoring None or non-existent paths."""
        if not file_path:
            return
        try:
            os.remove(file_path)
            logger.debug(f"Removed file: {file_path}")
        except FileNotFoundError:
            pass  # 文件可能已被删除
        except Exception as e:
            logger.warning(
                f"Failed to remove file {file_path}: {e}"
            )
