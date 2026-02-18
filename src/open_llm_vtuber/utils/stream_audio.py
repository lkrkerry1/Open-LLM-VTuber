import base64
from pydub import AudioSegment
from pydub.utils import make_chunks
from loguru import logger


def _get_volume_by_chunks(
    audio: AudioSegment, chunk_length_ms: int
) -> list:
    chunks = make_chunks(audio, chunk_length_ms)
    volumes = [chunk.rms for chunk in chunks]
    max_volume = max(volumes)
    if max_volume == 0:
        raise ValueError("Audio is empty or all zero.")
    return [volume / max_volume for volume in volumes]


def prepare_audio_payload(
    audio_path: str | None,
    chunk_length_ms: int = 20,
    display_text: dict | None = None,
    actions: dict | None = None,
    forwarded: bool = False,
) -> dict[str, any]:
    """
    Prepares the audio payload for sending to a broadcast endpoint.
    If audio_path is None, returns a payload with audio=None for silent display.

    Parameters:
        audio_path (str | None): The path to the audio file to be processed, or None for silent display
        chunk_length_ms (int): The length of each audio chunk in milliseconds
        display_text (dict, optional): Dictionary with 'text', 'name', 'avatar' for display
        actions (dict, optional): Dictionary of actions associated with the audio
        forwarded (bool): Whether this is forwarded audio

    Returns:
        dict: The audio payload to be sent
    """
    if not audio_path:
        # Return payload for silent display
        logger.debug(
            f"Creating silent payload for display text: {display_text}"
        )
        return {
            "type": "audio",
            "audio": None,
            "volumes": [],
            "slice_length": chunk_length_ms,
            "display_text": display_text,
            "actions": actions,
            "forwarded": forwarded,
        }

    try:
        audio = AudioSegment.from_file(audio_path)
        audio_bytes = audio.export(format="wav").read()
    except Exception as e:
        raise ValueError(
            f"Error loading or converting generated audio file to wav file '{audio_path}': {e}"
        )
    audio_base64 = base64.b64encode(audio_bytes).decode(
        "utf-8"
    )
    volumes = _get_volume_by_chunks(audio, chunk_length_ms)

    payload = {
        "type": "audio",
        "audio": audio_base64,
        "volumes": volumes,
        "slice_length": chunk_length_ms,
        "display_text": display_text,
        "actions": actions,
        "forwarded": forwarded,
    }

    return payload
