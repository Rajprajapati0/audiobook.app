import os
import os

import edge_tts


async def _generate_speech_edge(text: str, voice: str, output_path: str):
    """
    Uses Microsoft's Edge TTS service to turn text into an MP3 file.

    Writes to a temporary '.tmp' file first, then renames it to the real
    filename only once the write has fully succeeded. This way, if the app
    crashes mid-generation, you're left with a stray .tmp file (harmless,
    cleaned up on next startup) instead of a corrupted .mp3 sitting at the
    path the rest of the app thinks is a finished, valid file.
    """
    tmp_path = output_path + ".tmp"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(tmp_path)
    os.replace(tmp_path, output_path)  # atomic on virtually all systems


async def generate_speech(provider: str, text: str, voice: str, output_path: str):
    """
    Generic entry point for text-to-speech.
    Right now only 'edge' exists — later providers (Google, Azure, ElevenLabs)
    get added here as new branches, without changing any code that calls this function.

    Writes are atomic: audio is generated to a temporary file first, and only
    renamed to the real filename once generation fully succeeds. This means
    a crash mid-generation can never leave a corrupt, half-written MP3 sitting
    at the final path — either the finished file exists, or nothing does.
    """
    tmp_path = output_path + ".tmp"
    try:
        if provider == "edge":
            await _generate_speech_edge(text, voice, tmp_path)
        else:
            raise ValueError(f"Unknown TTS provider: {provider}")
        os.replace(tmp_path, output_path)  # atomic rename
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)  # don't leave a half-written leftover behind
        raise


async def list_english_voices():
    """Returns a simplified list of English voices available from Edge TTS."""
    all_voices = await edge_tts.list_voices()
    english = [v for v in all_voices if v["Locale"].startswith("en-")]
    return [
        {
            "short_name": v["ShortName"],
            "gender": v["Gender"],
            "locale": v["Locale"],
            "friendly_name": v.get("FriendlyName", v["ShortName"]),
        }
        for v in english
    ]