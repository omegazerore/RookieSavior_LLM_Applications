import asyncio
import os
import json
import base64

import pyaudio
import websockets

from initialization import credential_init

# pip install -r requirements.txt
# pip install websocket

# Configuration
credential_init()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
MODEL = "gpt-realtime-translate"
# The streaming translation WebSocket endpoint
WS_URL = "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"

headers = [
    f"Authorization: Bearer {OPENAI_API_KEY}",
    "OpenAI-Safety-Identifier: hashed-user-id"  # Adjust based on official documentation header requirements
]


async def send_audio_stream(websocket):
    """
    Streams raw microphone audio via PyAudio to the API.
    Expects 24kHz, 16-bit, mono PCM audio.
    """
    print("Audio stream sender started...")

    audio_queue = asyncio.Queue()

    def pyaudio_callback(in_data, frame_count, time_info, status_flags):
        """
        PyAudio callback: called when audio input buffer is ready.
        Matches the sounddevice callback pattern from notebook.ipynb.
        """
        # Encode PCM bytes to base64 for API transmission
        encoded = base64.b64encode(in_data).decode("utf-8")
        # Use put_nowait since we are in a callback (non-blocking)
        audio_queue.put_nowait(encoded)
        return (None, pyaudio.paContinue)

    p = pyaudio.PyAudio()

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            input=True,
            frames_per_buffer=2400,  # 100ms at 24kHz, mono, 16-bit
            stream_callback=pyaudio_callback
        )

        stream.start_stream()

        # Continuously read encoded audio from queue and send to WebSocket
        while stream.is_active():
            try:
                audio_chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
                audio_event = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_chunk
                }
                await websocket.send(json.dumps(audio_event))
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        print("Stopping audio stream...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        close_event = {"type": "session.close"}
        await websocket.send(json.dumps(close_event))


async def receive_translation_stream(websocket):
    """
    Listens for continuous translated audio chunks and transcript text deltas.
    """
    print("Translation receiver started...")
    try:
        async for message in websocket:
            event = json.loads(message)
            event_type = event.get("type")

            # 1. Capture text transcript updates (deltas)
            if event_type == "conversation.item.transcript.delta":
                text_delta = event.get("delta", "")
                print(text_delta, end="", flush=True)

            # 2. Capture translated output audio chunks
            elif event_type == "conversation.item.audio.delta":
                # 24kHz PCM16 Audio payload
                audio_delta_b64 = event.get("delta")
                audio_bytes = base64.b64decode(audio_delta_b64)
                # Route audio_bytes directly to your speaker output/media pipeline here
                pass

            elif event_type == "error":
                print(f"\nAPI Error: {event.get('error')}")

    except websockets.exceptions.ConnectionClosed:
        print("\nConnection closed by the server.")


async def main():
    if not OPENAI_API_KEY:
        print("Please set your OPENAI_API_KEY environment variable.")
        return

    print(f"Connecting to {WS_URL} using {MODEL}...")
    async with websockets.connect(WS_URL, extra_headers=headers) as websocket:
        # Initialize and update the session configuration
        # For translation, one target language is configured per session
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "target_language": "en",  # Target output language (e.g., English)
                "voice": "coral"  # Tone/pitch dynamically matches speaker, but sets profile
            }
        }
        await websocket.send(json.dumps(session_update))
        print("Session configured. Start speaking or streaming audio.")

        # Concurrently stream audio up and handle incoming translations down
        sender_task = asyncio.create_task(send_audio_stream(websocket))
        receiver_task = asyncio.create_task(receive_translation_stream(websocket))

        await asyncio.gather(sender_task, receiver_task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
