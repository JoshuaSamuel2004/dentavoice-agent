FROM python:3.12-slim
 
WORKDIR /app
 
# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Apply the Sarvam TTS fix (linear16 codec + PCM mime type)
RUN SARVAM_TTS_PATH=$(python -c "import livekit.plugins.sarvam.tts as m; print(m.__file__)") && \
    sed -i 's/mime_type="audio\/wav",\(.*stream=True\)/mime_type="audio\/pcm",\1/' "$SARVAM_TTS_PATH" && \
    sed -i '/data\["temperature"\]/a\                    data["output_audio_codec"] = "linear16"' "$SARVAM_TTS_PATH"
 
# Copy agent code
COPY agent.py .
COPY .env .
 
# Run the agent
CMD ["python", "agent.py", "start"]
 