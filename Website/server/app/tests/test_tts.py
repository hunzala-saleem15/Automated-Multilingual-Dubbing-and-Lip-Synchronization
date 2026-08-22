from app.models_ai.xtts import tts_model


tts_model.generate_speech(
    text="Hello, this is a test of my multilingual dubbing system.",
    speaker_wav=r"D:\Website\server\test\speaker.wav",
    language="en",
    output_path=r"D:\Website\server\test\output.wav"
)

print("Done")
