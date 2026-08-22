from app.models_ai.asr import asr_model

result = asr_model.transcribe("test.wav")

print(result)