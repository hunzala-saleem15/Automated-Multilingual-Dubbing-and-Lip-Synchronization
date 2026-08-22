from server.app.models_ai.talklip import talklip_model

talklip_model.generate_video(
    video_path="input.mp4",
    audio_path="outputs/test.wav",
    output_path="outputs/final.mp4"
)