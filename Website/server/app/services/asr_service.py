from faster_whisper import WhisperModel
import torch


class ASRService:

    def __init__(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        compute_type = (
            "float16"
            if device == "cuda"
            else "int8"
        )

        print(f"Loading Whisper Large-v3 on {device}")

        self.model = WhisperModel(
            "large-v3",
            device=device,
            compute_type=compute_type
        )


    def transcribe(self, audio_path):

        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True
        )


        result = []

        full_text = ""

        for segment in segments:

            full_text += segment.text + " "

            result.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })


        return {
            "text": full_text.strip(),
            "segments": result,
            "language": info.language
        }