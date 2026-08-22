import torch
from faster_whisper import WhisperModel


class ASRModel:

    def __init__(
        self,
        model_size="medium"
    ):

        print("Loading Whisper ASR model...")

        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
        else:
            self.device = "cpu"
            self.compute_type = "int8"


        print(f"Device: {self.device}")
        print(f"Compute type: {self.compute_type}")


        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type
        )


        print("Whisper ASR model loaded successfully.")



    def transcribe(self, audio_path):

        print("Transcribing audio...")


        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5
        )


        transcript = ""
        segment_list = []


        for segment in segments:

            transcript += segment.text + " "

            segment_list.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                }
            )


        return {

            "language": info.language,

            "language_probability":
                info.language_probability,

            "text":
                transcript.strip(),

            "segments":
                segment_list
        }



# Singleton model
asr_model = ASRModel()