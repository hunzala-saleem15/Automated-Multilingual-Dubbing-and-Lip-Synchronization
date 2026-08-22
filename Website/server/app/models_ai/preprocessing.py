import os
import subprocess
import cv2
FFMPEG_PATH = r"D:\ffmpeg\bin\ffmpeg.exe"

class VideoPreprocessor:

    def __init__(self):
        pass

    def extract_audio(self, video_path, output_audio):

        os.makedirs(os.path.dirname(output_audio), exist_ok=True)

        command = [
            FFMPEG_PATH,
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_audio
        ]

        subprocess.run(command, check=True)

        return output_audio

    def extract_frames(self, video_path, output_folder):

        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_path = os.path.join(
                output_folder,
                f"{count:06d}.jpg"
            )

            cv2.imwrite(frame_path, frame)

            count += 1

        cap.release()

        return output_folder

    def get_video_info(self, video_path):

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        duration = frames / fps if fps else 0

        cap.release()

        return {

            "fps": fps,

            "width": width,

            "height": height,

            "frames": frames,

            "duration": duration

        }

    def merge_audio_video(
        self,
        video_path,
        audio_path,
        output_path
    ):

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        command = [

            FFMPEG_PATH,

            "-y",

            "-i", video_path,

            "-i", audio_path,

            "-c:v", "copy",

            "-c:a", "aac",

            "-shortest",

            output_path
        ]

        subprocess.run(command, check=True)

        return output_path


# Initialize once
preprocessor = VideoPreprocessor()