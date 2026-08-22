import os
import subprocess


class XTTSModel:

    def __init__(self):
        self.python = r"D:\Website\server\XTTS\venv\Scripts\python.exe"
        self.script = r"D:\Website\server\XTTS\venv\xtts_inference.py"
        print("XTTS Wrapper Ready")

    def generate_speech(
        self,
        text,
        speaker_wav,
        language,
        output_path
    ):

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print("\n========== XTTS DEBUG ==========")
        print("Python :", self.python)
        print("Script :", self.script)
        print("PWD    :", os.getcwd())
        print("Speaker:", speaker_wav)
        print("Exists :", os.path.exists(speaker_wav))

        if os.path.exists(speaker_wav):
            print("Size   :", os.path.getsize(speaker_wav))
            print("Abs    :", os.path.abspath(speaker_wav))

        print("Output :", output_path)
        print("Python exists :", os.path.exists(self.python))
        print("Script exists :", os.path.exists(self.script))
        print("================================\n")

        command = [
            self.python,
            self.script,
            "--text", text,
            "--speaker_wav", speaker_wav,
            "--language", language,
            "--output", output_path
        ]

        print("COMMAND:")
        print(" ".join(f'"{x}"' if " " in x else x for x in command))

        try:
            result = subprocess.run(
                command,
                cwd=r"D:\Website\server"
            )

            print("RETURN CODE:", result.returncode)

            if result.returncode != 0:
                raise RuntimeError(
                    f"XTTS failed with exit code {result.returncode}"
                )

        except Exception as e:
            print("\n========== XTTS EXCEPTION ==========")
            print(type(e).__name__)
            print(e)
            print("====================================")
            raise

        return output_path


tts_model = XTTSModel()