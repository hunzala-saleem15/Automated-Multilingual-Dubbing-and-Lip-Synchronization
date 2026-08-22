import os
import shutil
import subprocess

# Local helper function import / fallback logic
try:
    from app.services.dubbing_service import update_job
except ImportError:
    def update_job(job_id, status, progress):
        pass


class TalkLipModel:
    def __init__(
        self,
        talklip_root=r"D:\Website\server\Talklip",
        checkpoint=r"D:\Website\server\app\model_weights\talklip\checkpoint_step000060000.pth",
        avhubert_root=r"D:\Website\server\Talklip\av_hubert",
        conda_path=r"C:\Users\user\miniconda3\condabin\conda.bat"
    ):
        self.talklip_root = talklip_root
        self.checkpoint = checkpoint
        self.avhubert_root = avhubert_root
        self.conda_path = conda_path

        # Runtime folders
        self.runtime_root = os.path.join(
            self.talklip_root,
            "runtime"
        )

        self.video_root = os.path.join(self.runtime_root, "videos")
        self.audio_root = os.path.join(self.runtime_root, "audio")
        self.bbx_root = os.path.join(self.runtime_root, "bbx")
        self.save_root = os.path.join(self.runtime_root, "result")

        os.makedirs(self.video_root, exist_ok=True)
        os.makedirs(self.audio_root, exist_ok=True)
        os.makedirs(self.bbx_root, exist_ok=True)
        os.makedirs(self.save_root, exist_ok=True)

        print("TalkLip Model initialized successfully.")

    def generate_video(
        self,
        video_path,
        audio_path,
        bbx_folder,
        output_path,
        progress_callback=None,
        job_id=None
    ):

        if job_id:
            update_job(job_id, "Preparing TalkLip Runtime", 91)

        if progress_callback:
            progress_callback("Preparing TalkLip Runtime", 91)

    ############################################################
    # Sample Name
    ############################################################

        sample = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        print("=" * 60)
        print("TALKLIP SAMPLE :", sample)
        print("=" * 60)

    ############################################################
    # Clean Previous Runtime Files
    ############################################################

        for folder in [
            self.video_root,
            self.audio_root,
            self.bbx_root,
            self.save_root,
        ]:
            os.makedirs(folder, exist_ok=True)

            for f in os.listdir(folder):
                path = os.path.join(folder, f)

                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except Exception:
                    pass

    ############################################################
    # Copy Video
    ############################################################

        runtime_video = os.path.join(
            self.video_root,
            sample + ".mp4"
        )

        shutil.copy2(video_path, runtime_video)

    ############################################################
    # Copy Audio
    ############################################################

        runtime_audio = os.path.join(
            self.audio_root,
            sample + ".wav"
        )

        shutil.copy2(audio_path, runtime_audio)

    ############################################################
    # Copy BBX
    ############################################################

        if not os.path.isdir(bbx_folder):
            raise FileNotFoundError(
                f"BBX folder not found:\n{bbx_folder}"
            )

        source_bbx = os.path.join(
            bbx_folder,
            sample + ".npy"
        )

        if not os.path.isfile(source_bbx):
            raise FileNotFoundError(
                f"BBX file not found:\n{source_bbx}"
            )

        runtime_bbx = os.path.join(
            self.bbx_root,
            sample + ".npy"
        )

        shutil.copy2(source_bbx, runtime_bbx)

    ############################################################
    # Filelist
    ############################################################

        filelist = os.path.join(
            self.runtime_root,
            "filelist.txt"
        )

        with open(filelist, "w") as f:
            f.write(sample + "\n")

    ############################################################
    # Execute TalkLip
    ############################################################

        print("=" * 60)
        print("Running TalkLip...")
        print("=" * 60)

        cmd = [
            self.conda_path,
            "run",
            "-n",
            "talklip",
            "python",
            "inf_test.py",
            "--filelist", filelist,
            "--video_root", self.video_root,
            "--audio_root", self.audio_root,
            "--bbx_root", self.bbx_root,
            "--save_root", self.save_root,
            "--ckpt_path", self.checkpoint,
            "--avhubert_root", self.avhubert_root,
        ]

        result = subprocess.run(
            cmd,
            cwd=self.talklip_root,
            capture_output=True,
            text=True,
            shell=True
        )

        print("=" * 60)
        print("STDOUT")
        print(result.stdout)

        print("=" * 60)
        print("STDERR")
        print(result.stderr)

        print("=" * 60)
        print("RETURN CODE:", result.returncode)

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

    ############################################################
    # Find Output
    ############################################################

        runtime_output = os.path.join(
            self.save_root,
            sample + ".mp4"
        )

        if not os.path.isfile(runtime_output):

            fallback = os.path.join(
                self.save_root,
                "sample.mp4"
            )

            if os.path.isfile(fallback):
                runtime_output = fallback
            else:

                files = [
                    f for f in os.listdir(self.save_root)
                    if f.lower().endswith(".mp4")
                ]

                if len(files) == 0:
                    raise FileNotFoundError(
                        f"No output video found inside:\n{self.save_root}"
                    )

                runtime_output = os.path.join(
                    self.save_root,
                    files[0]
                )

    ############################################################
    # Copy Final Video
    ############################################################

        if job_id:
            update_job(job_id, "Copying Final Video", 97)

        if progress_callback:
            progress_callback("Copying Final Video", 97)

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        shutil.copy2(runtime_output, output_path)

        print("=" * 60)
        print("TalkLip Finished Successfully")
        print(runtime_output)
        print("=" * 60)

        if job_id:
            update_job(job_id, "Completed", 100)

        if progress_callback:
            progress_callback("Completed", 100)

        return output_path

talklip_model = TalkLipModel()