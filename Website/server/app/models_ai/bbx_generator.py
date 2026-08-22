import os
import subprocess


class BBXGenerator:

    def __init__(
        self,
        talklip_root=r"D:\Website\server\Talklip"
    ):
        self.talklip_root = talklip_root

    def generate(
        self,
        video_path,
        output_folder
    ):

        os.makedirs(output_folder, exist_ok=True)

        command = [

            "python",

            "preparation/bbx_extract.py",

            "--video_path",
            video_path,

            "--save_path",
            output_folder
        ]

        print("Generating Bounding Boxes...")

        subprocess.run(
            command,
            cwd=self.talklip_root,
            check=True
        )

        print("Bounding Boxes Generated.")

        return output_folder


bbx_generator = BBXGenerator()