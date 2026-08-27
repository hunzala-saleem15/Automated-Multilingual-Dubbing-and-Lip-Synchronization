# Automated-Multilingual-Dubbing-and-Lip-Synchronization
Automatic Multilingual Dubbing and Lip Synchronization is a research-oriented project focused on generating multilingual dubbed videos while preserving natural visual synchronization between speech and lip movements.
The system processes short videos and converts them into multiple target languages through an end-to-end pipeline involving:
-	Automatic Speech Recognition (ASR)
-	Machine Translation (MT)
-	Text-to-Speech Synthesis (TTS)
-	Lip Synchronization

The primary objective is to create a scalable and modular multilingual video dubbing framework that maintains:
-	Speaker identity
-	Gender characteristics
-	Approximate age
-	Audio-video synchronization

# Problem Statement
Language is main barrier in learning as most of the content is available in foreign languages. Although watching content online facilitates with subtitles feature but Subtitles can be difficult to follow because viewers must read and watch at the same time. On the other hand, dubbing requires extra time, resources, and voice actors, which increases production cost. As a result, media creators must produce multiple versions of the same content for different languages, making the process slow and expensive.
# Motivation
Online learning and digital media are becoming more popular, but understanding content in foreign languages is still difficult for many users. Subtitles help, but they require continuous reading along with watching visuals, which can reduce focus and make the experience less smooth. On the other hand, dubbing provides a better viewing experience but is costly and takes a lot of time due to the need for voice actors and production work.
A survey was conducted to understand user preferences, and most participants showed a clear preference for dubbed or synthetic voice content because it allows them to fully focus on visuals without distraction. About 70% of users supported the use of synthetic dubbing in educational videos, while 30% still preferred subtitles.
These responses highlight the need for a simpler, more efficient, and user-friendly way to consume multilingual content, especially in education

# Base Papers

| Base Papers | Titles |
|---|---|
| Automatic Speech Recognition | *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations* |
| Machine Translation | *NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs* |
| Text To Speech | *XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model* |
| Lip Synchronisation | *Seeing what you said: Talking face generation guided by a lip reading expert.* |

# Architecture Overview
![image alt](https://github.com/hunzala-saleem15/Automated-Multilingual-Dubbing-and-Lip-Synchronization/blob/main/Architecture.jpeg?raw=true)

# Datasets

## Machine Translation Datasets

| Dataset | Link |
|---|---|
| Europarl | https://www.statmt.org/europarl/ |
| CoVoST-2 | https://huggingface.co/datasets/facebook/covost2 |
| WMT News | https://huggingface.co/datasets/wmt/wmt19 |
| OPUS-100 | https://opus.nlpl.eu/opus-100.php |

## TTS Datasets

| Dataset | Link |
|---|---|
| LibriTTS | https://www.openslr.org/60/ |
| LjSpeech | https://keithito.com/LJ-Speech-Dataset/ |
| CSS10 | https://github.com/Kyubyong/css10 |
| Common Voice | https://www.kaggle.com/datasets/mozillaorg/common-voice |

## Automatic Speech Recognition Datasets

| Dataset | Link |
|---|---|
| LibriSpeech | https://huggingface.co/datasets/openslr/librispeech_asr |
| Common Voice | https://www.kaggle.com/datasets/mozillaorg/common-voice |
| FLEURS | https://huggingface.co/datasets/google/fleurs |
| GigaSpeech | http://huggingface.co/datasets/speechcolab/gigaspeech |

# Installation
## Clone Repository
- git clone https://github.com/hunzala-saleem15/Automated-Multilingual-Dubbing-and-Lip-Synchronization.git
- cd Automated-Multilingual-Dubbing-and-Lip-Synchronization

# Models
## Text_to_Speech
The models and the checkpoints are available here:
https://huggingface.co/hunzala-saleem/My-TTS-Models

## Automatic Speech Recognition
The models and the checkpoints are available here:
https://huggingface.co/Laika-Sarfraz11/ASR_Models

## Machine Translation
The models and the checkpoints are available here:

https://huggingface.co/NidaIlyas/mt_lmt-finetuned

https://huggingface.co/NidaIlyas/mt_lmt-finetuned-en-zh-balanced

https://huggingface.co/NidaIlyas/mt_lmt-lora-en-hi

https://huggingface.co/NidaIlyas/mt_lmt-finetuned-en-ar-fast

# Website

The project includes a web-based application consisting of both a frontend and a backend. The website provides an interface for users to upload videos, select the target language, process the video through the multilingual dubbing pipeline, and view the generated output.

## Frontend

The frontend of the website is responsible for providing the user interface and allowing users to interact with the multilingual dubbing system. It includes features such as:

- User account and authentication
- Video upload
- Target language selection
- Video processing
- Result visualization
- Downloading the generated dubbed video

The frontend is located in the `Website/client` folder of this repository.

### Frontend Screenshot

![image alt](https://github.com/hunzala-saleem15/Automated-Multilingual-Dubbing-and-Lip-Synchronization/blob/main/frontend.png?raw=true)

## Backend

The backend is responsible for handling the API requests, video processing, AI model inference, and communication between the frontend and the different components of the dubbing pipeline.

The backend is located in the `Website/server` folder of this repository and is implemented using FastAPI.

The backend integrates the following components:

- Automatic Speech Recognition (ASR)
- Machine Translation (MT)
- Text-to-Speech (TTS)
- Lip Synchronization
- Audio and video processing

The **frontend and backend of the website are successfully running and integrated** as part of the complete multilingual dubbing and lip synchronization system.

## Model Weights

The model weights required by the website are hosted on Hugging Face. The repository contains the required model folders for:

- NLLB
- TalkLip
- Whisper
- XTTS

The complete model weights are available here:

https://huggingface.co/hunzala-saleem/Automated-Multilingual-Dubbing-and-Lip-Synchronization

# Team Members

- Hanzala Saleem
- Nida Ilyas
- Ayesha Sarfraz
- Laika Sarfraz

# Acknowledgements
The authors would like to thank the developers and researchers behind the open-source ASR, Machine Translation, Text-to-Speech, and Lip Synchronization models utilized in this project. We also acknowledge the maintainers of all publicly available datasets used for training, evaluation, and experimentation throughout this research.
Special thanks to GIFT University for providing GPU infrastructure, computational facilities, and technical support, which played a crucial role in successfully conducting experiments and developing the complete multilingual dubbing and lip synchronization pipeline.
We further appreciate the contributions of the open-source research community whose work in speech processing, natural language processing, computer vision, and generative AI made this project possible.
# License

This project is intended for:

- Academic Research
- Educational Purposes
- Non-commercial experimentation

Please ensure compliance with:

- Dataset licenses
- Pretrained model licenses
- Third-party repository licenses
