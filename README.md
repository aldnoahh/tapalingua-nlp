# tapalingua-nlp

___Baseline Document: https://docs.google.com/document/d/1OnXwndZ6SDP10-hM4SP-Hq5bE0bQWijTAQlbBrowWIY/edit?usp=sharing
<br />
This script used Facebook`s Wav2vec2 for converting audio from video files to word sequences.

The following are the dependencies:<br />
--python 3.7<br />
--pytorch 1.9.0<br />
--librosa 0.8.1<br />
--transformers 4.8.2<br />
--ffmpeg<br />
<br />
<br />
1. Ensure the version of Python is 3.7 as this script and its dependencies require it. <br />
2. Clone this repo.
3. Then change directory:
      cd tapalingua-nlp <br />
4. Install dependencies using: sudo bash setup.sh <br />
5. If the dependencies are not installed, you may need to manually install them using: python3.7 -m pip install xxxx  <br />
6. After installation of the above dependencies, you need to save the .mp4 video in this directory, and run the script as: python3.7 analysis.py record.mp4  <br />
7. First run will take time, as the script downloads weight files for the model which is nearly 1.35 Gb.  <br />
8. A wav file for the video file is also generated stored in the same directory. <br />
9. After successful run, you will get .txt file in the same directory with the video. Example: for record.mp4, record.txt is generated. <br />

The txt file is in the csv form:<br /> <br />
transcription<br />
------Full Transcription of Speech------   <br />
word,start timestamp, end timestamp, pause<br />
---,--------,---------,---------
