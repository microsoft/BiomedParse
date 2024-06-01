# Customer Operator [only need training deformable vision encoder]
cd modeling/vision/encoder/ops && sh make.sh && cd ../../../../

# System Package [only need for demo in SEEM]
sudo apt update
sudo apt install ffmpeg

#pip install gradio==3.44.4
#pip install openai-whisper
#pip install protobuf==3.20.*