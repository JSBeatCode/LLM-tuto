https://claude.ai/share/617b231a-4b72-40bb-82b5-6d0f8f89dee5
13_continousPretrain 과 같은 claude chat 방을 씀.

===
[uv 생성]
uv venv .venv-cuda --python 3.11

===
[requirements.txt 생성]
pipreqs . --force

requirements.txt 에서 torch 관련 지우기
.venv-cuda\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt

필요에 따라 버전 낮추기:
예) 
numpy==2.1.3
trl==0.9.6
===
[uv 실행]
cd C:\Users\jsd\Documents\programming\LLM-AI-practice_V2\homePractice\14_fineTuningSFT\
.venv-cuda/Scripts/Activate.ps1
python "C:\Users\jsd\Documents\programming\LLM-AI-practice_V2\homePractice\14_fineTuningSFT\14_train.py"