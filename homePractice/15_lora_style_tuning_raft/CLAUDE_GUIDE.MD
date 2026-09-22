https://claude.ai/share/ad56df2e-cfbe-457f-96f2-712d9f679c6f

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
cd C:\Users\jsd\Documents\programming\LLM-AI-practice_V2\homePractice\15_lora_style_tuning_raft\
.venv-cuda/Scripts/Activate.ps1
python "C:\Users\jsd\Documents\programming\LLM-AI-practice_V2\homePractice\15_lora_style_tuning_raft\15_lora_style_tuning_raft.py"