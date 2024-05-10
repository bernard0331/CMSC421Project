@echo off


rem Windows

rem Standard requirements
pip install -r requirements.txt

rem Custom install commands for packages requiring specific sources
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
