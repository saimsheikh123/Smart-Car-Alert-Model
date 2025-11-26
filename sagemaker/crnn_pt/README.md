CRNN (PyTorch) SageMaker packaging

Place the checkpoint file at model root (default name multi_audio_crnn.pth). Optionally include class_names.json for label ordering.

model.tar.gz layout:
- multi_audio_crnn.pth
- class_names.json (optional)
- code/inference.py
- code/requirements.txt

Build on Windows PowerShell from this folder:
   tar -czf model.tar.gz multi_audio_crnn.pth class_names.json code

Deploy using sagemaker/deploy_to_sagemaker.py with --variant crnn
