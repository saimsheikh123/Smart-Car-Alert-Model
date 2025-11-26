AST (Hugging Face) SageMaker packaging

Contents expected in model.tar.gz root:
- config.json
- model.safetensors (or pytorch_model.bin)
- preprocessor_config.json
- class_names.json (optional, provides label ordering)
- code/inference.py (this folder)
- code/requirements.txt (libs for audio IO)

Build on Windows PowerShell:

1) From repo root:
   cd cmpe-281-models\sagemaker\ast_hf

2) Package tarball (pulling artifacts from ast_6class_model):
   tar -czf model.tar.gz -C ..\..\ast_6class_model config.json model.safetensors preprocessor_config.json class_names.json -C ..\ast_hf code

If tar lacks -C support on your system, copy the files into this folder and run:
   tar -czf model.tar.gz config.json model.safetensors preprocessor_config.json class_names.json code

Deploy using sagemaker/deploy_to_sagemaker.py with --variant ast
