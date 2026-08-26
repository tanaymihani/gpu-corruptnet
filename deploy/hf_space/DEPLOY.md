# Deploy the demo to a free Hugging Face Space

Result: a public URL like `https://huggingface.co/spaces/<you>/gpu-corruptnet` you can put on
your resume/GitHub.

## 1. Get the model from Colab (one training run)
Re-run the Colab notebook (cell 1 to pull latest code, then the ResNet train+calibrate cell),
then download both files from Drive:
```python
from google.colab import files
import glob
files.download(sorted(glob.glob('/content/drive/MyDrive/corruptnet_runs/model_*.pt'))[-1])
files.download(sorted(glob.glob('/content/drive/MyDrive/corruptnet_runs/calibration_*.json'))[-1])
```

## 2. Create the Space
- Go to https://huggingface.co/new-space → name `gpu-corruptnet`, **SDK: Streamlit**, Public.
- Install the CLI and log in (needs a token from https://huggingface.co/settings/tokens):
  ```bash
  pip install -U huggingface_hub
  huggingface-cli login
  ```

## 3. Push the files + model
```bash
git clone https://huggingface.co/spaces/<your-username>/gpu-corruptnet hf-space
cd hf-space

# copy the deploy files from this repo:
cp /path/to/gpu-corruptnet/deploy/hf_space/{app.py,requirements.txt,README.md,.gitattributes} .

# add the model + calibration you downloaded (rename to fixed names):
cp ~/Downloads/model_resnet50_*.pt        model.pt
cp ~/Downloads/calibration_resnet50_*.json calibration.json

git lfs install
git lfs track "*.pt"
git add .
git commit -m "GPU-CorruptNet demo"
git push
```

The Space builds automatically (a few minutes) and serves the demo at its public URL. If it
shows "No model bundled," the `model.pt` didn't upload via LFS — confirm `git lfs track "*.pt"`
ran before `git add`.
