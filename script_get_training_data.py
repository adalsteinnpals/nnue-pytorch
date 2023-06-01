import os
import sys
import subprocess
import gdown


training_data_url = "https://drive.google.com/file/d/1RFkQES3DpsiJqsOtUshENtzPfFgUmEff/view?usp=sharing"

if not os.path.exists("training_data.7z"):
    gdown.download(training_data_url, quiet=False,fuzzy=True)

# unzip training data
if not os.path.exists("training_data.binpack"):
    subprocess.run(["7z", "x", "training_data.7z"])

