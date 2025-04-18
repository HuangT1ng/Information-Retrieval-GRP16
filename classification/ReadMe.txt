# --- Setting up virtual environment --- #

# create virtual environment
python -m venv nlu_classification_env

# activate virtual environment
nlu_classification_env\Scripts\activate

# --- Installing Packages --- #

pip install jupyter pandas numpy scikit-learn nltk matplotlib seaborn notebook tensorflow contractions




# Download NLTK data
--- Step 1: ---
Create "nltk_data" folder in virtual environment

--- Step 2: ---
Run NLTK download code...

import nltk
nltk.download()

Pop-up window should pop up. Change download destination and download all packages.