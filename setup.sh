# Clone t5x, flaxformer and my forked transformers repo
git clone https://github.com/google-research/t5x.git
git clone https://github.com/google/flaxformer.git
git clone https://github.com/stancld/transformers.git

# Download checkpoint for the local base model
gsutil -m cp -r "gs://t5-data/pretrained_models/t5x/longt5/local_base/checkpoint_1000000" .
mv checkpoint_1000000 LongT5-Local-Base

# Install t5x from the source
cd ./t5x
python -m pip install -e .
cd ..

# Instal transformers from the source on the branch where the LongT5 is developed
cd ./transformers
git checkout new_model/LongT5
python -m pip install -e .
cd ..

# Install Handle flaxformer
mv ./flaxformer/flaxformer/ _flaxformer/
rm -rf flaxformer/
mv _flaxformer/ flaxformer/

# Install additional dependencies
python -m pip install -r flaxformer_requirements.txt
