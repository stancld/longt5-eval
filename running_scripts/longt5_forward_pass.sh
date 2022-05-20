cd ..
python forward_pass.py \
	--config-file flaxformer/t5x/configs/longt5/models/longt5_1_1_base.gin \
	--checkpoint-dir google-checkpoints/LongT5-Local-Base \
	--hf-model-path Stancld/LongT5-Local-Base
cd -
