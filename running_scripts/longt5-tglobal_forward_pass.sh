cd ..
python forward_pass.py \
	--config-file flaxformer/t5x/configs/longt5/models/longt5_1_1_transient_global_base.gin \
	--checkpoint-dir google-checkpoints/LongT5-TGlobal-Base \
	--hf-model-path Stancld/LongT5-TGlobal-Base
cd -
