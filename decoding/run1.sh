# CUDA_VISIBLE_DEVICES=0 python cross_layer_predict_simple.py --layer2 0
# CUDA_VISIBLE_DEVICES=0 python cross_layer_predict_simple.py --layer2 2
# CUDA_VISIBLE_DEVICES=0 python cross_layer_predict_simple.py --layer2 4
# CUDA_VISIBLE_DEVICES=0 python cross_layer_predict_simple.py --layer2 6

CUDA_VISIBLE_DEVICES=0 python ACL_evaluate_ppl_block_filter.py --block 4