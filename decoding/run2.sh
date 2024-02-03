# CUDA_VISIBLE_DEVICES=1 python cross_layer_predict_simple.py --layer2 8
# CUDA_VISIBLE_DEVICES=1 python cross_layer_predict_simple.py --layer2 10
# CUDA_VISIBLE_DEVICES=1 python cross_layer_predict_simple.py --layer2 12
# CUDA_VISIBLE_DEVICES=1 python cross_layer_predict_simple.py --layer2 14

CUDA_VISIBLE_DEVICES=1 python ACL_evaluate_ppl_block_filter.py --block 1
CUDA_VISIBLE_DEVICES=1 python ACL_evaluate_ppl_block_filter.py --block 2