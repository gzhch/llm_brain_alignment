# CUDA_VISIBLE_DEVICES=2 python cross_layer_predict_simple.py --layer2 16
# CUDA_VISIBLE_DEVICES=2 python cross_layer_predict_simple.py --layer2 18
# CUDA_VISIBLE_DEVICES=2 python cross_layer_predict_simple.py --layer2 20
# CUDA_VISIBLE_DEVICES=2 python cross_layer_predict_simple.py --layer2 22

CUDA_VISIBLE_DEVICES=2 python ACL_evaluate_ppl_block_filter.py --block 3
CUDA_VISIBLE_DEVICES=2 python ACL_evaluate_ppl_block_filter.py --block 4