# CUDA_VISIBLE_DEVICES=3 python cross_layer_predict_simple.py --layer2 24
# CUDA_VISIBLE_DEVICES=3 python cross_layer_predict_simple.py --layer2 26
# CUDA_VISIBLE_DEVICES=3 python cross_layer_predict_simple.py --layer2 28
# CUDA_VISIBLE_DEVICES=3 python cross_layer_predict_simple.py --layer2 30

# CUDA_VISIBLE_DEVICES=3 python ACL_evaluate_ppl_blockwise.py --seed 1
# CUDA_VISIBLE_DEVICES=3 python ACL_evaluate_ppl_blockwise.py --seed 2
# CUDA_VISIBLE_DEVICES=3 python ACL_evaluate_ppl_blockwise.py --seed 3
# CUDA_VISIBLE_DEVICES=3 python ACL_evaluate_ppl_blockwise.py --seed 4

CUDA_VISIBLE_DEVICES=3 python ACL_evaluate_ppl_block_filter.py --block 1