
## Data

1. Download [training data](https://utexas.box.com/shared/static/3go1g4gcdar2cntjit2knz5jwr3mvxwe.zip) and extract contents into new `data_train/` directory. Stimulus data for `train_stimulus/` and response data for `train_response/[SUBJECT_ID]` can be downloaded from [OpenNeuro](https://openneuro.org/datasets/ds003020/).
 
2. Download [test data](https://utexas.box.com/shared/static/ae5u0t3sh4f46nvmrd3skniq0kk2t5uh.zip) and extract contents into new `data_test/` directory. Stimulus data for `test_stimulus/[EXPERIMENT]` and response data for `test_response/[SUBJECT_ID]` can be downloaded from [OpenNeuro](https://openneuro.org/datasets/ds004510/).

## Code

```
# align the fMRI signals to LLM activations
python align_reverse.py \
    --subject [SUBJECT] \
    --layer [LAYER] \
    --act_name [ACT] \
    --window [WINDOW]

# align the LLM activations of different layers to each other
python align_llm.py \
    --subject [SUBJECT] \
    --layer [LAYER] \
    --layer2 [LAYER2] \
    --act_name [ACT] \
    --window [WINDOW]
```
