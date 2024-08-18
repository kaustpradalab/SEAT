# SEAT Attention 
This is our implementation for AAAI'22 submission *SEAT: Stable and Explainable Attention*. We provide detailed environments setup and script for quickly running our experiments.


# Environments Setup
1. We use Python 3.6 in our experiments. Please Use the following command to install the dependencies:
```shell
pip install -r ./attention/requirements.txt
python -m spacy download en
```

2. Preprocess the dataset using the following command.
```shell
python ./attention/preprocess/prepare_data_w2v.py
python ./attention/preprocess/prepare_data_bert.py
```

3. Export current dir to PYTHONPATH
```shell
export PYTHONPATH=$PYTHONPATH:"the-dir-of-root-of-repo"
```

4. build the git repository
```shell
#In the root directory
git init
git commit
```

# Run Our Main Experiments
## 1. Vanila Attention
First train the baseline models and then the model ckpt and attention score will be saved in the output directory
```shell
python ./attention/train.py --dataset sst --data_dir . --output_dir ./outputs/ --attention tanh --encoder simple-rnn --exp_name baseline --train_mode std_train --bsize 32 --n_epoch 20  --seed 2 --lr 0.01
```

## 2. Other Baseline Methods and Ours

### SEAT(Ours)
```shell
python ./attention/train.py --dataset sst --data_dir . --output_dir ./outputs/ --attention tanh \
--encoder simple-rnn \
--exp_name baseline --lambda_1 1 --lambda_2 1000 --pgd_radius 0.001 --x_pgd_radius 0.01 \
--K 7 --seed 2 --train_mode adv_train --bsize 32 --n_epoch 20 --lr 0.01  --method ours \
--eval_baseline
```

### Other Baseline Methods
- Please replace method slot with choice from ['word-at', 'word-iat', 'attention-iat', 'attention-at', 'attention-rp'] to evalute baseline methods.

```shell
python ./attention/train.py --dataset sst --data_dir . --output_dir ./outputs/ --attention tanh \
--encoder simple-rnn \
--exp_name baseline --lambda_1 1 --lambda_2 1000 --pgd_radius 0.001 --x_pgd_radius 0.01 \
--K 7 --seed 2 --train_mode adv_train --bsize 32 --n_epoch 20 --lr 0.01  --method [the-method-you-wanna-test] \
--eval_baseline
```

## Code Implementation References
- Thanks [code](https://github.com/sarahwie/attention) provided by Sarthak Jain & Byron Wallace for their paper *[Attention is not not Explanation](https://arxiv.org/abs/1908.04626)* 
