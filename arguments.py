
import argparse

parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument("--train_mode", type=str, choices=['std_train', 'adv_train'], default="std_train")
parser.add_argument('--dataset', type=str)
parser.add_argument("--data_dir", type=str, default=".")
parser.add_argument("--output_dir", type=str,default="./outputs/")
parser.add_argument('--encoder', type=str, choices=[ 'average', 'lstm','simple-rnn','bert'], default="lstm")
parser.add_argument('--attention', type=str, choices=['tanh', 'frozen', 'pre-loaded'], required=False)
parser.add_argument('--n_epoch', type=int, required=False, default=40)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--gold_label_dir', type=str, required=False)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--lmbda', type=float, required=False)
parser.add_argument('--adversarial', action='store_const', required=False, const=True)
parser.add_argument("--bsize", type=int, default=16)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--save_on_metric", type=str, default="roc_auc")

parser.add_argument('--pgd_radius', type=float,default=0.1)
parser.add_argument('--pgd_step', type=float,default=10)
parser.add_argument('--pgd_step_size', type=float,default=0.02)
parser.add_argument('--pgd_norm_type', type=str,default="l-infty")

parser.add_argument('--x_pgd_radius', type=float,default=0.05)
parser.add_argument('--x_pgd_step', type=float,default=10)
parser.add_argument('--x_pgd_step_size', type=float,default=0.01)
parser.add_argument('--x_pgd_norm_type', type=str,default="l-infty")

parser.add_argument('--lambda_1', type=float, default=1e-2)
parser.add_argument('--lambda_2', type=float, default=1e-2)
parser.add_argument('--lambda_3', type=float, default=1e-2)
parser.add_argument('--lambda_4', type=float, default=1e-2)
parser.add_argument('--exp_name', type=str, default="debug")
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--topk_prox_metric', type=str, choices=['l1', 'l2',"kl-full", 'jsd-full',"kl-topk", 'jsd-topk'], default='l1')


parser.add_argument("--wandb_entity", type=str, default="yixin")

parser.add_argument("--parallel", action="store_true", help="use parallel data loader")

parser.add_argument("--use_attention", action="store_true", help="use attention",default=True)

parser.add_argument("--lr", type=float, default=1e-3)

parser.add_argument("--eval_baseline", action="store_true", help="evaluate baseline")

parser.add_argument("--method", type=str, choices=['word-at', 'word-iat', 'attention-iat', 'attention-at', 'attention-rp', 'ours'], default=None)

parser.add_argument("--perturbed_input", action='store_true', help="perturbed input")

args, extras = parser.parse_known_args()