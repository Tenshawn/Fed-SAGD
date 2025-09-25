# FedSAGD


## Requirements

Some pip packages are required by this library, and may need to be installed. For more details, see `requirements.txt`. We recommend running `pip install --requirement "requirements.txt"`.

## Task and dataset summary

Note that we put the dataset under the directory .\fl-master\Folder



| Directory        | Model                               | Task Summary              |
|------------------|-------------------------------------|---------------------------|
| CIFAR-10         | CNN (with two convolutional layers) | Image classification      |
| CIFAR-100        | Resnet18                            | Image classification      |
| EMNIST           | Logistic Model                      | Digit recognition         |
| Shakespeare      | RNN with 2 LSTM layers              | Next-character prediction |



## Training
In this code, we compare 13 optimization methods: FedANAG, MimeLite, SCAFFOLD, FedDyn, FedProx, FedAvg, FedAdam, FedCM, FedAvgM, FedLNAG, FedANAG, FedACG, and FedSAGD. Those methods use vanilla SGD on clients. To recreate our experimental results for each optimizer, for example, for 200 clients and 2% participation rate, on the cifar100 data set with Dirichlet (0.3) split, run those commands for different methods:


FedSAGD:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 200 --frac 0.02 --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1  --mu 0.01 --beta_0 0.1 --beta_ 0.9 --lammda 0.001 --lr_decay 0.998 --method FedSAGD --filepath ours_CIFAR100_seed200.txt
```
--beta_ is hyperparameters for FedSAGD.
--lammda is hyperparameters for FedSAGD.
--mu is hyperparameters for FedSAGD.

FedANAG:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 200 --frac 0.02 --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1 --beta_ 0.9 --weigh_delay 0.001 --lr_decay 0.998 --method fed_nesterov --filepath FedANAG_CIFAR100_seed200.txt
```
--beta_ is hyperparameters for FedANAG.

MimeLite:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 200 --frac 0.02 --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1 --beta_ 0.9 --weigh_delay 0.001 --lr_decay 0.998 --method mime --filepath MimeLite_CIFAR10_CIFAR100_seed200.txt
```
--beta_ is hyperparameters for MimeLite.

FedCM:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 200 --frac 0.02 --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1 --beta_ 0.9 --weigh_delay 0.001 --lr_decay 0.998 --method fedcm --filepath FedANAG_CIFAR100_seed200.txt
```
--beta_ is hyperparameters for FedCM.

FedAdam:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 200 --frac 0.02 --dataset CIFAR100 --local_ep 5 --local_bs 50 --globallr 0.1 --bs 50 --rule_arg 0.3 --lr 0.1 --beta_ 0.9 --weigh_delay 0.001 --lr_decay 0.998 --method fedadam --filepath FedANAG_CIFAR100_seed200.txt
```
--beta_ is hyperparameters for FedAdam.


FedACG:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 200 --frac 0.02 --dataset CIFAR100 --local_ep 5 --local_bs 50 --globallr 0.1 --bs 50 --rule_arg 0.3 --lr 0.1 --beta_ 0.9 --weigh_delay 0.001 --mu --0.01 --lr_decay 0.998 --method fedacg --filepath FedANAG_CIFAR100_seed200.txt
```
--mu is hyperparameters for FedACG.
--beta_ is hyperparameters for FedACG.

FedANAG:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 200 --frac 0.02 --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1 --beta_ 0.9 --weigh_delay 0.001 --lr_decay 0.998 --method fed_nesterov --filepath FedANAG_CIFAR100_seed200.txt
```
--beta_ is hyperparameters for FedANAG.

SCAFFOLD:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 200 --frac 0.02 --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1 --weigh_delay 0.001 --lr_decay 0.998 --method scaffold --filepath scaffold_CIFAR100_seed200.txt
```

FedDyn:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 200 --frac 0.02 --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1 --weigh_delay 0.001 --lr_decay 0.998 --coe 0.1 --method feddyn --filepath feddyn_CIFAR100_CIFAR100_seed200.txt
```
--coe is hyperparameters for FedDyn.

FedProx:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000 --num_users 200 --frac 0.02 --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1 --beta_ 0 --weigh_delay 0.001 --mu 0.001 --lr_decay 0.998 --method fedprox --filepath fedprox_CIFAR100_seed200.txt
```
``--mu``` is hyperparameters for FedProx.

FedAvg:
```
python main_fed.py --seed 100 --gpu 0 --epochs 2000  --num_users 200 --frac 0.02 --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1 --weigh_delay 0.001 --lr_decay 0.998 --method fedavg --filepath fedavg_CIFAR100_seed200.txt
```


FedAvgM:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000   --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1 --beta_ 0.9 --weigh_delay 0.001 --lr_decay 0.998 --method fedavgm --filepath fedavgm_CIFAR100_seed200.txt
```
--beta_ is hyperparameters for FedAvgM.

FedLNAG:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000 --num_users 200 --frac 0.02 --dataset CIFAR100 --local_ep 5 --local_bs 50 --bs 50 --rule_arg 0.3 --lr 0.1 --beta_ 0.9 --weigh_delay 0.001  --lr_decay 0.998 --method fed_localnesterov --filepath fed_localnesterov_CIFAR100_seed200.txt
```
--beta_ is hyperparameters for FedLNAG.






