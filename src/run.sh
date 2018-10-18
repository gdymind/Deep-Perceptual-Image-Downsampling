python main.py --loss 1*MSE+1*SSIM --n_shallow_feature 8 --n_feature 8 --growth_rate 8 --data_range 2-2/2-2 --epochs 2000 --reset y
python main.py --loss 1*MSE --n_shallow_feature 8 --n_feature 8 --growth_rate 8 --data_range 3-3/3-3 --epochs 45000 --reset y  --lr_decay 10000

python main.py --loss 1*VGG --n_shallow_feature 8 --n_feature 8 --growth_rate 8 --data_range 1-1/1-1 --epochs 2000 --reset y