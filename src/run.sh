#DPID
python main.py --loss 1*SSIM --data_name DIV2K/HighRes --data_range 1-400/1-12 --epochs 200 --lr_decay 20 --lr 0.001 --reset n --scales 4 --patch_size 512 --n_ResDenseBlock 4 --n_dense_layer 7
python main.py --loss 1*SSIM --data_name DIV2K/HighRes --data_range 1-70/1-12 --epochs 200 --lr_decay 00 --lr 0.001 --reset n --scales 4 --patch_size 512

#DPIDU
python main.py --loss 1*SSIM --data_name DIV2K/HighRes --data_range 1-110/1-12 --epochs 1000 --lr_decay 40 --lr 0.001 --reset y --scales 4 --patch_size 512 --n_ResDenseBlock 8 --n_feature 40  --n_dense_layer 10 --model DPIDU

#DPIDSK
python main.py --loss 1*SSIM --data_name DIV2K/HighRes --data_range 1-110/1-12 --epochs 500 --lr_decay 30 --lr 0.001 --reset y --scales 4 --patch_size 512 --n_ResDenseBlock 8 --n_feature 40  --n_dense_layer 10 --model DPIDSK

#MSK
python main.py --loss 1*SSIM --data_name DIV2K/HighRes --data_range 1-110/1-12 --epochs 500 --lr_decay 30 --lr 0.001 --reset y --scales 4 --patch_size 512 --n_ResDenseBlock 16 --n_feature 40  --n_dense_layer 8 --model MSK --batch_size 2


#REC0
python main.py --loss 1*SSIM --data_name DIV2K/HighRes --data_range 1-110/1-12 --epochs 500 --lr_decay 30 --lr 0.001 --reset y --scales 4 --patch_size 512 --n_ResDenseBlock 8 --n_feature 40  --n_dense_layer 10 --model REC --batch_size 2

#REC
python main.py --loss 1*SSIM --data_name DIV2K/HighRes --data_range 1-850/1-12 --epochs 500 --lr_decay 200 --lr 0.001 --reset y --scales 4 --patch_size 512 --n_ResDenseBlock 8 --n_feature 40  --n_dense_layer 10 --model REC --batch_size 2

#DUNET_1
python main.py --loss 1*SSIM --data_name DIV2K/HighRes --data_range 1-850/1-12 --epochs 500 --lr_decay 50 --lr 0.001 --reset y --scales 4 --patch_size 512 --n_feature 64 --model DUNET --batch_size 2

#DUNET_2
python main.py --loss 1*SSIM --data_name DIV2K/HighRes --data_range 1-850/1-12 --epochs 500 --lr_decay 20 --lr 0.001 --reset y --scales 4 --patch_size 512 --n_feature 64 --model DUNET --batch_size 2
