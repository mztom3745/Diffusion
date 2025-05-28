python -u train.py \
  --save_dir /root/autodl-tmp \
  --data_path /root/autodl-pub/cifar-10 \
  --model_name unet_fm \
  --batch_size 512 \
  --num_epochs 50 \
  --resume_path /root/autodl-tmp/unet_fm_epoch10.pt
