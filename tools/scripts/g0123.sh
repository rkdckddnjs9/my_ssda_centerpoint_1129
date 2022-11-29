# nuscenes pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
  pretrain.py --launcher pytorch --cfg_file ./cfgs/nuscenes_models/pv_rcnn.yaml \
  --split train_1.00_1 --extra_tag split_1.00_1 --ckpt_save_interval 5 \
  --repeat 10 --workers 4 \
  --dbinfos nuscenes_dbinfos_10sweeps_train_1.00_1.pkl \
2>&1|tee ../output/LOG_nus.log &

# nuscenes train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
  train.py --launcher pytorch --cfg_file ./cfgs/nuscenes_models/pv_rcnn_ssl.yaml \
  --split train_0.10_5 --extra_tag split_0.10_5 --ckpt_save_interval 2 \
  --pretrained_model "/mnt/sdb/hojun/3DIoUMatch-PVRCNN/output/cfgs/nuscenes_models/pv_rcnn/split_0.10_3/ckpt/checkpoint_epoch_50.pth" \
  --repeat 5 --thresh '0.5,0.25,0.25' --sem_thresh '0.4,0.0,0.0' \
  --dbinfos nuscenes_dbinfos_train_0.10_5.pkl --workers 4 \
2>&1|tee ../output/LOG_nus.log &

# omega pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
  pretrain.py --launcher pytorch --cfg_file ./cfgs/omega_models/pv_rcnn.yaml \
  --split train_0.10_1 --extra_tag split_0.10_1 --ckpt_save_interval 5 \
  --repeat 10 \
  --dbinfos omega_dbinfos_1sweeps_train_0.10_1.pkl \
2>&1|tee ../output/LOG_ome_10_1.log &

# omega train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
  train.py --launcher pytorch --cfg_file ./cfgs/omega_models/pv_rcnn_ssl.yaml \
  --split train_0.10_1 --extra_tag split_0.10_1 --ckpt_save_interval 2 \
  --pretrained_model "/mnt/sdb/hojun/3DIoUMatch-PVRCNN/output/cfgs/omega_models/pv_rcnn/split_0.10_1/ckpt/checkpoint_epoch_50.pth" \
  --repeat 5 --thresh '0.5,0.25,0.25,0.25,0.25' --sem_thresh '0.4,0.0,0.0,0.0,0.0' \
  --dbinfos omega_dbinfos_train_0.10_1.pkl --workers 4 \
2>&1|tee ../output/LOG_ome_10_1.log &
