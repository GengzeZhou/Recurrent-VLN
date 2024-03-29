name=Reborn-1.0
NUM_GPUS=1

flag="--vlnbert prevalent

      --aug data/prevalent/prevalent_aug.json
      --test_only 0
      --log_every 2000

      --world_size 1

      --train train

      --features pano
      --feature_size 768
      --maxAction 20
      --batchSize 8
      --feedback sample
      --lr 1e-5
      --iters 300000
      --optim adamW

      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

# CUDA_VISIBLE_DEVICES=2 python r2r_src/train.py $flag --name $name
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} r2r_src/train.py $flag --name $name
