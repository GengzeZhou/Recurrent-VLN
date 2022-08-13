name=Reborn-1.0

flag="--vlnbert prevalent

      --aug data/prevalent/prevalent_aug.json
      --test_only 0
      --log_every 1000

      --world_size 2

      --train train

      --features places365
      --maxAction 20
      --batchSize 16
      --feedback sample
      --lr 1e-5
      --iters 300000
      --optim adamW

      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

mkdir -p snap/$name
# CUDA_VISIBLE_DEVICES=2 python r2r_src/train.py $flag --name $name
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM r2r_src/train.py $flag --name $name
