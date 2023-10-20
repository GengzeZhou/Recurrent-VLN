name=VLNBERT-test-Prevalent

NUM_GPUS=1

flag="--vlnbert prevalent

      --submit 1
      --test_only 0
      --aug data/prevalent/prevalent_aug.json
      
      --world_size 1
      --train valid
      --load snap/VLNBERT-PREVALENT-final/state_dict/best_val_unseen

      --features places365
      --maxAction 15
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

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} r2r_src/train.py $flag --name $name
