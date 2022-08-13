import torch

import os
import time
import json
import random
import numpy as np
from collections import defaultdict

from utils import timeSince, print_progress
from utils import setup_seed, read_img_features, prepare_dataset

from distributed import init_distributed, is_default_gpu
from distributed import all_gather, merge_dist_results

from env import R2RBatch
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args

import warnings
warnings.filterwarnings("ignore")
from tensorboardX import SummaryWriter

log_dir = 'logs/%s/snap/' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
PLACE365_FEATURES = 'img_features/ResNet-152-places365.tsv'

if args.features == 'imagenet':
    features = IMAGENET_FEATURES
elif args.features == 'places365':
    features = PLACE365_FEATURES

print(args); print('')


def build_dataset(args, rank=0):

    # Load the env img features
    feat_dict = read_img_features(features, test_only=args.test_only)

    if args.test_only:
        featurized_scans = None
        val_env_names = ['val_train_seen']
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    # Create the training environment
    train_episodes = prepare_dataset(args, splits=['train'])
    train_env = R2RBatch(feat_dict, 
        batch_size=args.batchSize, seed=971214+rank,
        splits=['train'],
        data=train_episodes, )
    # Load the augmentation data
    aug_path = args.aug
    train_aug_episodes = prepare_dataset(args, splits=[aug_path])
    aug_env = R2RBatch(feat_dict, 
        batch_size=args.batchSize, seed=971214+rank,
        splits=[aug_path], 
        data=train_aug_episodes, name='aug', )

    # Setup the validation data
    val_envs = {}
    for split in val_env_names:
        val_episodes = prepare_dataset(args, splits=[split], allocate_rank=rank)
        val_envs[split] = (
            R2RBatch(
                feat_dict, 
                batch_size=args.batchSize, seed=971214+rank,
                splits=[split],
                data=val_episodes,
            ),
            Evaluation([split], featurized_scans, data=val_episodes),
        )

    return train_env, val_envs, aug_env


''' train the listener '''
def train(train_env, val_envs, aug_env=None, n_iters=1000, log_every=10, rank=-1):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        writer = SummaryWriter(log_dir=log_dir)
        record_file = open('./logs/%s/'%(args.name) + 'train_logs.txt', 'a')
        record_file.write(str(args) + '\n\n')
        record_file.close()

    listner = Seq2SeqAgent(
        train_env, results_path="", episode_len=args.maxAction, rank=rank)

    start_iter = 0
    if args.load is not None:
        if args.aug is None:
            start_iter = listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration ".format(args.load, start_iter))
        else:
            load_iter = listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration ".format(args.load, load_iter))

    start = time.time()
    print('\nListener training starts, start iteration: %s' % str(start_iter))

    best_val = {'val_unseen': {"sr": 0., "spl": 0., "state":""}}

    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=args.feedback)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                args.ml_weight = 0.2
                listner.train(1, feedback=args.feedback)

                # Train with Augmented data
                listner.env = aug_env
                args.ml_weight = 0.2
                listner.train(1, feedback=args.feedback)

                if default_gpu:
                    print_progress(jdx+1, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)
            length = max(len(listner.logs['critic_loss']), 1)
            critic_loss = sum(listner.logs['critic_loss']) / total
            RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/RL_loss", RL_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)

        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            result = listner.get_results()
            all_results = merge_dist_results(all_gather(result))

            if default_gpu:
                score_summary, _ = evaluator.score(all_results)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.4f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)

                # select model by spl+sr
                if env_name in best_val:
                    if score_summary['spl'] + score_summary['success_rate'] >= best_val[env_name]['spl'] + best_val[env_name]['sr']:
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['sr'] = score_summary['success_rate']
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        listner.save(idx, os.path.join('./logs/%s'%(args.name), "state_dict", "best_%s" % (env_name)))
                listner.save(idx, os.path.join('./logs/%s'%(args.name), "state_dict", "latest_dict"))

                record_file = open('./logs/%s/eval.txt'%(args.name), 'a')
                record_file.write(loss_str + '\n')
                record_file.close()

        if default_gpu:
            print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                iter, float(iter)/n_iters*100, loss_str)))
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])
                record_file = open('./logs/%s/eval.txt'%(args.name), 'a')
                record_file.write('BEST RESULT TILL NOW: ' + env_name + ' | ' + best_val[env_name]['state'] + '\n')
                record_file.close()


def valid(train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent = Seq2SeqAgent(
        train_env, results_path="", episode_len=args.maxAction, rank=rank)

    print("Loaded the listener model at iter %d from %s" % (
        agent.load(args.load), args.load))

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        result = agent.get_results()
        all_results = merge_dist_results(all_gather(preds))

        if default_gpu:
            if env_name != '':
                score_summary, _ = evaluator.score(all_results)
                loss_str = "Env name: %s" % env_name
                for metric,val in score_summary.items():
                    loss_str += ', %s: %.4f' % (metric, val)
                print(loss_str)
                record_file = open('./logs/%s/eval.txt'%(args.name), 'a')
                record_file.write(loss_str + '\n')
                record_file.close()
                
            if args.submit:
                json.dump(
                    all_results,
                    open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )


if __name__ == "__main__":

    setup_seed()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    train_env, val_envs, aug_env = build_dataset(args, rank=rank)

    if args.train == 'train':
        train(
            train_env, val_envs, aug_env=aug_env, 
            n_iters=args.iters, log_every=args.log_every, rank=rank,
        )
    elif args.train == 'valid':
        valid(val_envs, rank=rank)
    else:
        assert False
