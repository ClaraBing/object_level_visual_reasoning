import os
import numpy as np
np.random.seed(95)
import random
random.seed(95)
import argparse
import model.models as models
from inference import inference
# import ipdb

if __name__ == '__main__':
    # Possible models
    model_names = sorted(name for name in models.__dict__
                         if not name.startswith("__")
                         and callable(models.__dict__[name]))

    # Parser
    parser = argparse.ArgumentParser(description='Pytorch implementation: Object level visual reasoning in videos')
    parser.add_argument('--machine', type=str, required=True, help="Machine on which the experiment is running. e.g. macondo2")
    parser.add_argument('--gpu-id', type=int, required=True, help='GPU Id on which the experiment is running. Assuming single GPU for now.')
    parser.add_argument('--task-id', type=int, required=True, help='Task id of the experiment on a particular GPU.')
    parser.add_argument('--device', default='cuda:0',
                        help="A string specifying the device on which models and data reside. 'cpu' or 'cuda:x'")
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        default='orn_two_heads',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: orn_two_heads')
    parser.add_argument('--use-vo-branch', type=int, default=0,
                        help='Whether to add another branch for verb-object. Diff from GCN.')
    parser.add_argument('--use-obj-rel', type=int, default=1)
    parser.add_argument('--gcn-version', type=str, choices=['None', 'v1', 'v2'], default='v1')
    parser.add_argument('--use-obj-gcn', type=int, choices=[0,1])
    parser.add_argument('--use-context-gcn', type=int, choices=[0,1])
    parser.add_argument('--adj-type', type=str, default='uniform', choices=['prior', 'uniform', 'learned'],
                        help="Type of adjacency matrix for GCN. Choose among 'prior', 'uniform', or 'learned'.")
    parser.add_argument('--use-flow', type=int, default=0, help='whether to add a branch for optical flow')
    parser.add_argument('--two-layer-context', type=int, default=0, help='Whether to use 2 layers in the context classification head.')
    parser.add_argument('--use-wv-weights', type=int, default=0,
                        help='Whether to use word vectors as classifier weights. Valid only when "two-layer-context is 1".')
    parser.add_argument('--freeze-wv-weights', type=int, default=1,
                        help='Whether to train the last layer of the context classifier. Valid only when "use-wv-weights" is 1.')
    parser.add_argument('--depth', default=50, type=int,
                        metavar='D', help='depth of the backbone')
    parser.add_argument('--loss', default='ce', choices=['ce', 'ce+ce'],
                        help='Whether to include obj loss.')
    parser.add_argument('--dataset', metavar='D',
                        default='vlog',
                        help='dataset name')
    parser.add_argument('--train-set', metavar='T',
                        default='train+val',
                        help='Training set: could be train or train+val when you want to get the final accuracy')
    parser.add_argument('--root', metavar='D',
                        default='./data/vlog',
                        help='location of the dataset directory')
    parser.add_argument('--feats-obj-dir', type=str, default='/vision2/u/cy3/data/EPIC/fm/objects/')
    parser.add_argument('--feats-ctxt-dir', type=str, default='/vision2/u/cy3/data/EPIC/fm/context/')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='B', help='batch size')
    parser.add_argument('-t', '--t', default=2, type=int,
                        metavar='T', help='number of timesteps extracted from a video')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--nb-classes', type=int, default=125, help='number of verb classes. 125 for EPIC.')
    parser.add_argument('--nb-obj-classes', type=int, default=353, help='number of obj classes. 353 for EPIC.')
    parser.add_argument('--nb-crops', type=int, default=10, metavar='N',
                        help='number of crops while testing')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--print-freq', default=1, type=int, metavar='P',
                        help='frequence of printing in the log')
    parser.add_argument('--resume',
                        default='/tmp/my_resume',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--save-token', default='', type=str, 
                        help='special (hopefully uniq) token as notes for different experiment settings')
    parser.add_argument('--ckpt-name', default='model_best.pth', type=str, help='checkpoint filename (w/o dir path)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluation mode')
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='cuda mode')
    parser.add_argument('--add-background', dest='add_background', action='store_true',
                        help='add the background as one object')
    parser.add_argument('--mask-dir', type=str, default='masks/preds_100x_100_50', help='dir of pre-extracted mask files')
    parser.add_argument('--mask-confidence', type=float, default=0.50, metavar='LR',
                        help='mininum confidence for the masks')
    parser.add_argument('--pooling', metavar='POOL',
                        default='rnn',
                        help='final pooling methods: avg or rnn')
    parser.add_argument('--heads', metavar='H',
                        default='object',
                        # default='object+context',
                        help='what are the heads of the model: object or context or object+context')
    parser.add_argument('--blocks', metavar='LB',
                        default='2D_2D_2D_2.5D',
                        help='Nature of the 4 residual blocks: B1_B2_B3_B4 where Bi can be 2D, 3D or 2.5D')
    parser.add_argument('--object-head', metavar='BH',
                        default='2D',
                        help='Nature of the residual block of the object head: Bi where can be 2D, 3D or 2.5D')
    # GCN model params
    parser.add_argument('--D-obj', type=int, default=2048, help='obj feats (excluding embedded mask)')
    parser.add_argument('--D-verb', type=int, default=2048),
    parser.add_argument('--D-obj-embed', type=int, default=32),
    parser.add_argument('--D-verb-embed', type=int, default=32),
    parser.add_argument('--n-layers', type=int, default=2),
    parser.add_argument('--n-top-objs', type=int, default=10),

    # Args
    args, _ = parser.parse_known_args()

    # Dict
    options = vars(args)
    options['ckpt_file'] = os.path.join(
        options['resume'],
        '{}_gpu{}_task{}.exp'.format(options['machine'], options['gpu_id'], options['task_id']))

    print('\nOptions:')
    for key in options:
      print('{}: {}'.format(key, options[key]))
    print('\n')

    inference.main(options)
