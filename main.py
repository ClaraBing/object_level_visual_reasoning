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
    parser.add_argument('--device', default='cuda:0',
                        help="A string specifying the device on which models and data reside. 'cpu' or 'cuda:x'")
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        default='orn_two_heads',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: orn_two_heads')
    parser.add_argument('--use-gcn', type=int, choices=[0,1])
    parser.add_argument('--adj-type', type=str, default='uniform', choices=['prior', 'uniform', 'learned'],
                        help="Type of adjacency matrix for GCN. Choose among 'prior', 'uniform', or 'learned'.")
    parser.add_argument('--depth', default=50, type=int,
                        metavar='D', help='depth of the backbone')
    parser.add_argument('--dataset', metavar='D',
                        default='vlog',
                        help='dataset name')
    parser.add_argument('--train-set', metavar='T',
                        default='train+val',
                        help='Training set: could be train or train+val when you want to get the final accuracy')
    parser.add_argument('--root', metavar='D',
                        default='./data/vlog',
                        help='location of the dataset directory')
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
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--print-freq', default=1, type=int, metavar='P',
                        help='frequence of printing in the log')
    parser.add_argument('--resume',
                        default='/tmp/my_resume',
                        # default='./resume/vlog',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluation mode')
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='cuda mode')
    parser.add_argument('--add-background', dest='add_background', action='store_true',
                        help='add the background as one object')
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
                        help='Nature of teh residual block of the object head: Bi where can be 2D, 3D or 2.5D')
    # GCN model params
    parser.add_argument('--D-obj', type=int, default=2048)
    parser.add_argument('--D-verb', type=int, default=2048),
    parser.add_argument('--D-obj-embed', type=int, default=512),
    parser.add_argument('--D-verb-embed', type=int, default=512),
    parser.add_argument('--n-layers', type=int, default=2),
    parser.add_argument('--n-top-objs', type=int, default=10),

    # Args
    args, _ = parser.parse_known_args()

    # Dict
    options = vars(args)

    inference.main(options)
