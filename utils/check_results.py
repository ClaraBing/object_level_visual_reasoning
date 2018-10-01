import os
from glob import glob
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_lines(lines, labels, fout):
  assert(len(lines) == len(labels))
  fig = plt.figure()
  ax = plt.subplot(111)
  for (line, label) in zip(lines, labels):
    plt.plot(range(len(line)), line, label=label)

  # shrink current axis by 20%
  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
  # put legend to the right of the current axis
  ax.legend(loc='upper center', bbox_to_anchor=(1, 0.5))
  plt.savefig(fout)
  plt.clf()

def get_loss_metric(fckpts, label_removes=[]):
  losses, metrics, labels = [], [], []
  for fckpt in fckpts:
    label = fckpt.split('/')[-2].split('.')[0]
    for each in label_removes:
      label = label.replace(each, '')
    ckpt = torch.load(fckpt)
    loss, metric = ckpt['loss_history'], ckpt['metric_history']
    losses += loss,
    metrics += metric,
    labels += label,
  return losses, metrics, labels

def get_loss_metric_wrapper():
  ckpt_root = '/vision2/u/bingbin/ORN/'

  fckpts = glob(os.path.join(ckpt_root, '*heads*gcnObj*', '*best.pth'))
  # fckpts = [glob(os.path.join(fckpt_dir, '*best.pth')) for fckpt_dir in fckpt_dirs]
  label_removes = ['epic_', 'bt8_lr1e-04_wd1e-05_', '_oEmb128_vEmb128_nLayer2_nTopObjs10']

  losses, metrics, labels = get_loss_metric(fckpts, label_removes)

  plot_lines(losses, labels, fout=os.path.join(ckpt_root, 'result_losses.png'))  
  plot_lines(metrics, labels, fout=os.path.join(ckpt_root, 'result_metrics.png'))


def check_epic_loss_from_log():
  ckpt_root = '/vision2/u/bingbin/ORN/'
  label_removes = ['epic_', 'bt8_lr1e-04_wd1e-05_', '_oEmb128_vEmb128_nLayer2_nTopObjs10']

  losses = []
  labels = []
  for each in glob(os.path.join(ckpt_root, 'epic_headsobject_*')):
    flog = glob(os.path.join(each, 'train*.log'))
    if not flog:
      continue
    flog = flog[0]
    with open(flog, 'r') as fin:
      loss = [float(line.split(' ')[1].split('=')[1][:-1]) for line in fin]
      losses += loss,
      label = flog.split('/')[-2].split('.')[0]
      for each in label_removes:
        label = label.replace(each, '')
      labels += label,
      print('{}: {}'.format(label, loss))
  plot_lines(losses, labels, fout=os.path.join(ckpt_root, 'epic_loss_from_log.png'))

if __name__ == '__main__':
  # get_loss_metric_wrapper()
  check_epic_loss_from_log()
