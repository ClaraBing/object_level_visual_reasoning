import pickle
from loader.vlog import VLOG
from loader.epic import EPIC
from loader.feats_epic import FeatsEPIC
import torch
from loader.videodataset import my_collate
from torch.nn import Module
import torch.nn as nn
import torch
from utils.meter import *
import shutil
import math
import cv2
import pdb


def load_pickle(file):
    with open(file, mode='rb') as f:
        df = pickle.load(f, encoding='latin1')
    return df


def get_datasets_and_dataloaders(options, device='cpu'):
    # Choice of Dataset
    if options['dataset'] == 'vlog':
        VideoDataset = VLOG
        if options['train_set'] == 'train':
            train_set_name = 'train'
            val_set_name = 'val'
            nb_crops = 1
        elif options['train_set'] == 'train+val':
            train_set_name = 'train+val'
            val_set_name = 'test'
            nb_crops = options['nb_crops']
        else:
            raise NameError
        options['nb_classes'] = 30
    elif options['dataset'] == 'epic':
        VideoDataset = EPIC
        train_set_name = 'train'
        val_set_name = 'val'
        nb_crops = options['nb_crops']
        options['nb_classes'] = 125
    elif options['dataset'] == 'feats_epic':
        VideoDataset = FeatsEPIC
        train_set_name = 'train'
        val_set_name = 'val'
        nb_crops = options['nb_crops']
        options['nb_classes'] = 125
    else:
        raise NameError

    # Dataset
    train_dataset = VideoDataset(options,
                                 dataset=train_set_name,
                                 nb_crops=1,
                                 usual_transform=True,
                                 add_background=options['add_background'])
    val_dataset = VideoDataset(options,
                               dataset=val_set_name,
                               nb_crops=nb_crops,
                               usual_transform=True,
                               add_background=options['add_background'])

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=options['batch_size'],
                                               shuffle=True,
                                               num_workers=options['workers'],
                                               pin_memory=device!='cpu',
                                               collate_fn=my_collate)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=options['batch_size'],
                                             shuffle=False,
                                             num_workers=options['workers'],
                                             pin_memory=device!='cpu',
                                             collate_fn=my_collate)

    return train_dataset, val_dataset, train_loader, val_loader


def get_loss_and_metric(options):
    if options['dataset'] == 'vlog':
        # Metric
        metric = AveragePrecisionMeter
        # Loss
        loss = CriterionLinearCombination(['bce', 'ce'], [15.0, 1.0])
    elif options['dataset'] == 'epic' or options['dataset'] == 'feats_epic':
       # TODO: use better metric/loss
        metric = AveragePrecisionMeter
        if options['loss'] == 'ce+ce':
          loss = CriterionLinearCombination(['ce', 'ce'], [5.0, 1.0])
        else:
          loss = CriterionLinearCombination(['ce'], [1.0])
    else:
        raise NameError

    return loss, metric


class CriterionLinearCombination(Module):
    def __init__(self, list_criterion_names, list_weights):
        super(CriterionLinearCombination, self).__init__()
        assert len(list_criterion_names) == len(list_weights)

        self.list_criterion, self.list_weights = [], []
        for i, criterion_name in enumerate(list_criterion_names):
            # Criterion
            if criterion_name == 'bce':
                self.list_criterion.append(nn.BCEWithLogitsLoss())
            elif criterion_name == 'ce':
                self.list_criterion.append(nn.CrossEntropyLoss())
            elif criterion_name == 'l1':
                self.list_criterion.append(nn.L1Loss())
            else:
                raise Exception
            # Weight
            self.list_weights.append(list_weights[i])

    def forward(self, list_input, list_target, device='cpu'):
        assert len(list_input) == len(list_target)
        loss = 0.0
        for i in range(len(self.list_criterion)):
            # Cast depending of the criterion
            criterion_i, weight_i = self.list_criterion[i], self.list_weights[i]
            target_i, input_i = list_target[i], list_input[i]
            if input_i is not None:
                if isinstance(criterion_i, nn.CrossEntropyLoss):
                    if target_i.dim() == 2:
                      target_i = target_i.nonzero()[:, 1]
                    target_i = target_i.type(torch.LongTensor)
                elif isinstance(criterion_i, nn.BCEWithLogitsLoss):
                    target_i = target_i.type(torch.FloatTensor)

                target_i = target_i.to(device)

                # Compute the loss and add
                input_i = input_i.view(-1, input_i.size(-1))
                loss_i = weight_i * criterion_i(input_i, target_i)
                loss = loss + loss_i

        return loss


def load_from_dir(model, optimizer, options):
    ''' load from resume found in the dir'''
    epoch = 0
    if options['resume']:
        if os.path.isdir(options['resume']):
            # ckpt_resume = os.path.join(options['resume'], options['ckpt_name'])
            ckpt_file = options['ckpt_file']
            if not os.path.exists(ckpt_file):
              ckpt_resume = 'model_best.pth'
            else:
              with open(ckpt_file, 'r') as fin:
                ckpt_resume = fin.readline().strip()

            ckpt_resume = os.path.join(options['resume'], ckpt_resume)
            if not os.path.isfile(ckpt_resume):
              untarred = ckpt_resume.replace('.tar', '')
              if os.path.isfile(untarred):
                ckpt_resume = untarred
            if os.path.isfile(ckpt_resume):
                options['ckpt_resume'] = ckpt_resume
                
                print("\n=> loading checkpoint '{}'".format(ckpt_resume))
                checkpoint = torch.load(ckpt_resume, map_location=lambda storage, loc: storage)
                epoch = checkpoint['epoch']
                # Remove the fc_classifier's
                updated_params = {}
                model_dict = model.state_dict()
                for k, v in checkpoint['state_dict'].items():
                    # Train classifier fom scratch
                    if "fc_classifier" in k and not options['evaluate']:
                        pass
                    # Look if the size if the same
                    if k in list(model_dict.keys()):
                        v_new_size, v_old_size = v.size(), model_dict[k].size()
                        if v_old_size == v_new_size:
                            updated_params[k] = v

                # Load
                new_params = model.state_dict()
                new_params.update(updated_params)
                model.load_state_dict(new_params)

                # Optim
                updated_params = {}
                new_params = optimizer.state_dict()
                for k, v in checkpoint['state_dict'].items():
                    if k not in list(new_params.keys()):
                        updated_params[k] = v

                new_params.update(updated_params)
                optimizer.load_state_dict(new_params)

                # Epoch
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(ckpt_resume, checkpoint['epoch']))
            else:
                print("\n=> no checkpoint found at '{}'".format(options['resume']))
        else:
            os.makedirs(options['resume'])

    return model, optimizer, epoch


def print_number(number):
    """ print a ' every 3 number starting from the left (e.g 23999 -> 23'999)"""
    len_3 = round(len(str(number)) / 3.)

    j = 0
    number = str(number)
    for i in range(1, len_3 + 1):
        k = i * 3 + j
        number = number[:-k] + '\'' + number[-k:]
        j += 1

    # remove ' if it is at the begining
    if number[0] == '\'':
        return number[1:]
    else:
        return number


def write_to_log(log_path, dataset, resume, epoch, metrics, metrics_per_class, err_str=''):
    # Global metric
    file_full_name = os.path.join(resume, log_path)
    with open(file_full_name, 'a+') as f:
        if err_str:
            f.write(err_str+'\n')
        elif metrics is not None:
            f.write('Epoch=%03d, Loss=%.4f, Metric=%.4f\n' % (epoch, metrics[0], metrics[1]))
        else:
            f.write('Unknown exception (may be pdb.)')

    # Per class metric
    if metrics_per_class is not None:
        file_full_name = os.path.join(resume, log_path + '_per_class_metrics_log')
        np.savetxt(file_full_name, metrics_per_class.numpy(), fmt='%10.4f', delimiter=',')


def transform_input(x, dim, T=8):
    diff = len(x.size()) - dim

    if diff > 0:
        B, C, T, W, H = x.size()
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, C, W, H)
    elif diff < 0:
        _, C, W, H = x.size()
        x = x.view(-1, T, C, W, H)
        x = x.transpose(1, 2)

    return x


def count_nb_params(enum_params):
    nb_params = 0
    for parameter in enum_params:
        nb_param_w = 1
        for s in parameter.size():
            nb_param_w *= s
        nb_params += nb_param_w
    return nb_params


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    full_filename = os.path.join(save_dir, filename)
    torch.save(state, full_filename)
    if is_best:
        full_filename_best = os.path.join(save_dir, filename.replace('.pth', '_best.pth'))
        shutil.copyfile(full_filename, full_filename_best)


def decode_videoId(bytes_id):
    try:
        idx_1st_0 = (bytes_id == 0).argmax(axis=0)  # find zero padding
    except:
        idx_1st_0 = (bytes_id == 0).argmax()  # find zero padding

    str_video_id = bytes_id[:idx_1st_0].tobytes().decode("utf-8")  # remove zero padding
    return str_video_id


def store_preds(preds, id, list_correct_preds, obj_id, dataset='vlog'):
    # sigmoid or softmax
    # if dataset == 'vlog':
    #     f = sigmoid
    # else:
    #     raise NameError

    # cpu - np
    id_np = id.cpu().numpy()
    preds_cpu = np.round(preds.cpu().numpy().astype(np.float16), 2)
    obj_id_cpu = obj_id.cpu().numpy().astype(np.float16)

    # Lop
    dict_good, dict_failure, dict_obj = {}, {}, {}
    for i, p in enumerate(preds_cpu):
        # catch id
        id_i = decode_videoId(id_np[i])

        # obj id
        dict_obj[id_i] = np.round(obj_id_cpu[i].sum(0).sum(0), 2)

        # good or failure
        if i in list_correct_preds:
            dict_good[id_i] = p
        else:
            dict_failure[id_i] = p

    return dict_good, dict_failure, dict_obj


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

