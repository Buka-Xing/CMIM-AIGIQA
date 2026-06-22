import torch.nn as nn
import numpy as np
import random
import scipy.stats
from utils_self import _preprocess2
from utils_self import *
import os
import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from model import CMIM
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='MI-AIGCIQA')
    parser.add_argument('-f', default='', type=str)

    # Dropouts
    parser.add_argument('--dropout_a', type=float, default=0.1,
                        help='dropout of acoustic LSTM out layer')
    parser.add_argument('--dropout_v', type=float, default=0.1,
                        help='dropout of visual LSTM out layer')
    parser.add_argument('--dropout_prj', type=float, default=0.1,
                        help='dropout of projection layer')

    # Architecture
    parser.add_argument('--multiseed', action='store_true', help='training using multiple seed')
    parser.add_argument('--contrast', action='store_true', help='using contrast learning')
    parser.add_argument('--n_layer', type=int, default=1,
                        help='number of layers in LSTM encoders (default: 1)')
    parser.add_argument('--cpc_layers', type=int, default=1,
                        help='number of layers in CPC NCE estimator (default: 1)')
    parser.add_argument('--d_vh', type=int, default=1024,
                        help='hidden size in visual rnn')
    parser.add_argument('--d_vout', type=int, default=1024,
                        help='output size in visual rnn')
    parser.add_argument('--bidirectional', action='store_true', help='Whether to use bidirectional rnn')
    parser.add_argument('--d_prjh', type=int, default=128,
                        help='hidden size in projection network')
    parser.add_argument('--pretrain_emb', type=int, default=768,
                        help='dimension of pretrained model output')

    # Activations
    parser.add_argument('--mmilb_mid_activation', type=str, default='ReLU',
                        help='Activation layer type in the middle of all MMILB modules')
    parser.add_argument('--mmilb_last_activation', type=str, default='Tanh',
                        help='Activation layer type at the end of all MMILB modules')
    parser.add_argument('--cpc_activation', type=str, default='Tanh',
                        help='Activation layer type in all CPC modules')

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=5, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--lr_main', type=float, default=1e-3,
                        help='initial learning rate for main model parameters (default: 1e-3)')
    parser.add_argument('--lr_bert', type=float, default=5e-5,
                        help='initial learning rate for bert parameters (default: 5e-5)')
    parser.add_argument('--lr_mmilb', type=float, default=1e-3,
                        help='initial learning rate for mmilb parameters (default: 1e-3)')
    parser.add_argument('--alpha', type=float, default=0.1, help='weight for CPC NCE estimation item (default: 0.1)')
    parser.add_argument('--beta', type=float, default=0.1, help='weight for lld item (default: 0.1)')

    parser.add_argument('--weight_decay_main', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_bert', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_club', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')

    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of epochs (default: 40)')
    parser.add_argument('--when', type=int, default=10,
                        help='when to decay learning rate (default: 20)')
    parser.add_argument('--patience', type=int, default=5,
                        help='when to stop training if best never change')
    parser.add_argument('--update_batch', type=int, default=1,
                        help='update batch interval')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of result logging (default: 100)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    args = parser.parse_args()
    return args

device = 'cuda:2'
args = get_args()

def train(model, optimizer, criterion, stage=1):
    running_loss = 0
    model.eval()
    left_batch = args.update_batch

    mem_pos_tv = []
    mem_neg_tv = []
    mem_pos_ta = []
    mem_neg_ta = []

    loader = train_loaders[0]

    avg_loss = 0
    loop = tqdm.tqdm(loader, desc='Epoch:{}'.format(epoch))
    i_batch = 0
    mem_size = 1

    for sample_batched in loop:
        i_batch += 1
        img, prompt, gmos = sample_batched['img'], sample_batched['prompt'],sample_batched['mos']
        img = img.squeeze(1).to(device)
        gmos = gmos.to(device)

        if stage == 0:
            gmos = None

        model.zero_grad()
        optimizer.zero_grad()

        lld, nce, preds, H = model(prompt, img)
        if stage == 1:
            y_loss = criterion(preds.flatten().float(), gmos.float())

            if args.contrast:
                loss = y_loss + args.alpha * nce - args.beta * lld  # nce对应原文中得CPC loss, lld对应原文中的BA loss
            else:
                loss = y_loss
            if i_batch > mem_size:
                loss -= args.beta * H
            loss.backward()

        elif stage == 0:
            # maximize likelihood equals minimize neg-likelihood
            loss = - lld
            loss.backward()
        else:
            raise ValueError('stage index can either be 0 or 1')

        left_batch -= 1
        if left_batch == 0:
            left_batch = args.update_batch
            optimizer.step()

        running_loss += loss.data.item()
        avg_loss = running_loss / i_batch
        Epoch_info = 'Stage: {} Epoch:{} Loss:{:.4f} lr:{:e}'.format(stage, epoch, avg_loss, optimizer.state_dict()['param_groups'][0]['lr'])
        loop.set_description(Epoch_info)

    return avg_loss

def eval(loader, phase, dataset):
    model.eval()

    q_mos = []
    q_hat = []

    for sample_batched in tqdm.tqdm(loader, desc='{}:{}'.format(dataset, phase)):
        img, prompt, gmos = sample_batched['img'], sample_batched['prompt'], sample_batched['mos']
        img = img.squeeze(1).to(device)
        gmos = gmos.to(device)
        q_mos = q_mos + gmos.cpu().tolist()

        with torch.no_grad():
            _, _, preds, _ = model(prompt, img)
            preds = preds[0].flatten()

        quality_preds = preds.squeeze().unsqueeze(dim=0)
        q_hat = q_hat + quality_preds.cpu().tolist()

    # Compute SRCC and PLCC
    srcc = abs(scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0])
    plcc = abs(scipy.stats.pearsonr(x=q_mos, y=q_hat)[0])

    print_text = dataset + ':' + phase + ': ' +  'srcc:{:.4f}  plcc{:.4f}\n'.format(srcc,plcc)
    log_and_print(base_logger,print_text)

    return  (srcc+plcc)/2, srcc, plcc

def set_optimizer(model, args):
    mmilb_param = []
    main_param = []
    bert_param = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            if 'visual_encoder' in name:
                bert_param.append(p)
            elif 'mi' in name:
                mmilb_param.append(p)
            else:
                main_param.append(p)

    optimizer_mmilb = getattr(torch.optim, args.optim)(mmilb_param, lr=args.lr_mmilb, weight_decay=args.weight_decay_club)

    optimizer_main_group = [
        {'params': bert_param, 'weight_decay': args.weight_decay_bert, 'lr': args.lr_bert},
        {'params': main_param, 'weight_decay': args.weight_decay_main, 'lr': args.lr_main}
    ]

    optimizer_main = getattr(torch.optim, args.optim)(optimizer_main_group)

    scheduler_mmilb = torch.optim.lr_scheduler.StepLR(optimizer_mmilb, step_size=step_size, gamma=0.5) # self
    scheduler_main  = torch.optim.lr_scheduler.StepLR(optimizer_main, step_size=step_size, gamma=0.5) # self
    return optimizer_mmilb, optimizer_main, scheduler_mmilb, scheduler_main

if __name__ == '__main__':
    Train_dataset  = 'PKU-AIGIQA'         # ['AGIQA-3k', 'AIGCIQA2023', 'PKU-AIGIQA']
    Test_dataset   = 'AGIQA-3k'           # ['AGIQA-3k', 'AIGCIQA2023', 'PKU-AIGIQA']
    mos_type       = 'quality'            # ['quality', 'consis'] for AGIQA-3k, ['quality', 'consis', 'authn'] for AIGCIQA2023 and PKU-AIGIQA
    main_lr        = 1e-5                 # the lr for training the stage 1
    mmilb_lr       = 5e-6                 # the lr for training the stage 0 (only the Forward MIM)
    num_epoch      = 10                   # number of epochs for every train-test split valiadation
    step_size      = 2                    # Number of steps of lr decay
    train_batch    = 5                    # the batch size for training, setting 5 for reproducibility of GPU with limited memory
    val_batch      = 1                    # the batch size for validation
    num_workers    = 5
    alpha          = 0.1                  # the hyperparam for forward MIM training
    beta           = 0.1                  # the hyperparam for backward MIM training
    checkpoint_dir = './weights_final/MI-AIGIQA_BLIP_Cross_Train%s_Test%s_batch%s_main-lr%s_mmilb-lr%s_%s' \
                     % (Train_dataset, Test_dataset, train_batch, main_lr, mmilb_lr, mos_type)
    os.makedirs(checkpoint_dir, exist_ok=True)

    args.contrast   = True          # whether pretraining stage0, default=True. Or directly training stage1
    args.d_prjh     = 1024          # the dimension of the image features processed by ImageReward
    args.d_tin      = 768           # the dimension of the test features processed by ImageReward
    args.alpha      = alpha
    args.beta       = beta
    args.lr_main    = main_lr
    args.lr_bert    = main_lr
    args.lr_mmilb   = mmilb_lr
    args.num_epochs = num_epoch
    args.batch_size = train_batch

    seed = 20200626

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    criterion = nn.MSELoss(reduction='sum').to(device)
    preprocess2 = _preprocess2()    # process RGB image to tensor

    best_result_list = []
    base_logger = get_logger(os.path.join(checkpoint_dir,'train_test.log'), 'log')
    log_and_print(base_logger,
                  'main_lr:{:e}, mmilb_lr{:3}, num_epoch:{:d}, train_batch:{:d}, val_batch:{:d}'.format(main_lr,mmilb_lr,num_epoch,train_batch,val_batch))

    model = CMIM(args, device=device).to(device)
    optimizer_mmilb, optimizer_main, scheduler_mmilb, scheduler_main = set_optimizer(model, args)

    train_loss = []
    start_epoch = 0

    best_result = { 'quality': 0.0 ,'srcc': 0.0, 'plcc': 0.0}
    best_epoch  = { 'quality': 0}

    aigc_train_csv = './Database/%s/full.csv' % Train_dataset
    aigc_val_csv   = './Database/%s/full.csv' % Test_dataset

    if   Train_dataset == 'AGIQA-1k':
        aigc_train_set = '../AIGIQA/IQA_models/IPCE/data/%s/Image' % Train_dataset
    elif Train_dataset == 'AGIQA-3k':
        aigc_train_set = '../AIGIQA/IQA_models/IPCE/data/%s/Image' % Train_dataset
    elif Train_dataset == 'AIGCIQA2023':
        aigc_train_set = '../AIGIQA/IQA_models/IPCE/data/%s/Image' % Train_dataset
    elif Train_dataset == 'PKU-AIGIQA':
        aigc_train_set = '../AIGIQA/IQA_models/IPCE/data/%s/Image' % Train_dataset

    if   Test_dataset == 'AGIQA-1k':
        aigc_test_set = '../AIGIQA/IQA_models/IPCE/data/%s/Image' % Test_dataset
    elif Test_dataset == 'AGIQA-3k':
        aigc_test_set = '../AIGIQA/IQA_models/IPCE/data/%s/Image' % Test_dataset
    elif Test_dataset == 'AIGCIQA2023':
        aigc_test_set = '../AIGIQA/IQA_models/IPCE/data/%s/Image' % Test_dataset
    elif Test_dataset == 'PKU-AIGIQA':
        aigc_test_set = '../AIGIQA/IQA_models/IPCE/data/%s/Image' % Test_dataset

    aigc_train_loader = set_dataset_pyiqa(Train_dataset, aigc_train_csv, train_batch, aigc_train_set, num_workers, preprocess2, mos_type, False)
    aigc_val_loader = set_dataset_pyiqa(Test_dataset, aigc_val_csv, val_batch, aigc_test_set, num_workers, preprocess2, mos_type, True)  # 15和8代表的是batch数量

    train_loaders = [aigc_train_loader]

    result_pkl = {}
    for epoch in range(0, num_epoch):
        if args.contrast:
            avg_loss_stage0 = train(model, optimizer_mmilb, criterion, 0)
        scheduler_mmilb.step()

        # minimize all losses left
        avg_loss_stage1 = train(model, optimizer_main, criterion, 1)
        scheduler_main.step()

        # Pick and report current best performance
        avg_score, srcc, plcc = eval(aigc_val_loader, phase='val', dataset='live')
        if avg_score > best_result['quality']:  # avg_score = (srcc + plcc) / 2
            log_and_print(base_logger, '**********New quality best!**********\n')
            best_epoch['quality'] = epoch
            best_result['quality'] = avg_score
            best_result['srcc'] = srcc
            best_result['plcc'] = plcc
            dir = os.path.join(checkpoint_dir)
            os.makedirs(dir, exist_ok=True)

        log_and_print(base_logger, '...............current quality best...............')
        log_and_print(base_logger, 'best quality epoch:{}\n'.format(best_epoch['quality']))
        log_and_print(base_logger,
                      'best quality result:{}, plcc{}, srcc:{}\n'.format(best_result['quality'], best_result['plcc'],
                                                                         best_result['srcc'], ))

    best_result_list.append(best_result)

    log_and_print(base_logger,'all_finished')