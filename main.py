from __future__ import print_function
import argparse
import torch
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
from optimizer import PruneAdam
from model import MLPnet, GNNnet, CNN1Dnet, CNN2Dnet, SSGCNnet
from utils import regularized_nll_loss, admm_loss, \
    initialize_Z_and_U, update_X, update_Z, update_Z_l1, update_U, \
    print_convergence, print_prune, apply_prune, apply_l1_prune
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import time
import method

def train(args, model, device, train_loader, test_loader, optimizer, dataset, model_style):
    for epoch in range(args.num_pre_epochs):
        #print('Pre epoch: {}'.format(epoch + 1))
        model.train()
        #for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = regularized_nll_loss(args, model, output, target)
            loss.backward()
            optimizer.step()
        test(args, model, device, test_loader, dataset, model_style)

    Z, U = initialize_Z_and_U(model)
    for epoch in range(args.num_epochs):
        model.train()
        #print('Epoch: {}'.format(epoch + 1))
        #for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = admm_loss(args, device, model, Z, U, output, target)
            loss.backward()
            optimizer.step()
        X = update_X(model)
        Z = update_Z_l1(X, U, args) if args.l1 else update_Z(X, U, args)
        U = update_U(U, X, Z)
        print_convergence(model, X, Z)
        test(args, model, device, test_loader, dataset, model_style)


def test(args, model, device, test_loader, dataset, model_style):
    loss = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    val_TP = 1.0
    val_TN = 1.0
    val_FN = 1.0
    val_FP = 1.0
    predict_total = []
    label_total = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, predict = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            batch_loss = loss(predict, target)
            predict_val = np.argmax(predict.cpu().data.numpy(), axis=1)
            predict_total = np.append(predict_total, predict_val)
            label_val = target.numpy()
            label_total = np.append(label_total, label_val)


    test_loss /= len(test_loader.dataset)


    val_TP = ((predict_total == 1) & (label_total == 1)).sum().item()
    val_TN = ((predict_total == 0) & (label_total == 0)).sum().item()
    val_FN = ((predict_total == 0) & (label_total == 1)).sum().item()
    val_FP = ((predict_total == 1) & (label_total == 0)).sum().item()

    val_spe = val_TN / (val_FP + val_TN + 0.000001)
    val_rec = val_TP / (val_TP + val_FN + 0.000001)
    val_pre = val_TP / (val_TP + val_FP + 0.000001)
    #test_acc = (val_TP + val_TN) / (val_FP + val_TN + val_TP + val_FN + 0.00001)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Spe:{:.6f}, Rec:{:.6f}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), val_spe, val_rec))

    with open('save/{}_{}_epoch50.txt'.format(dataset, model_style), 'a+') as f:
        f.write(str(format(test_loss, '.5f')) + '\t' + str(format(100. * correct / len(test_loader.dataset), '.5f')) + '\t' + str(format(val_spe, '.5f')) + '\t' + str(format(val_rec, '.5f')) + '\t' + str(format(val_pre, '.5f')) + '\n')

    acc = correct / len(test_loader.dataset)

    return acc



def retrain(args, model, mask, device, train_loader, test_loader, optimizer, dataset, model_style):
    best_acc = 0.0
    for epoch in range(args.num_re_epochs):
        #print('Re epoch: {}'.format(epoch + 1))
        model.train()
        #for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.prune_step(mask)

        acc = test(args, model, device, test_loader, dataset, model_style)
        if (acc > best_acc and epoch > 10):
            best_acc = acc
    with open('result/{}_{}_acc.txt'.format(model_style, dataset), 'a+') as f:
        f.write(str(format(best_acc, '.5f')) + '\n')

def load_data(dataset,train_label):
    path = "../submit/data/"
    
    path_data_t = path + dataset + '.mat'
    path_label = path + train_label + '.mat'

    data_t = sio.loadmat(path_data_t)[dataset]
    data_label = sio.loadmat(path_label)[train_label]
    data_label = data_label.flatten()

    return data_t, data_label


def convert_to_graph(data_f, func):
    samples = data_f.shape[0]

    time1 = time.time()
    adj_f = func(data_f)
    time2 = time.time()

    adj_f_time = (time2 - time1) / samples

    print("adj_F convert time : %3.8f" % (adj_f_time))
    print("Data load Finished!")
    return adj_f, adj_f_time

def main(dataset,train_label,model_style):
    print("DataSet:{}".format(dataset))
    train_data, train_label = load_data(dataset,train_label)
    # methods = {'WF1': method.overlook_wf1, 'OG': method.overlook, 'WOG': method.overlookg, 'WS': method.overlook_WS, 'V': method.LPvisibility_v, 'LV': method.LPvisibility_lv, 'H': method.LPhorizontal_h, 'LH': method.LPhorizontal_lh}
    methods = {'WNFG2': method.overlook_wnfg2}
    # method flop
    for method_name in methods:
        print("Method:{}".format(method_name))
        adj_f, adj_f_time = convert_to_graph(train_data, methods[method_name])

        skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
        i = 0

        for train_index, val_index in skf.split(adj_f, train_label):
            i += 1
            print("Fold:{}".format(i))
            train_ch011, train_label1 = adj_f[train_index], train_label[train_index]
            val_ch011, val_label1 = adj_f[val_index], train_label[val_index]

            train_ch01 = np.array(train_ch011, dtype=float)
            val_ch01 = np.array(val_ch011, dtype=float)

            train_ch01 = torch.FloatTensor(train_ch01)
            #train_ch01 = torch.unsqueeze(train_ch01, dim=1).float() #1DCNN and SGCN need delete this code
            val_ch01 = torch.FloatTensor(val_ch01)
            #val_ch01 = torch.unsqueeze(val_ch01, dim=1).float() #1DCNN and SGCN need delete this code

            train_label11 = np.array(train_label1, dtype=int)
            val_label11 = np.array(val_label1, dtype=int)

            train_label11 = torch.LongTensor(train_label11)
            #train_label11 = torch.squeeze(train_label11, dim=1).long()
            # train_label = torch.squeeze(train_label, dim=0).long()
            val_label11 = torch.LongTensor(val_label11)
            #val_label11 = torch.squeeze(val_label11, dim=1).long()
            # val_label = torch.squeeze(val_label, dim=0).long()

            # print(train_ch01.shape)
            # print(train_label.shape)
            train_set = TensorDataset(train_ch01, train_label11)
            val_set = TensorDataset(val_ch01, val_label11)

            # torch.manual_seed(seed)
            train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
            test_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)


            #train_loader, test_loader = data_loader_generator(train_data, train_label)

            percent_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
            #print("percent:{}".format(percent))
            for percent in percent_list:
                # Training settings
                parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
                parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                                    help='input batch size for training (default: 64)')
                parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                                    help='input batch size for testing (default: 1000)')
                parser.add_argument('--percent', type=list, default=[percent, percent, percent, percent, percent, percent, percent, percent, percent, percent, percent],
                                    metavar='P', help='pruning percentage (default: 0.8)')
                parser.add_argument('--alpha', type=float, default=5e-4, metavar='L',
                                    help='l2 norm weight (default: 5e-4)')
                parser.add_argument('--rho', type=float, default=1e-2, metavar='R',
                                    help='cardinality weight (default: 1e-2)')
                parser.add_argument('--l1', default=False, action='store_true',
                                    help='prune weights with l1 regularization instead of cardinality')
                parser.add_argument('--l2', default=False, action='store_true',
                                    help='apply l2 regularization')
                parser.add_argument('--num_pre_epochs', type=int, default=30, metavar='P',
                                    help='number of epochs to pretrain (default: 3)')
                parser.add_argument('--num_epochs', type=int, default=30, metavar='N',
                                    help='number of epochs to train (default: 10)')
                parser.add_argument('--num_re_epochs', type=int, default=30, metavar='R',
                                    help='number of epochs to retrain (default: 3)')
                parser.add_argument('--lr', type=float, default=0.0015, metavar='LR',
                                    help='learning rate (default: 1e-2)')
                parser.add_argument('--adam_epsilon', type=float, default=1e-8, metavar='E',
                                    help='adam epsilon (default: 1e-8)')
                #parser.add_argument('--no-cuda', action='store_true', default=False,
                #                    help='disables CUDA training')
                parser.add_argument('--seed', type=int, default=0, metavar='S',
                                    help='random seed (default: 1)')
                parser.add_argument('--save-model', action='store_true', default=False,
                                    help='For Saving the current Model')
                parser.add_argument('--node_number', type=int, default=256, metavar='N',
                                    help='the number of input points (default: 100)')
                parser.add_argument('--k_hop', type=int, default=1, metavar='N',
                                    help='the number of k_hop (default: 2)')
                parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                                    help='input batch size for training (default: 64)')
                args = parser.parse_args()
            
                #use_cuda = not args.no_cuda and torch.cuda.is_available()
            
                torch.manual_seed(args.seed)
            
                #device = torch.device("cuda" if use_cuda else "cpu")
                device = torch.device("cpu")
            
                #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
            
                #model = sleep1DCNN().to(device)
                if model_style == 'MLP':
                    model = MLPnet(args.node_number, args.batch_size, args.k_hop).to(device)
                elif model_style == 'GNN':
                    model = GNNnet(args.node_number, args.batch_size, args.k_hop).to(device)
                elif model_style == 'CNN1D':
                    model = CNN1Dnet(args.node_number, args.batch_size, args.k_hop).to(device)
                else: 
                    model = SSGCNnet(args.node_number, args.batch_size, args.k_hop).to(device)
                optimizer = PruneAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
            
                train(args, model, device, train_loader, test_loader, optimizer, dataset, model_style)
                mask = apply_l1_prune(model, device, args) if args.l1 else apply_prune(model, device, args)
                print_prune(model)
                test(args, model, device, test_loader, dataset, model_style)
                retrain(args, model, mask, device, train_loader, test_loader, optimizer, dataset, model_style)


if __name__ == "__main__":  
    main('AET', 'train_label_Bonn', 'SSGCN')
    main('BET', 'train_label_Bonn', 'SSGCN')
    main('CET', 'train_label_Bonn', 'SSGCN')
    main('DET', 'train_label_Bonn', 'SSGCN')

    main('AEF', 'train_label_Bonn', 'SSGCN')
    main('BEF', 'train_label_Bonn', 'SSGCN')
    main('CEF', 'train_label_Bonn', 'SSGCN')
    main('DEF', 'train_label_Bonn', 'SSGCN')
    
