import datetime
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import copy
import sys
import csv
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from tqdm import tqdm
import transformers

import read_dataset
import log
from model import BentoModel


def main():
    n_splits = 5 # 交差検証の分割数

    parser = argparse.ArgumentParser(description='bento train')

    parser.add_argument('--batch_size', help='batch size',default=32)
    parser.add_argument('--num_epochs', help='num_epochs',default=1)
    parser.add_argument('--in_w', help='in_w',default=640)
    parser.add_argument('--lr', help='lr',default=0.0001) 
    parser.add_argument('--weight_decay', help='weight decay',default=0.01)  
    parser.add_argument('--optim', help='optim',default="Adam", type=str)
    parser.add_argument('--txt_enc', help='text encoder model',default="T5", type=str) # T5 or clip
    parser.add_argument('--seed', help='seed',default= 0)
    parser.add_argument('--num_visual_last_dim', help='num_visual_last_dim',default=512)
    parser.add_argument('--num_text_last_dim', help='num_text_last_dim',default=512)
    parser.add_argument('--aug_degrees', help='aug_degrees',default=45)
    parser.add_argument('--bottle', help='bottle',default=128)
    parser.add_argument('--language', help='language of input text',default="en", type=str) # 0:english, 1:japanese
    args = parser.parse_args()

    print('~~~~~~~~~~ training start ~~~~~~~~~~~~~')
    # ~~~~~~~~~~~~~~~~ param ~~~~~~~~~~~~~~~~~~~
    batch_size = int(args.batch_size)#16
    num_epochs = int(args.num_epochs)#100
    in_w = int(args.in_w)#128
    in_h = in_w
    lr = float(args.lr)#0.001
    weight_decay = float(args.weight_decay)#0.001
    optim_flg = str(args.optim)
    txt_enc_name = str(args.txt_enc)
    seed = int(args.seed)
    scheduler_gamma = 0.99
    bottle = int(args.bottle)
    num_text_last_dim = int(args.num_text_last_dim)
    num_visual_last_dim = int(args.num_visual_last_dim)
    aug_degrees = int(args.aug_degrees)
    language = str(args.language)
    if language == "en":
        language_id = 0
    elif language == "ja":
        language_id = 1

    random_seed = seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
    

    # ~~~~~~~~~~~~~~~~ log folder ~~~~~~~~~~~~~~~~~~~~
    log_path = '/home/taki/bento_VL/log'
    # フォルダ作成
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    param_folder = f'rotate{aug_degrees}_txt_enc-{txt_enc_name}_epoch{num_epochs}_{language}_visal{num_visual_last_dim}_text{num_text_last_dim}_bottle{bottle}_optim{optim_flg}_batch{batch_size}_w,h{in_w}_lr{lr}_wDecay{weight_decay}_seed{seed}_pooler_output'
    path = os.path.join(log_path, param_folder)
    print(param_folder)
    # フォルダ作成
    if not os.path.exists(path):
        os.mkdir(path)


    # 交差検証全体の結果用のlog用リスト定義
    all_corr_list = []
    all_test_loss_list = []

    # ~~~~~~~~~~~~ set data transforms ~~~~~~~~~~~~~~~
    if "clip" in txt_enc_name:
        tokenizer = transformers.CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32") 
    elif "T5" in txt_enc_name:
        tokenizer = transformers.T5TokenizerFast.from_pretrained('t5-small')
    
    # resnet だけのために使わないけどtokenizerは必要
    #tokenizer = transformers.T5TokenizerFast.from_pretrained('t5-small')

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=aug_degrees),
        transforms.Resize((in_w, in_h)),
        transforms.ToTensor()#,
        #transforms.Normalize((0.5,), (0.5,)) #グレースケールのとき
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #これやるとrgb値変になる
    ])

    test_transform = transforms.Compose([
        transforms.Resize((in_w, in_h)),
        transforms.ToTensor()#,
        #transforms.Normalize((0.5,), (0.5,))
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    col = ["彩り", "健康的", "満足感", "ユニークさ", "食べやすさ", "適量", "くずれない"]

    ##################  交差検証用データ作成  #####################
    
    df = pd.read_csv("./score_ml_promax_7_update.csv")

    # 交差検証法(5分割)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_list = []
    test_list = []

    # dfをtrainとtestに分ける
    # train_listは800*7のdfが分割数（5個）だけ入っている
    for train_idx, test_idx in kf.split(df):
        train_data = df.iloc[train_idx].reset_index(drop=True)
        test_data = df.iloc[test_idx].reset_index(drop=True)
        train_list.append(train_data)
        test_list.append(test_data)



    #############################  交差検証法 #######################################{{{
    for idx in range(len(train_list)): # idx：分割されたデータが

        out_path = os.path.join(path, f"closs_valid{idx}")
        if not os.path.exists(out_path):
            os.mkdir(out_path)     

        train_dataset = read_dataset.read_dataset(
            train_list[idx], language_id=language_id, transform=train_transform)
        test_dataset = read_dataset.read_dataset(
            test_list[idx], language_id=language_id, transform=test_transform)

        print("train_dataset:", len(train_dataset))
        print("test_dataset:", len(test_dataset))
        
        validation_size = int(len(test_dataset) / 2)
        test_size       = len(test_dataset) - validation_size

        # valをtestと分ける
        test_dataset, val_dataset = torch.utils.data.random_split(
            test_dataset, [test_size, validation_size], generator=torch.Generator().manual_seed(seed))

        # # drop_lastで余りを入れるか入れないか
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)  
        val_loader = torch.utils.data.DataLoader(
            val_dataset,   batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,  batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)


        #~~~~~~~~~~~~~~~~~~~  gpu setup~~~~~~~~~~~~~~~~~~~~~~~~

        # gpuが使えるならgpuを使用、無理ならcpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  


        #~~~~~~~~~~~~~~~~~~~  net setup~~~~~~~~~~~~~~~~~~~~~~~~
        net = BentoModel(txt_enc_name=txt_enc_name ,num_visual_last_dim=num_visual_last_dim, num_text_last_dim=num_text_last_dim, bottle=bottle)
        net = net.to(device)

        # criterion
        criterion = nn.MSELoss()
        l1_norm   = nn.L1Loss()


        # Observe that all parameters are being optimized
        if optim_flg == "SGD":
            optimizer = optim.SGD(  net.parameters(), 
                                    lr=lr, momentum=0.9, 
                                    weight_decay=weight_decay)
        elif optim_flg == "Adam" :
            optimizer = optim.AdamW(net.parameters(), 
                                    lr=lr,
                                    weight_decay=weight_decay)
        # Decay LR by a factor of 0.1 every 7 epochs
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)


        ############################  training ###################################
        # Training. {{{
        # =====
        history = log.History(keys=('train_loss',
                                    'val_loss',
                                    'epoch_cnt',
                                    'train_L1_loss',
                                    'val_L1_loss',
                                    'test_L1_loss',),
                                    output_dir=out_path)

        #best_model_wts = copy.deepcopy(net.state_dict())
        #best_acc = 0.0

        output_list = []
        input_list  = []
        score_list  = []
        index_list  = []

        min_loss    = 10
        min_L1_loss = 10
        best_model_wts = 0

        for epoch in range(num_epochs):# {{{epoch
            loop = tqdm(train_loader, unit='batch',desc='Epoch {:>3}'.format(epoch+1))

            # Train Step. {{{
            # =====
            net.train()

            for _, batch in enumerate(loop):
                
                G_meter = log.AverageMeter()
                inputs, score, txt, _ = batch

                txt = list(txt)

                # gpuに送る
                inputs = inputs.to(device)
                score = score.to(device)

                if "clip" in txt_enc_name:
                    txt_input = tokenizer(txt, return_tensors="pt", padding=True).to(device)
                else:
                    txt_input = tokenizer.batch_encode_plus(txt, max_length=80, padding="max_length", return_tensors="pt").to(device)
                #print(txt_input["input_ids"].shape)
                #print(txt_input["input_ids"][0])
                # Update network. {{{
                # =====
                # forward network
                outputs = net(inputs, txt_input["input_ids"],txt_input["attention_mask"])

                # backward network{{{
                optimizer.zero_grad()  # 勾配を０
                loss = criterion(outputs, score)
                l1_loss = l1_norm(outputs, score)
                loss.backward()
                optimizer.step()
                # }}}backward network

                # Get losses. {{{}}
                G_meter.update(loss.item(), inputs[0].size()[0])
                history({'train_loss': loss.item()})
                history({'train_L1_loss': l1_loss.item()})
                    # }}} get loss
                
                # }}}Update network

            # Print training log. {
            # =====
            msg = "[Train {}] Epoch {}/{}".format(
                'ResNet50', epoch + 1, num_epochs)
            msg += " - {}: {:.4f}".format('train_loss', G_meter.avg)
            msg += " - {}: {:.4f}".format('learning rate',
                                        scheduler.get_last_lr()[0])
            history({'epoch_cnt': epoch})

            print(msg)

            # }}}Train Step.

            # Validation Step. {
            # =====

            with torch.no_grad():  # 勾配
                net.eval()
                loop_val = tqdm(val_loader, unit='batch',desc='Epoch {:>3}'.format(epoch + 1))
                epoch_loss    = 0
                epoch_L1_loss = 0
                iter_cnt      = 0

                for _, batch in enumerate(loop_val):
                    iter_cnt = iter_cnt + 1
                    G_meter = log.AverageMeter()
                    inputs, score, txt, index = batch

                    inputs = inputs.to(device)
                    score = score.to(device)
                    index  = index.numpy().copy()
                    txt = list(txt)
                    if "clip" in txt_enc_name:
                        txt_input = tokenizer(txt, return_tensors="pt", padding=True).to(device)
                    else:
                        txt_input = tokenizer.batch_encode_plus(txt, max_length=80, padding="max_length", return_tensors="pt").to(device)
                    #print(txt_input["input_ids"].shape)
                    outputs = net(inputs, txt_input["input_ids"],txt_input["attention_mask"])

                    loss = criterion(outputs, score)
                    l1_loss = l1_norm(outputs, score)

                    epoch_loss    = epoch_loss + loss
                    epoch_L1_loss = epoch_L1_loss + l1_loss

                    G_meter.update(loss.item(), inputs[0].size()[0])
                    history({'val_loss': loss.item()})
                    history({'val_L1_loss': l1_loss.item()})

                    inputs_img  = inputs.to('cpu').detach().numpy().copy()
                    output = outputs.to('cpu').detach().numpy().copy()
                    score  = score.to('cpu').detach().numpy().copy()

                # 50epoch毎にレーダーチャート保存{{
                if epoch  == 50:
                        # フォルダ作成
                    val_path = out_path + f"/{epoch}_epoch_result"
                    if not os.path.exists(val_path):
                        os.mkdir(val_path)
                    output_list.extend(output)
                    input_list.extend(inputs_img)
                    score_list.extend(score)
                    index_list.extend(index)
                    history.radar_chart(input_img=input_list[:10],
                                        output=output_list[:10], 
                                        score=score_list[:10], 
                                        dataframe=test_list[idx], 
                                        index_list=index_list,
                                        save_path=val_path,
                                        filename='epoch_radar_chart.png')
                    output_list = []
                    input_list  = []
                    score_list  = []
                    index_list  = []

                #}}
            # deep copy the model{{{
            epoch_loss_mean    = epoch_loss / iter_cnt
            epoch_L1_loss_mean = epoch_L1_loss / iter_cnt

            if epoch_loss_mean < min_loss:
                min_loss = epoch_loss_mean
                best_model_wts = copy.deepcopy(net.state_dict())
                print("best_model_wts updated")

            if epoch_L1_loss_mean < min_L1_loss:
                min_L1_loss = epoch_L1_loss_mean
            #}}}

            # } val step

            # Print validation log. {
            # =====
            msg = "[Validation {}] Epoch {}/{}".format(
                'VL', epoch + 1, num_epochs)
            msg += " - {}: {:.4f}".format('val_loss', G_meter.avg)

            print(msg)
            # } val log

            # sheduler step
            scheduler.step()
        # }}}} epoch
        
        # 重み保存
        torch.save(best_model_wts, out_path+"/model_dict.pth")

        #~~~~~~~~~~~~~~~~~~~  net setup~~~~~~~~~~~~~~~~~~~~~~~~
        net = BentoModel(txt_enc_name=txt_enc_name, num_visual_last_dim=num_visual_last_dim, num_text_last_dim=num_text_last_dim, bottle=bottle)
        # load weights
        net.load_state_dict(torch.load(out_path+"/model_dict.pth"))
        net = net.to(device)

        input_list  = []
        output_list = []
        score_list  = []
        index_list  = []
        test_loss_list = []
        # ~~~~~~~~~~~~~~ testdataに対する推論 ~~~~~~~~~~~~~~~~~~~~~
        print("~~~~~~~~~~~~~~ eval test data ~~~~~~~~~~~~~~~~~~")
        with torch.no_grad():  # 勾配の消失
            net.eval()
            loop_test = tqdm(test_loader, unit='batch',desc='Epoch {:>3}'.format(epoch + 1))

            for _, batch in enumerate(loop_test):
                inputs, score, txt, index = batch

                inputs = inputs.to(device)
                score = score.to(device)
                index  = index.numpy().copy()
                txt = list(txt)
                if "clip" in txt_enc_name:
                    txt_input = tokenizer(txt, return_tensors="pt", padding=True).to(device)
                else:
                    txt_input = tokenizer.batch_encode_plus(txt, max_length=80, padding="max_length", return_tensors="pt").to(device)
                #print(txt_input["input_ids"].shape)
                outputs = net(inputs, txt_input["input_ids"],txt_input["attention_mask"])

                l1_loss = l1_norm(outputs, score)
                
                history({'test_L1_loss': l1_loss.item()})
                
                inputs_img  = inputs.to('cpu').detach().numpy().copy()
                output = outputs.to('cpu').detach().numpy().copy()
                score  = score.to('cpu').detach().numpy().copy()
                

                input_list.extend(inputs_img)
                output_list.extend(output)
                score_list.extend(score)
                index_list.extend(index)
                test_loss_list.append(l1_loss.item())

        test_path = out_path + "/test_result"
        if not os.path.exists(test_path):
            os.mkdir(test_path)

        history.radar_chart(input_img=input_list[:30],
                            output=output_list[:30], 
                            score=score_list[:30], 
                            dataframe=test_list[idx], 
                            index_list=index_list,
                            save_path=test_path,
                            filename='test_radar_chart.png')

        output_list = np.array(output_list)
        score_list  = np.array(score_list)
        index_list  = np.array(index_list)

        o_df = pd.DataFrame(output_list)
        s_df = pd.DataFrame(score_list)
        i_df = pd.DataFrame(test_list[idx]["path"][index_list])

        #out_df   = pd.concat([i_df, o_df],axis=1)
        #out_df = out_df.dropna(how='all')
        #out_df = out_df.reset_index(drop=True)


        #score_df = pd.concat([i_df, s_df],axis=1)
        #score_df = score_df.dropna(how='all')
        #score_df = score_df.reset_index(drop=True)

        o_df.to_csv(out_path + "/test_out_df.csv")
        s_df.to_csv(out_path + "/test_score_df.csv")
        i_df.to_csv(out_path+ "/testdata_index.csv")


        # 相関係数

        # ~~~~~~ plot graph ~~~~~~~~
        
        ## log出力用相関係数算出
        corr_dict = history.calc_corr_dict(score_df=s_df, out_df=o_df)

        #print("# ~~~~~~ plotting graph ~~~~~~~~")
        history.plot_loss("MSE") # 引数：MSE or MAE
        history.plot_loss("MAE")
        history.score_pred_plot(output=output_list, score=score_list)
        history.save()

        # ~~~~~~ save log ~~~~~~~~~ 
        #dt_now = datetime.datetime.now()
        savepath = os.path.join(log_path, 'log.csv')
        if not os.path.exists(savepath):
            with open(savepath, 'w', encoding='utf_8_sig') as f: # 'w' 上書き
                writer = csv.writer(f)
                writer.writerow([
                            "txt_enc_name",
                            "seed",
                            "epoch",
                            "num_visual_last_dim",
                            "num_text_last_dim",
                            "bottle",
                            "optim",
                            "batch_size",
                            "in_w",
                            "lr",
                            "weight_decay",
                            "min_L1_loss",
                            "corr_list",
                            "corr mean",
                            "test loss",
                            "min_MSE_loss",
                            "closs_valid",
                            "random rotate degree",])
            

        with open(savepath, 'a', encoding='utf_8_sig') as f: # 'a' 追記
            writer = csv.writer(f)
            writer.writerow([
                            txt_enc_name,
                            seed,
                            num_epochs,
                            num_visual_last_dim,
                            num_text_last_dim,
                            bottle,
                            optim_flg,
                            batch_size,
                            in_w,
                            lr,
                            weight_decay,
                            min_L1_loss.to('cpu').detach().numpy().copy(),
                            corr_dict,
                            sum(corr_dict.values()) / len(corr_dict),
                            np.mean(test_loss_list),
                            min_loss.to('cpu').detach().numpy().copy(),
                            idx,
                            aug_degrees,])
        print(f"completed{idx}_phrase")

        all_corr_list.append(corr_dict)
        all_test_loss_list.append(np.mean(test_loss_list))


    all_corr_mean, all_corr_std       =  history.calc_clossvalid_corr_mean_std(corr_list=all_corr_list)
    all_test_loss_mean                =  np.mean(all_test_loss_list)


    # ~~~~~~ save log ~~~~~~~~~ 
    #dt_now = datetime.datetime.now()
    log_savepath = os.path.join(log_path, 'clossvalid_log.csv')
    if not os.path.exists(log_savepath):
        with open(log_savepath, 'w', encoding='utf_8_sig') as f: # 'w' 上書き
            writer = csv.writer(f)
            writer.writerow([
                        "txt_enc_name",
                        "seed",
                        "epoch",
                        "num_visual_last_dim",
                        "num_text_last_dim",
                        "bottle",
                        "optim",
                        "batch_size",
                        "in_w",
                        "lr",
                        "weight_decay",
                        "all corr",
                        "all std",
                        "test loss",
                        "corr mean",
                        "random rotate degree",])

    with open(log_savepath, 'a', encoding='utf_8_sig') as f: # 'a' 追記
        writer = csv.writer(f)
        writer.writerow([
                        txt_enc_name,
                        seed,
                        num_epochs,
                        num_visual_last_dim,
                        num_text_last_dim,
                        bottle,
                        optim_flg,
                        batch_size,
                        in_w,
                        lr,
                        weight_decay,
                        all_corr_mean,
                        all_corr_std,
                        all_test_loss_mean,
                        sum(all_corr_mean.values()) / len(all_corr_mean),
                        aug_degrees])


    print("~~~~~~~~~~ completed ~~~~~~~~~~~~")
if __name__ == '__main__':
    main()
