from cProfile import label
import os
from xml.dom.minidom import AttributeList
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.cm as cm
import pickle


#from logging import getLogger


# 平均と現在の値を計算
class AverageMeter(object):
    def __init__(self):
        self.reset()

    @property
    def avg(self):
        return self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

#学習曲線を描いたり
class History(object):

    def __init__(self, keys, output_dir):

        self.output_dir = output_dir
        self.keys = keys

        self.logs = {key: [] for key in keys}

        col = ["color","healthy","satisfaction","uniqueness","ease of eating","appropriate amount","not collapse"]
        col_ja = ["色鮮やか", "体にいい, 体によさそう", "ボリューム感がある", "珍しい", "一口サイズ", "早く食べれる", "場所が固まっている"]

        attribute_name = ["male","female","age~40","age40~50","age50~60","age60~"]

        self.col = col
        self.col_ja = col_ja
        self.attribute_name = attribute_name

        plt.rcParams["figure.figsize"] = (6.4, 4.8)
    
    # 引数：data　historyが呼ばれたときに辞書型のkeyとvalueをlogsに追加する
    def __call__(self, data): 
        for key, value in data.items():
            self.logs[key].append(value)

    def save(self, filename='history.pkl'):
        savepath = os.path.join(self.output_dir, filename)
        with open(savepath, 'wb') as f:
            pickle.dump(self.logs, f)

        with open(os.path.join(self.output_dir, "histry.csv"), 'a') as f: # 'a' 追記
            writer = csv.writer(f)
            for key, value in self.logs.items():
                writer.writerow([key,value])


    def plot_loss(self,loss_name ,filename='loss.png'):# 引数：loss_nameはMSE or MAE
        filename = loss_name + filename
        if loss_name == 'MSE':
            train_key = 'train_loss'
            val_key   = 'val_loss'
        elif loss_name == 'MAE':
            train_key = 'train_L1_loss'
            val_key   = 'val_L1_loss'
        
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title(train_key)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title(val_key)

        train_x = np.arange(len(self.logs[train_key]))
        train_y = self.logs[train_key]

        val_x = np.arange(len(self.logs[val_key]))
        val_y = self.logs[val_key]

        ax1.plot(train_x, train_y, label=train_key, color='blue')
        ax2.plot(val_x, val_y, label=val_key, color='red')

        #plt.rcParams["figure.figsize"] = (6.4, 4.8)
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        if loss_name == 'MAE':
            ax1.set_ylim([0, 0.25])
            ax2.set_ylim([0, 0.25])
        else:
            ax1.set_ylim([0, 0.05])
            ax2.set_ylim([0, 0.05])
        fig.tight_layout()              #レイアウトの設定
        plt.show()

        save_path = os.path.join(self.output_dir, filename)
        #logger.info('Save {}'.format(save_path))
        plt.savefig(save_path, transparent=True)
        plt.clf() # 図全体をクリア
        plt.cla() # 軸をクリア
        plt.close('all') # closes all the figure windows


    def plot_roc_curve(self, filename='roc.png'):
        fpr = self.logs['fpr']
        tpr = self.logs['tpr']
        plt.plot(fpr, tpr, marker='o')
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.grid()

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, transparent=True)
        plt.clf() # 図全体をクリア
        plt.cla() # 軸をクリア
        plt.close('all') # closes all the figure windows


    def plot_acc(self, filename='acc.png'):
        train_x = np.arange(len(self.logs['train_acc']))
        train_acc = self.logs['train_acc']

        val_x = np.arange(len(self.logs['val_acc']))
        val_acc = self.logs['val_acc']


        plt.title("acc")
        plt.plot(train_x, train_acc, color='blue', label='train_acc')
        plt.plot(val_x, val_acc, color='red', label='val_acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend(loc='best')

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, transparent=True)
        plt.clf() # 図全体をクリア
        plt.cla() # 軸をクリア
        plt.close('all') # closes all the figure windows

    # 入力はバッチごとのリストで来る
    def radar_chart(self ,input_img, output, score, dataframe , index_list,save_path, filename='radar_chart.png'):

        #col = ["彩り", "健康的", "満足感", "ユニークさ", "食べやすさ", "適量", "くずれない"]
        col = self.col

        for i in range(len(output)):
            
            L1_loss = abs(output[i] - score[i])

            # 多角形を閉じるためにデータの最後に最初の値を追加する。
            output_values = np.append(output[i], output[i][0])
            score_values  = np.append(score[i], score[i][0])

            # プロットする角度を生成する。
            angles = np.linspace(0, 2 * np.pi, len(col) + 1 , endpoint=True)

            fig = plt.figure(figsize=(12, 12))

            # libのmatplotのファイルのデフォルトフォントを変えた
            #plt.rcParams['font.family'] = 'IPAexGothic'
            #print(plt.rcParams["font.family"])
            #plt.rcParams["font.family"] = 'sans-serif'   # 使用するフォント
            ax0 = fig.add_subplot(2, 2, 1)
            ax1 = fig.add_subplot(2, 2, 2, polar=True)


            ax0.imshow(input_img[i].transpose(1, 2, 0))
            ax0.set_title(dataframe['path'][index_list[i]], pad=20)


            # 極座標でaxを作成。
            # レーダーチャートの線を引く
            ax1.plot(angles, output_values, label="output")
            ax1.plot(angles, score_values, label="ground truth")

            # 項目ラベルの表示
            ax1.set_thetagrids(angles[:-1] * 180 / np.pi, col)#,fontname="IPAexGothic"
            ax1.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0]) # メモリ線
            
            ax1.legend(bbox_to_anchor=(1, 1), loc='center', borderaxespad=0)

            ax1.set_title(f"L1loss_mean:{np.round(L1_loss.mean(), 3)}" , pad=20)
            plt.show()

            img_path = save_path +  f"/{i}_"+filename

            plt.savefig(img_path)#, transparent=True)
            plt.clf() # 図全体をクリア
            plt.cla() # 軸をクリア
            plt.close('all') # closes all the figure windows


    # yyplot 作成関数
    def score_pred_plot(self, score, output, filename='score_predicted.png'):
        #output = output.to('cpu').detach().numpy().copy()
        #score  = score.to('cpu').detach().numpy().copy()
        
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20,10))
        #col = ["彩り", "健康的", "満足感", "ユニークさ", "食べやすさ", "適量", "くずれない"]
        ax = ax.flatten()

        #plt.rcParams["font.family"] = 'sans-serif'   # 使用するフォント
        col = self.col

        for i in range(1,len(ax)):

            ax[i].scatter(  score[:,i-1], 
                            output[:,i-1], 
                            label=col[i-1] + f":{np.corrcoef(score[:,i-1],output[:,i-1])[0,1]}")
            ax[i].set_xlim(0,1)
            ax[i].set_ylim(0,1)
            ax[i].set_xlabel('score', fontsize=14)
            ax[i].set_ylabel('output', fontsize=14)
            ax[i].legend(loc="upper right")


        save_path = os.path.join(self.output_dir, filename)
        
        plt.savefig(save_path, transparent=True)
        plt.clf() # 図全体をクリア
        plt.cla() # 軸をクリア
        plt.close('all') # closes all the figure windows


    # yyplot 作成関数

    # こいつはdfを引数とすることに注意:うえのplotとは異なる
    def score_pred_plot_by_attribute(self, score_df, output_df, filename='attribute_score_predicted.png'):
        #output = output.to('cpu').detach().numpy().copy()
        #score  = score.to('cpu').detach().numpy().copy()
        
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20,10))
        #col = ["彩り", "健康的", "満足感", "ユニークさ", "食べやすさ", "適量", "くずれない"]
        ax = ax.flatten()

        #plt.rcParams["font.family"] = 'sans-serif'   # 使用するフォント
        col = self.col

        for i in range(len(ax)-1):
            temp_score = score_df[i]
            temp_out = output_df[i]

            ax[i].scatter(  temp_score, 
                            temp_out, 
                            label=col[i] + f":{np.corrcoef(temp_score,temp_out)[0,1]}")
            ax[i].set_xlim(0,1)
            ax[i].set_ylim(0,1)
            ax[i].set_xlabel('score', fontsize=14)
            ax[i].set_ylabel('output', fontsize=14)
            ax[i].legend(loc="upper right")


        save_path = os.path.join(self.output_dir, filename)
        
        plt.savefig(save_path, transparent=True)
        plt.clf() # 図全体をクリア
        plt.cla() # 軸をクリア
        plt.close('all') # closes all the figure windows


    def calc_corr_dict(self, score_df, out_df):
        corr_dict = {}
        for i in range(7):
            corr_dict[self.col_ja[i]] = np.corrcoef(score_df[i],out_df[i])[0,1]
        
        return corr_dict

    # 交差検証の各因子に対しての相関係数の平均、標準偏差を算出し、辞書型で返す
    def calc_clossvalid_corr_mean_std(self, corr_list):
        corr_mean_dict = {}
        corr_std_dict = {}

        for factor_i in range(len(self.col_ja)):
            temp_list = []
            for cv_num in range(len(corr_list)):
                temp_list.append(corr_list[cv_num][self.col_ja[factor_i]])
            corr_mean_dict[self.col_ja[factor_i]] = np.mean(temp_list)
            corr_std_dict[self.col_ja[factor_i]] = np.std(temp_list)

        return corr_mean_dict, corr_std_dict
