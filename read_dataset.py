import json
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from sklearn import preprocessing
from scipy.stats import pearsonr # 科学技術計算ライブラリ
from sklearn.preprocessing import OneHotEncoder

#PytorchのDatasetクラスを継承
class read_dataset(Dataset):
    def __init__(self, df, language_id, transform=None):
        
        image_paths = df['path']
        json_dir = "/dataset/dataset/taki/data/bento/bento_dataset1000_json"
        score = {}
        for i in range(1,8):
            tmp = f"factor{i}"
            score[tmp] = df[tmp]

        self.df = df

        # image_pathsとjson_pathsは同indexに同じ画像のbasenameが入っている（例：index 4には11n24y (24).jpgと11n24y (24).jsonが入っている）
        self.image_paths = image_paths
        self.json_paths = image_paths.str.replace(".jpg", ".json")


        # jsonファイルを読み込み
        json_data_dict = {}
        for path in self.json_paths:
            with open(os.path.join(json_dir, path), "r") as f:
                json_data = json.load(f)
                json_data_dict[path] = json_data

        self.json_data_dict = json_data_dict
        self.score = score
        self.transform = transform
        self.language_id = language_id # 0:english, 1:japanese




    def __getitem__(self, index):#引数のindexの画像の情報を返す
        path = "/dataset/dataset/taki/data/bento/bento_dataset1000/" +  self.image_paths[index]
        
        #　画像読み込み
        img = Image.open(path)

        #transform事前処理実施
        if self.transform is not None:
            img = self.transform(img)
        
        img = torch.Tensor(img)
        
        # score読み込み
        score = self.df.loc[index].drop(["path"])
        score = torch.Tensor(score)

        txt = ""
        for data in self.json_data_dict[self.json_paths[index]]["shapes"]:
            if data["group_id"] == self.language_id:
                txt = txt + data["label"] + "."

        return img, score, txt, index


    def __len__(self):
        #データ数を返す
        return len(self.image_paths)

if __name__ == '__main__':
    #transformで32x32画素に変換して、テンソル化。
    transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
    #データセット作成
    dataset = read_dataset("./score_ml_promax_7.csv",transform)
    #dataloader化
    dataloader = DataLoader(dataset, batch_size=32)

    #データローダの中身確認
    for img,label ,image_path in dataloader:
        print('label=',label)
        print('image_path=',image_path)
        print('img.shape=',img.shape)