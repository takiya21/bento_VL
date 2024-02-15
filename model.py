import os
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights
import torchinfo
from torchvision.models.feature_extraction import get_graph_node_names,create_feature_extractor
from torch import Tensor
import transformers

class BentoModel(nn.Module):
    def __init__(self, txt_enc_name:str, num_visual_last_dim:int, num_text_last_dim:int, bottle:int, out_dim:int = 7)->None:
        super().__init__()

        self.bottle = bottle
        self.out_dim = out_dim
        self.txt_enc_name = txt_enc_name
        max_length = 80

        
        #画像エンコーダー
        resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)#.requires_grad_(False)
        if "resnet" in self.txt_enc_name:
            num_ftrs = resnet18.fc.in_features
            resnet18.fc = nn.Linear(num_ftrs, 7)
            self.resnet18 = resnet18
        else:
            #resnet18.eval()
            self.resnet18_feature_extractor = create_feature_extractor(
                resnet18, 
                return_nodes={"flatten":"features"}
            )
        #言語エンコーダー
        if "clip" in txt_enc_name:
            self.text_model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")#.requires_grad_(False)
            #self.text_model.eval()
        elif "T5" in txt_enc_name:
            self.text_model = transformers.T5EncoderModel.from_pretrained('t5-small')#.requires_grad_(False)
            #self.text_model.eval()
        
        
        #結合層
        self.gap = nn.AdaptiveAvgPool2d((max_length,1))
        self.txt_fc1 = nn.Linear(max_length, num_text_last_dim)
        self.txt_fc = nn.Linear(512, num_text_last_dim)
        self.fc1 = nn.Linear(num_visual_last_dim+num_text_last_dim, self.bottle)
        self.vision_fc = nn.Linear(512, num_visual_last_dim)
        self.only_fc = nn.Linear(num_text_last_dim, self.bottle)
        self.fc2 = nn.Linear(self.bottle, out_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, img:Tensor, input_ids:Tensor, attention_mask:Tensor)->Tensor:
        """_summary_

        Args:
            img (Tensor): (batch_size, 3, h, w)
            input_ids (Tensor): (batch_size, sequence_length)
            attention_mask (Tensor): (batch_size, sequence_length)

        Returns:
            Tensor: _description_
        """
        # print("img.shape:", img.shape)
        # print("input_ids.shape:", input_ids.shape)

        if "only" in self.txt_enc_name:
            txt_f = self.text_model(input_ids,attention_mask)["pooler_output"]
            x = self.only_fc(txt_f)
            x = self.fc2(x)
            x = self.sigmoid(x)

        elif "resnet" in self.txt_enc_name:
            img_f = self.resnet18(img)
            x = self.sigmoid(img_f)
            
        elif "T5" in self.txt_enc_name:
            img_f = self.resnet18_feature_extractor(img)["features"] # (1, 512)
            txt_f = self.text_model(input_ids,attention_mask)["last_hidden_state"]
            txt_f = self.gap(txt_f).view(img_f.size(0), -1)
            txt_f = self.txt_fc1(txt_f)
            img_f = self.vision_fc(img_f)
            x = torch.cat((img_f,txt_f), 1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.sigmoid(x)

        else:
            img_f = self.resnet18_feature_extractor(img)["features"] # (1, 512)
            txt_f = self.text_model(input_ids,attention_mask)["pooler_output"]
            img_f = self.vision_fc(img_f)
            txt_f = self.txt_fc(txt_f)
            x = torch.cat((img_f,txt_f), 1)#画像言語特徴量の結合
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.sigmoid(x)

        return x