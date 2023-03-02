
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Reorder(nn.Module):
    def forward(self, input):
        return input.permute((0, 2, 1))
    
def seq_text_encoder(n_tokens, hid_size):
    return nn.Sequential(
        nn.Embedding(n_tokens, embedding_dim=hid_size),
        Reorder(),
        nn.Conv1d(in_channels=hid_size, out_channels=hid_size, kernel_size=3),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(output_size=1),
        Flatten()
    )


class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = seq_text_encoder(n_tokens=n_tokens, hid_size=hid_size)      
        
        self.full_emb = seq_text_encoder(n_tokens=n_tokens, hid_size=hid_size)
        
        self.category_out = nn.Sequential(
            nn.Linear(in_features=n_cat_features, out_features=hid_size),
            nn.ReLU(),
            nn.Linear(in_features=hid_size, out_features=hid_size),
            nn.ReLU()
        )

        # Example for the final layers (after the concatenation)
        self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=hid_size * 2)
        self.final_dense = nn.Linear(in_features=hid_size * 2, out_features=1)

        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        # print(input1.shape, input2.shape, input3.shape)
        title = self.title_emb(input1)
        # print(title.shape)
        full = self.full_emb(input2)
        # print(full.shape)
        category = self.category_out(input3)     
        # print(category.shape)  
        
        concatenated = torch.cat(
            [
                title,
                full,
                category
            ], dim=1
        )
        # print(concatenated.shape)
        
        out = self.inter_dense(concatenated)
        # print(out.shape)
        out = self.final_dense(out)
        # print(out.shape)
        
        return out