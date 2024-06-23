import torch
import torch.nn as nn
from efficient_kan.kan import KAN

class KANeRF(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=32):   
        super(KANeRF, self).__init__()
        
        print(embedding_dim_pos * 6 + 3)
        self.block1    = KAN([embedding_dim_pos * 6 + 3, hidden_dim, hidden_dim])
        # density estimation
        self.block2    = KAN([embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim, hidden_dim+1])

        # color estimation
        self.block3    = KAN([embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2])
        self.block4    = KAN([hidden_dim // 2, 3])

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos) # emb_x: [batch_size, embedding_dim_pos * 6]
        emb_d = self.positional_encoding(d, self.embedding_dim_direction) # emb_d: [batch_size, embedding_dim_direction * 6]        
        h = self.block1(emb_x) # h: [batch_size, hidden_dim]
        tmp = self.block2(torch.cat((h, emb_x), dim=1)) # tmp: [batch_size, hidden_dim + 1]
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1]) # h: [batch_size, hidden_dim], sigma: [batch_size]
        h = self.block3(torch.cat((h, emb_d), dim=1)) # h: [batch_size, hidden_dim // 2]
        c = self.block4(h) # c: [batch_size, 3]
        return c, sigma