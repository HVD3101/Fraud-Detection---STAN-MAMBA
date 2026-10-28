import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_minimal import MambaBlock, ModelArgs, ResidualBlock


class stan_model(nn.Module):
    """
        Kiến trúc mô hình STAN + MAMBA:
        1. Spatio-Temporal Attention
        2. Linear Projection
        3. Mamba
        4. Global Pooling (MAX + AVG)
        5. Dense Layer
    """

    def __init__(
            self,
            time_windows_dim: int,
            spatio_windows_dim: int,
            feat_dim: int,
            num_classes: int,
            attention_hidden_dim: int,
            mamba_d_model: int = 128,  # Thêm kích thước d_model cho Mamba (không gian ẩn MAMBA)
    ) -> None:
        super().__init__()
        self.time_windows_dim = time_windows_dim
        self.spatio_windows_dim = spatio_windows_dim
        self.windows_dim = [self.time_windows_dim, self.spatio_windows_dim]
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.attention_hidden_dim = attention_hidden_dim
        self.mamba_d_model = mamba_d_model

        #1. Spatio-Temporal Attention
        self.attention_W_1 = nn.Parameter(torch.Tensor(self.feat_dim, self.attention_hidden_dim).uniform_(0., 1.))
        self.attention_W_2 = nn.Parameter(torch.Tensor(self.feat_dim * 2, self.attention_hidden_dim).uniform_(0., 1.))
        self.attention_U_1 = nn.Parameter(torch.Tensor(self.feat_dim, self.attention_hidden_dim).uniform_(0., 1.))
        self.attention_U_2 = nn.Parameter(torch.Tensor(self.feat_dim * 2, self.attention_hidden_dim).uniform_(0., 1.))
        self.attention_V_1 = nn.Parameter(torch.Tensor(self.attention_hidden_dim, 1).uniform_(0., 1.))
        self.attention_V_2 = nn.Parameter(torch.Tensor(self.attention_hidden_dim, 1).uniform_(0., 1.))

        #2. Linear Projection
        self.linear_proj = nn.Sequential(
            nn.LazyLinear(self.mamba_d_model),
            nn.LayerNorm(self.mamba_d_model)
        )

        # 3. MAMBA
        args = ModelArgs(
            d_model=self.mamba_d_model,
            n_layer=2,
            vocab_size=1,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.mamba = nn.Sequential(*[ResidualBlock(args) for _ in range(args.n_layer)])

        # 4 & 5. Global Pooling (Max + Avg) & Dense Layer
        self.dense = nn.Sequential(
            nn.Linear(self.mamba_d_model * 2, 64),  # *2 vì ghép Max và Avg
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3), # Ngắt ngẫu nhiên để tránh Overfitting
            nn.Linear(64, self.num_classes)
        )

    def attention_layer(self, X: torch.Tensor, dim=2):
        self.output_att = []
        input_att = torch.split(X, 1, dim=dim)
        for index, x_i in enumerate(input_att):
            x_i = x_i.reshape(-1, self.windows_dim[2 - dim], self.feat_dim * dim)
            c_i = self.attention(x_i, input_att, index, dim=dim)
            inp = torch.concat([x_i, c_i], axis=2)
            self.output_att.append(inp)

        input_conv = torch.reshape(torch.concat(self.output_att, axis=dim),
                                   [-1, self.time_windows_dim, self.spatio_windows_dim, self.feat_dim * 2 * dim])
        return input_conv

    def attention(self, x_i, x, index, dim=1):
        e_i = []
        c_i = []

        for i in range(len(x)):
            output = x[i]
            output = output.reshape(-1, self.windows_dim[2 - dim], self.feat_dim * dim)
            if dim == 1:
                att_hidden = torch.tanh(torch.add(torch.matmul(
                    x_i, self.attention_W_1), torch.matmul(output, self.attention_U_1)))
                e_i_j = torch.matmul(att_hidden, self.attention_V_1)
            else:
                att_hidden = torch.tanh(torch.add(torch.matmul(
                    x_i, self.attention_W_2), torch.matmul(output, self.attention_U_2)))
                e_i_j = torch.matmul(att_hidden, self.attention_V_2)
            e_i.append(e_i_j)

        e_i = torch.concat(e_i, axis=2)
        alpha_i = F.softmax(e_i, dim=2)
        alpha_i = torch.split(alpha_i, 1, 2)

        for j, (alpha_i_j, output) in enumerate(zip(alpha_i, x)):
            if j == index:
                continue
            else:
                output = output.reshape(-1, self.windows_dim[2 - dim], self.feat_dim * dim)
                c_i_j = torch.multiply(alpha_i_j, output)
                c_i.append(c_i_j)
        rshape = [-1, self.time_windows_dim, self.spatio_windows_dim, self.feat_dim * dim]
        rshape[dim] -= 1

        c_i = torch.reshape(torch.concat(c_i, axis=1), rshape)
        c_i = torch.sum(c_i, dim=dim)
        return c_i


    def forward(self, X_nume):
        # 1. Spatio-Temporal Attention
        out = self.attention_layer(X_nume, dim=1)
        out = self.attention_layer(out, dim=2)

        # 2. Làm phẳng Không gian và Đặc trưng (S x F)
        batch_size, time_dim, spatio_dim, f_dim = out.shape
        out = out.view(batch_size, time_dim, spatio_dim * f_dim)

        # 3. Chiếu tuyến tính (Linear Layer)
        out = self.linear_proj(out)

        # 4. Mamba Block
        out = self.mamba(out)

        # 5. Global Pooling (MAX + AVG)
        avg_pool = out.mean(dim=1)
        max_pool = out.max(dim=1)[0]
        out = torch.cat([avg_pool, max_pool], dim=1)

        # 6. Mạng phân loại (Dense Layer)
        out = self.dense(out)

        return out