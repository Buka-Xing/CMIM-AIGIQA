import torch
from torch import nn
from modules.encoders import CPC, MMILB

import ImageReward
from inspect import isfunction
from einops import rearrange, repeat
from torch import einsum

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)  # 注意K和V分别是text_feature里提取的, 所以最终结果可以看作图像文本的相似性对文本特征的加权?
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 根据提供的函数对指定序列做映射

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale  # 就是q和k做矩阵乘法

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class ImageTextRegression(nn.Module):
    def __init__(self, inchannels=768, outchannels=512):
        super().__init__()

        self.down_channel = nn.Conv2d(1024, 768, kernel_size=1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        scale = inchannels ** -0.5
        self.cross_attention = CrossAttention(outchannels)
        self.norm1 = nn.LayerNorm(outchannels)
        self.norm2 = nn.LayerNorm(outchannels)
        self.proj = nn.Parameter(scale * torch.randn(inchannels, outchannels))

        # Self Added
        self.q_predictor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(1024, 1024 // 4),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024 // 4, 1),
        )

        # self.gating_network = nn.Sequential(
        #     nn.Linear(outchannels, 4),
        #     nn.Softmax(dim=1)
        # )
        # proj_experts = []
        # self.proj_nums = 4
        # for i in range(self.proj_nums):
        #     proj_experts.append(Projection(outchannels, outchannels))
        # self.proj_experts = nn.Sequential(*proj_experts)

    def forward(self, x, text_features):
        f_dis = self.down_channel(x.unsqueeze(-1).unsqueeze(-1))
        f_dis = self.conv(f_dis)

        B, C, W, H = f_dis.shape
        L = W * H

        f_dis = f_dis.view(B, C, L).permute(0, 2, 1).contiguous()

        text_features = text_features.unsqueeze(1) @ self.proj  # 实际上等同于torch.bmm 投影到512维度
        # text_features = text_features @ self.proj  # 实际上等同于torch.bmm 投影到512维度

        f_dis = self.norm1(f_dis)

        # f_dis = f_dis + self.cross_attention(f_dis, self.norm2(text_features))  # 以相加的形式完成Cross-attention
        f_dis = torch.cat([f_dis, self.cross_attention(f_dis, self.norm2(text_features))], dim=-1)  # self-added

        f_dis = f_dis.permute(0, 2, 1).contiguous().view(B, -1, W, H)

        f_dis = self.pool(f_dis)

        f_dis = f_dis.view(f_dis.size(0), -1).squeeze()

        pred  = self.q_predictor(f_dis)
        return f_dis, pred

class CMIM(nn.Module):
    def __init__(self, hp, device):
        """Construct CMIM-AIGIQA model.
        Args: 
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()
        self.hp = hp
        hp.d_tout = hp.d_tin
        self.device = device
        self.fix_rate = 0.5   # freeze half of the paramaters of BLIP

        # BLIP with ImageReward weights
        self.reward = ImageReward.load("ImageReward-v1.0", device=self.device)
        for name, parms in self.reward.mlp.named_parameters():
            parms.requires_grad_(False)
        for name, parms in self.reward.blip.named_parameters():
            if '_proj' in name:
                parms.requires_grad_(False)
        if self.fix_rate > 0:
            text_fix_num = "layer.{}".format(int(12 * self.fix_rate))
            image_fix_num = "blocks.{}".format(int(24 * self.fix_rate))
            for name, parms in self.reward.blip.text_encoder.named_parameters():
                parms.requires_grad_(False)
                if text_fix_num in name:
                    break
            for name, parms in self.reward.blip.visual_encoder.named_parameters():
                parms.requires_grad_(False)
                if image_fix_num in name:
                    break

        # Forward MIM
        self.mi_vt = MMILB(
            x_size = hp.d_vout,
            y_size = hp.d_tout,
            mid_activation = hp.mmilb_mid_activation,
            last_activation = hp.mmilb_last_activation
        )

        # CPC MI bound
        self.cpc_zt = CPC(
            x_size = hp.d_tout, # to be predicted
            y_size = hp.d_prjh,
            n_layers = hp.cpc_layers,
            activation = hp.cpc_activation
        )
        self.cpc_zv = CPC(
            x_size = hp.d_vout,
            y_size = hp.d_prjh,
            n_layers = hp.cpc_layers,
            activation = hp.cpc_activation
        )

        # self.fusion_prj = SubNet(
        #     in_size = dim_sum,
        #     hidden_size = hp.d_prjh,
        #     n_class = hp.n_class,
        #     dropout = hp.dropout_prj
        # )

        # Self Settings
        self.fusion_prj = ImageTextRegression()

        self.down_channel = nn.Conv2d(1024, 768, kernel_size=1)
        self.cross_attention = CrossAttention(768)
        self.text_pick = nn.Conv2d(35, 1, kernel_size=1)
        self.img_pick  = nn.Conv2d(197, 1, kernel_size=1)

    def forward(self, sentences, visual):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        # BLIP
        text_input = self.reward.blip.tokenizer(sentences, padding='max_length', truncation=True, max_length=35,
                                                return_tensors="pt").to(visual.device)

        image_embeds = self.reward.blip.visual_encoder(visual)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(visual.device)
        text_output = self.reward.blip.text_encoder(text_input.input_ids,
                                                    attention_mask=text_input.attention_mask,
                                                    encoder_hidden_states=image_embeds,
                                                    # 要通过image embedding生成text embedding
                                                    encoder_attention_mask=image_atts,
                                                    return_dict=True,
                                                    )
        text_features = text_output.last_hidden_state
        image_embedding = self.down_channel(image_embeds[:,1:,:].permute(0,2,1).unsqueeze(-1)).squeeze(-1).permute(0,2,1)
        text = text_features + self.cross_attention(text_features, image_embedding)

        text     = text[:,0,:]  # We only pick the CLS token for both text and img features
        visual   = image_embeds[:,0,:]

        # Forward MIM
        lld_tv, H_tv = self.mi_vt(x=visual, y=text)

        # Multimodal fusion and score prediction
        fusion, preds = self.fusion_prj(visual, text)

        # Backward MIM
        nce_t = self.cpc_zt(text, fusion)
        nce_v = self.cpc_zv(visual, fusion)
        
        nce = nce_t + nce_v

        lld = lld_tv
        H = H_tv

        return lld, nce, preds, H