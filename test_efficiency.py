import time
import torch
import torch.nn as nn
from model import CMIM

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

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

def build_args():
    args = get_args()
    args.contrast = True
    args.d_prjh   = 1024
    args.d_tin    = 768
    args.n_class  = 1
    args.alpha    = 0.1
    args.beta     = 0.1
    args.lr_main  = 1e-5
    args.lr_bert  = 1e-5
    args.lr_mmilb = 1e-5
    args.num_epochs = 1
    args.batch_size = 1
    return args

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class MMIMTensorWrapper(nn.Module):
    """Wrap MMIM so its forward only takes tensors (pre-tokenized text + image),
    making it compatible with thop / fvcore FLOPs counters.

    Replicates the exact compute path of MMIM.forward at inference time
    (i.e. when y=None, mem=None), but skips returning lld/nce/H so we only
    measure the cost of producing `preds`.
    """
    def __init__(self, mmim):
        super().__init__()
        self.m = mmim

    def forward(self, input_ids, attention_mask, visual):
        m = self.m
        # BLIP visual + cross-modal text encoder
        image_embeds = m.reward.blip.visual_encoder(visual)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=visual.device)
        text_output = m.reward.blip.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        text_features = text_output.last_hidden_state

        image_embedding = m.down_channel(
            image_embeds[:, 1:, :].permute(0, 2, 1).unsqueeze(-1)
        ).squeeze(-1).permute(0, 2, 1)
        text = text_features + m.cross_attention(text_features, image_embedding)

        # Token-/patch-pick + image-text fusion regressor
        # mirrors the tail of MMIM.forward used to produce `preds`
        text_pick = m.text_pick(text.unsqueeze(-1)).squeeze(-1).squeeze(1)
        img_pick = m.img_pick(image_embeds.unsqueeze(-1)).squeeze(-1).squeeze(1)
        preds = m.fusion_prj(img_pick, text_pick)
        return preds


def count_parameters(model, name=""):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{name}] total: {total:,} ({total/1e6:.2f} M) | "
          f"trainable: {trainable:,} ({trainable/1e6:.2f} M)")
    return total, trainable


def measure_time_native(mmim, prompt, img, warmup=10, runs=50):
    """Use the native model(prompt, img) call (with tokenizer inside)."""
    mmim.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = mmim(prompt, img)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            _ = mmim(prompt, img)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.time()
    return (t1 - t0) / runs


def measure_flops(wrapper, inputs):
    if HAS_THOP:
        flops, _ = profile(wrapper, inputs=inputs, verbose=False)
        return flops
    if HAS_FVCORE:
        return FlopCountAnalysis(wrapper, inputs).total()
    return None


def main():
    args = build_args()

    mmim = CMIM(args, device=device)
    mmim.device = device
    mmim = mmim.to(device)
    mmim.eval()

    # ----- Parameters -----
    print("=" * 70)
    print("Parameter sizes (MI-AIGIQA / MMIM):")
    total_p, _ = count_parameters(mmim, "MMIM (whole)")
    count_parameters(mmim.reward, "  └─ ImageReward (BLIP)")
    count_parameters(mmim.fusion_prj, "  └─ fusion_prj (ImageTextRegression)")
    count_parameters(mmim.mi_vt, "  └─ mi_vt (MMILB)")
    count_parameters(mmim.cpc_zt, "  └─ cpc_zt")
    count_parameters(mmim.cpc_zv, "  └─ cpc_zv")
    print(f"  -> approx fp32 size: {total_p * 4 / (1024**2):.2f} MB")

    # ----- Dummy single-image inputs -----
    B = 1
    visual = torch.randn(B, 3, 224, 224, device=device)
    prompt = ["a generated image of a scene"]

    # Pre-tokenize for FLOPs wrapper (avoids passing list[str] to thop)
    text_input = mmim.reward.blip.tokenizer(
        prompt, padding='max_length', truncation=True,
        max_length=35, return_tensors="pt"
    ).to(device)

    wrapper = MMIMTensorWrapper(mmim).to(device)
    wrapper.eval()

    # Sanity check: tensor wrapper must run without error.
    with torch.no_grad():
        _ = wrapper(text_input.input_ids, text_input.attention_mask, visual)

    # ----- FLOPs -----
    print("=" * 70)
    flops = measure_flops(wrapper, (text_input.input_ids, text_input.attention_mask, visual))
    if flops is not None:
        print(f"FLOPs (1 image): {flops:,}  ({flops/1e9:.3f} GFLOPs)")
    else:
        print("FLOPs: skipped (install `thop` or `fvcore` to measure)")

    # ----- Inference time (native call, includes tokenizer overhead) -----
    print("=" * 70)
    avg_t = measure_time_native(mmim, prompt, visual, warmup=10, runs=50)
    print(f"Avg inference time (1 image): {avg_t*1000:.3f} ms  "
          f"({1.0/avg_t:.2f} FPS) on {device}")
    print("=" * 70)


if __name__ == "__main__":
    main()
