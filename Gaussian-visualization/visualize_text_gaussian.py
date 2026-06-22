"""
Empirically test whether BLIP text features E_t(P) follow a Gaussian distribution.

Pipeline:
    CSV(prompt column) -> BLIP text encoder -> features (N, 768)
    -> PCA(2D) scatter + theoretical Gaussian contours
    -> per-dimension QQ plots (max-var / min-var / 2 random dims)
    -> D'Agostino-Pearson normality test summary
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse  # noqa: F401  (kept for users who want ellipse instead of contour)

from sklearn.decomposition import PCA
from scipy import stats

import ImageReward

# ---- Patch: BLIP/MED in ImageReward unconditionally calls
# attention_probs.register_hook(...) for GradCAM, which crashes under
# torch.no_grad() because the tensor doesn't require grad.
# Make register_hook a no-op when the tensor has no grad.
_orig_register_hook = torch.Tensor.register_hook
def _safe_register_hook(self, hook):
    if not self.requires_grad:
        return None
    return _orig_register_hook(self, hook)
torch.Tensor.register_hook = _safe_register_hook


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default='../Database/AGIQA-3k/prompts.csv',
                   help="CSV file containing a 'prompt' column.")
    p.add_argument("--prompt_col", type=str, default="prompt",
                   help="Column name holding the text prompt (default: 'prompt').")
    p.add_argument("--out_dir", type=str,
                   default="./Gaussian-visualization/output",
                   help="Directory for figures and summary.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=35)
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=20200626)
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level (kept for D'Agostino fallback).")
    p.add_argument("--filliben_threshold", type=float, default=0.99,
                   help="Filliben's r threshold; dim is considered normal if r >= threshold "
                        "(typical critical values @alpha=0.05: ~0.987 for N=100, "
                        "~0.9925 for N=200, ~0.997 for N=500, ~0.9985 for N=1000).")
    p.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"],
                   help="Pooling for sequence features: 'cls' (token 0) or 'mean'.")
    return p.parse_args()


@torch.no_grad()
def extract_text_features(prompts, blip, tokenizer, device, batch_size=32,
                          max_length=35, pool="cls"):
    """Run BLIP text encoder on a list of prompts. Returns (N, D) numpy array."""
    feats = []
    blip.eval()

    # Try text-only mode first (no cross-attention to image). If unsupported,
    # fall back to multimodal mode with zero image embeddings.
    text_only_supported = True
    try:
        sample_in = tokenizer(["probe"], padding="max_length", truncation=True,
                              max_length=max_length, return_tensors="pt").to(device)
        _ = blip.text_encoder(sample_in.input_ids,
                              attention_mask=sample_in.attention_mask,
                              return_dict=True, mode="text")
    except (TypeError, ValueError):
        text_only_supported = False

    for i in tqdm(range(0, len(prompts), batch_size), desc="encoding text"):
        batch = prompts[i: i + batch_size]
        tok = tokenizer(batch, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        if text_only_supported:
            out = blip.text_encoder(tok.input_ids,
                                    attention_mask=tok.attention_mask,
                                    return_dict=True, mode="text")
        else:
            # Fallback: pass zero image embeds so cross-attn contributes ~0 modulation
            # (residual still propagates pure text). Image tokens count = 197 for ViT-L/16 @224.
            B = tok.input_ids.size(0)
            zero_img = torch.zeros(B, 197, 1024, device=device)
            zero_att = torch.ones(B, 197, dtype=torch.long, device=device)
            out = blip.text_encoder(tok.input_ids,
                                    attention_mask=tok.attention_mask,
                                    encoder_hidden_states=zero_img,
                                    encoder_attention_mask=zero_att,
                                    return_dict=True)
        h = out.last_hidden_state  # (B, L, D)
        if pool == "cls":
            f = h[:, 0, :]
        else:
            mask = tok.attention_mask.unsqueeze(-1).float()
            f = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        feats.append(f.float().cpu().numpy())

    feats = np.concatenate(feats, axis=0)
    return feats


def gaussian_contour_grid(mu, cov, xlim, ylim, n=200):
    """Evaluate 2D multivariate Gaussian density on a grid."""
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xs, ys)
    pos = np.dstack((X, Y))
    rv = stats.multivariate_normal(mean=mu, cov=cov, allow_singular=True)
    Z = rv.pdf(pos)
    return X, Y, Z


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- 1. Load prompts ----
    df = pd.read_csv(args.csv)
    if args.prompt_col not in df.columns:
        raise KeyError(f"Column '{args.prompt_col}' not found in {args.csv}. "
                       f"Available: {list(df.columns)}")
    prompts = [str(p) for p in df[args.prompt_col].tolist() if isinstance(p, str) or not pd.isna(p)]
    print(f"[info] loaded {len(prompts)} prompts from {args.csv}")

    # ---- 2. Load BLIP from ImageReward ----
    print(f"[info] loading ImageReward BLIP on {args.device} ...")
    reward = ImageReward.load("ImageReward-v1.0", device=args.device)
    blip = reward.blip
    tokenizer = blip.tokenizer

    # ---- 3. Extract text features ----
    feats = extract_text_features(prompts, blip, tokenizer,
                                  device=args.device,
                                  batch_size=args.batch_size,
                                  max_length=args.max_length,
                                  pool=args.pool)
    N, D = feats.shape
    print(f"[info] feature matrix: shape=({N}, {D})")

    # ---- 4. PCA to 2D + theoretical Gaussian contours ----
    pca = PCA(n_components=2, random_state=args.seed)
    feats2d = pca.fit_transform(feats)
    evr = pca.explained_variance_ratio_
    mu2 = feats2d.mean(axis=0)
    cov2 = np.cov(feats2d, rowvar=False)

    # ---- 5. Per-dim normality test ----
    # # [D'Agostino-Pearson] -- kept for reference; uncomment to switch back.
    # pvals = np.zeros(D)
    # for d in range(D):
    #     try:
    #         _, pvals[d] = stats.normaltest(feats[:, d])
    #     except ValueError:
    #         pvals[d] = 0.0
    # pass_rate = float(np.mean(pvals > args.alpha))

    # [Filliben's correlation coefficient]
    # r = corr(sorted_sample, theoretical_normal_quantiles_at_median_ranks)
    # scipy.stats.probplot returns this Pearson r (with Filliben's plotting positions
    # via order-statistic medians) as the third tuple element.
    filliben_r = np.zeros(D)
    for d in range(D):
        try:
            (_, _), (_, _, r) = stats.probplot(feats[:, d], dist="norm", fit=True)
            filliben_r[d] = r
        except ValueError:
            filliben_r[d] = 0.0
    pass_rate = float(np.mean(filliben_r >= args.filliben_threshold))

    # ---- 6. Pick representative dims for QQ ----
    variances = feats.var(axis=0)
    dim_max = int(np.argmax(variances))
    dim_min = int(np.argmin(variances))
    rng = np.random.default_rng(args.seed)
    remaining = list(set(range(D)) - {dim_max, dim_min})
    dim_rand = list(rng.choice(remaining, size=2, replace=False))
    qq_dims = [("max-var", dim_max), ("min-var", dim_min),
               ("random", dim_rand[0]), ("random", dim_rand[1])]

    # ---- 7. Combined figure: PCA panel (left) + 2x2 QQ grid (right) ----
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 4, width_ratios=[2, 2, 1, 1], wspace=0.35, hspace=0.35)

    # Left: PCA scatter spans both rows of the first 2 columns
    ax_pca = fig.add_subplot(gs[:, 0:2])
    ax_pca.scatter(feats2d[:, 0], feats2d[:, 1], s=8, alpha=0.35,
                   c="#1f77b4", edgecolors="none", label="text features")

    pad_x = 0.1 * (feats2d[:, 0].max() - feats2d[:, 0].min() + 1e-6)
    pad_y = 0.1 * (feats2d[:, 1].max() - feats2d[:, 1].min() + 1e-6)
    xlim = (feats2d[:, 0].min() - pad_x, feats2d[:, 0].max() + pad_x)
    ylim = (feats2d[:, 1].min() - pad_y, feats2d[:, 1].max() + pad_y)
    X, Y, Z = gaussian_contour_grid(mu2, cov2, xlim, ylim, n=250)

    levels = np.linspace(Z.max() * 0.05, Z.max() * 0.95, 5)
    cs = ax_pca.contour(X, Y, Z, levels=levels, colors="crimson", linewidths=1.4)
    ax_pca.clabel(cs, inline=True, fontsize=16, fmt="%.2g")

    ax_pca.set_xlim(xlim); ax_pca.set_ylim(ylim)
    ax_pca.set_xlabel(f"PC1  ({evr[0]*100:.2f}% var)")
    ax_pca.set_ylabel(f"PC2  ({evr[1]*100:.2f}% var)")
    ax_pca.set_title("(a) PCA(2D) of BLIP text features\nwith theoretical Gaussian density contours")
    ax_pca.legend(loc="upper right", fontsize=18, frameon=True)
    ax_pca.grid(alpha=0.25, linestyle="--")

    # Right: 2x2 QQ subplots
    qq_axes = [fig.add_subplot(gs[0, 2]),
               fig.add_subplot(gs[0, 3]),
               fig.add_subplot(gs[1, 2]),
               fig.add_subplot(gs[1, 3])]

    for ax, (label, d) in zip(qq_axes, qq_dims):
        x = feats[:, d]
        x_std = (x - x.mean()) / (x.std() + 1e-12)
        stats.probplot(x_std, dist="norm", plot=ax)
        # Recolor probplot defaults
        ax.get_lines()[0].set_marker(".")
        ax.get_lines()[0].set_markersize(3)
        ax.get_lines()[0].set_markerfacecolor("#1f77b4")
        ax.get_lines()[0].set_markeredgecolor("#1f77b4")
        ax.get_lines()[1].set_color("crimson")
        ax.get_lines()[1].set_linewidth(1.2)
        # # [D'Agostino-Pearson] -- kept for reference
        # _, p = stats.normaltest(x)
        # ax.set_title(f"dim {d} ({label})\nD'Agostino p={p:.2g}", fontsize=9)
        # [Filliben's r]
        r_d = filliben_r[d]
        ax.set_title(f"dim {d} ({label})\nFilliben r={r_d:.4f}", fontsize=10)
        ax.set_xlabel("Theoretical quantiles", fontsize=16)
        ax.set_ylabel("Sample quantiles", fontsize=16)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linestyle="--")

    fig.suptitle("(b) Per-dimension QQ plots vs N(0,1)", x=0.78, y=0.985, fontsize=15)
    fig.suptitle("Empirical Gaussianity of BLIP text features  E_t(P)",
                 fontsize=26, y=1.02)
    fig.tight_layout()

    out_pdf = os.path.join(args.out_dir, "blip_text_gaussian_check.pdf")
    out_png = os.path.join(args.out_dir, "blip_text_gaussian_check.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 8. Text summary ----
    summary = (
        "================ BLIP text-feature Gaussianity summary ================\n"
        f"  CSV                       : {args.csv}\n"
        f"  N (prompts)               : {N}\n"
        f"  D (feature dim)           : {D}\n"
        f"  PCA explained var ratio   : PC1 = {evr[0]*100:.3f}%,  PC2 = {evr[1]*100:.3f}%\n"
        f"  Cumulative (PC1+PC2)      : {(evr[0]+evr[1])*100:.3f}%\n"
        # f"  D'Agostino-Pearson alpha  : {args.alpha}\n"
        f"  Filliben r threshold      : {args.filliben_threshold}\n"
        f"  Filliben r (mean / median): {filliben_r.mean():.4f} / {np.median(filliben_r):.4f}\n"
        f"  Filliben r (min / max)    : {filliben_r.min():.4f} / {filliben_r.max():.4f}\n"
        f"  Dims passing normality    : {int(pass_rate*D)} / {D}  ({pass_rate*100:.2f}%)\n"
        f"  Selected QQ dims          : max-var={dim_max}, min-var={dim_min}, "
        f"rand=({dim_rand[0]}, {dim_rand[1]})\n"
        f"  Figures saved to          : {out_pdf}\n"
        f"                              {out_png}\n"
        "-----------------------------------------------------------------------\n"
        f"  Conclusion: {'features are broadly consistent with a Gaussian distribution.' if pass_rate >= 0.5 else 'features deviate substantially from Gaussian — assumption is NOT supported.'}\n"
        "======================================================================="
    )
    print(summary)
    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
