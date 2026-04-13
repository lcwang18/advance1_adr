import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LoFTRBlock(nn.Module):
    """
    LoFTR-paradigm block with:
    1) Self-Attention on each modality
    2) Cross-Attention between source and target
    """

    def __init__(self, dim: int, num_heads: int = 8, ff_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.self_attn_src = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_tgt = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_src = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_tgt = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1_src = nn.LayerNorm(dim)
        self.norm1_tgt = nn.LayerNorm(dim)
        self.norm2_src = nn.LayerNorm(dim)
        self.norm2_tgt = nn.LayerNorm(dim)
        self.norm3_src = nn.LayerNorm(dim)
        self.norm3_tgt = nn.LayerNorm(dim)

        self.ffn_src = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, dim),
        )
        self.ffn_tgt = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, src_seq: torch.Tensor, tgt_seq: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        src_sa, _ = self.self_attn_src(src_seq, src_seq, src_seq)
        tgt_sa, _ = self.self_attn_tgt(tgt_seq, tgt_seq, tgt_seq)
        src_seq = self.norm1_src(src_seq + src_sa)
        tgt_seq = self.norm1_tgt(tgt_seq + tgt_sa)

        # LoFTR-style cross attention for dense implicit correlation
        src_ca, _ = self.cross_attn_src(src_seq, tgt_seq, tgt_seq)
        tgt_ca, _ = self.cross_attn_tgt(tgt_seq, src_seq, src_seq)
        src_seq = self.norm2_src(src_seq + src_ca)
        tgt_seq = self.norm2_tgt(tgt_seq + tgt_ca)

        src_ffn = self.ffn_src(src_seq)
        tgt_ffn = self.ffn_tgt(tgt_seq)
        src_seq = self.norm3_src(src_seq + src_ffn)
        tgt_seq = self.norm3_tgt(tgt_seq + tgt_ffn)
        return src_seq, tgt_seq


class LoFTRAffine(nn.Module):
    """
    Coarse registration module:
    - 1/8-resolution feature extraction
    - Self/Cross-attention Transformer stack
    - GAP + MLP affine regression
    """

    def __init__(self, in_channels: int = 1, feat_dim: int = 128, num_blocks: int = 2, num_heads: int = 8):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 1/2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 1/4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feat_dim, kernel_size=3, stride=2, padding=1, bias=False),  # 1/8
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList(
            [LoFTRBlock(dim=feat_dim, num_heads=num_heads, ff_dim=feat_dim * 2) for _ in range(num_blocks)]
        )
        self.affine_head = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 6),
        )
        self._init_affine_head()

    def _init_affine_head(self) -> None:
        # Initialize to identity affine transform.
        final_fc = self.affine_head[-1]
        nn.init.zeros_(final_fc.weight)
        identity_bias = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        final_fc.bias.data.copy_(identity_bias)

    def _to_sequence(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        return feat.flatten(2).transpose(1, 2).contiguous()  # [B, HW, C]

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat_src = self.feature_net(src)  # 1/8 resolution
        feat_tgt = self.feature_net(tgt)  # 1/8 resolution
        src_seq = self._to_sequence(feat_src)
        tgt_seq = self._to_sequence(feat_tgt)

        for blk in self.blocks:
            src_seq, tgt_seq = blk(src_seq, tgt_seq)

        # Global average pooling over attended tokens, then MLP to affine params.
        src_gap = src_seq.mean(dim=1)
        tgt_gap = tgt_seq.mean(dim=1)
        fused = torch.cat([src_gap, tgt_gap], dim=-1)
        theta = self.affine_head(fused).view(-1, 2, 3)
        return {"theta": theta, "feat_src": feat_src, "feat_tgt": feat_tgt}


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.conv(x)
        return x, self.pool(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class EvidentialUNet(nn.Module):
    """
    Fine registration U-Net with evidential regression head:
    outputs 8 channels = [gamma_x, eta_x, kappa_x, rho_x, gamma_y, eta_y, kappa_y, rho_y].
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 32):
        super().__init__()
        self.enc1 = Down(in_channels, base_channels)
        self.enc2 = Down(base_channels, base_channels * 2)
        self.enc3 = Down(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.dec3 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = Up(base_channels * 2, base_channels, base_channels)

        # Evidential head: 8 channels (NIG params for x/y axis)
        self.head = nn.Conv2d(base_channels, 8, kernel_size=1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        x = self.bottleneck(x)

        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        raw = self.head(x)
        gamma_x, eta_x_raw, kappa_x_raw, rho_x_raw, gamma_y, eta_y_raw, kappa_y_raw, rho_y_raw = torch.chunk(
            raw, 8, dim=1
        )

        # Evidential constraints:
        # eta > 0, kappa > 1, rho > 0.
        eta_x = self.softplus(eta_x_raw) + 1e-6
        kappa_x = self.softplus(kappa_x_raw) + 1.0
        rho_x = self.softplus(rho_x_raw) + 1e-6

        eta_y = self.softplus(eta_y_raw) + 1e-6
        kappa_y = self.softplus(kappa_y_raw) + 1.0
        rho_y = self.softplus(rho_y_raw) + 1e-6

        gamma = torch.cat([gamma_x, gamma_y], dim=1)  # expected DVF
        eta = torch.cat([eta_x, eta_y], dim=1)
        kappa = torch.cat([kappa_x, kappa_y], dim=1)
        rho = torch.cat([rho_x, rho_y], dim=1)
        return {"gamma": gamma, "eta": eta, "kappa": kappa, "rho": rho}


class SpatialTransformer(nn.Module):
    def __init__(self, align_corners: bool = True, padding_mode: str = "border"):
        super().__init__()
        self.align_corners = align_corners
        self.padding_mode = padding_mode

    def warp_affine(self, img: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        grid = F.affine_grid(theta, img.size(), align_corners=self.align_corners)
        return F.grid_sample(
            img,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

    def warp_dvf(self, img: torch.Tensor, dvf: torch.Tensor) -> torch.Tensor:
        """
        dvf is expected as normalized displacement field with shape [B, 2, H, W].
        """
        b, _, h, w = img.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=img.device, dtype=img.dtype),
            torch.linspace(-1.0, 1.0, w, device=img.device, dtype=img.dtype),
            indexing="ij",
        )
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)  # [B, H, W, 2]
        final_grid = base_grid + dvf.permute(0, 2, 3, 1)
        return F.grid_sample(
            img,
            final_grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


class Advanced_ADRNet(nn.Module):
    """
    Advanced ADRNet:
    1) Coarse LoFTR-style affine registration
    2) Fine U-Net evidential DVF prediction
    """

    def __init__(
        self,
        in_channels: int = 1,
        loftr_feat_dim: int = 128,
        loftr_blocks: int = 2,
        loftr_heads: int = 8,
        unet_base_channels: int = 32,
    ):
        super().__init__()
        self.coarse = LoFTRAffine(
            in_channels=in_channels,
            feat_dim=loftr_feat_dim,
            num_blocks=loftr_blocks,
            num_heads=loftr_heads,
        )
        self.fine = EvidentialUNet(in_channels=in_channels * 2, base_channels=unet_base_channels)
        self.stn = SpatialTransformer()

    def forward(self, image_src: torch.Tensor, image_tgt: torch.Tensor) -> Dict[str, torch.Tensor]:
        coarse_out = self.coarse(image_src, image_tgt)
        theta = coarse_out["theta"]

        # Stage-1 coarse warp with affine grid_sample
        src_warp_affine = self.stn.warp_affine(image_src, theta)

        # Stage-2 fine input: concat target + coarsely warped source
        fine_in = torch.cat([image_tgt, src_warp_affine], dim=1)
        evidential_out = self.fine(fine_in)
        dvf = evidential_out["gamma"]

        # Final warp with predicted expected DVF
        src_warp_final = self.stn.warp_dvf(src_warp_affine, dvf)

        return {
            "theta": theta,
            "feat_src": coarse_out["feat_src"],
            "feat_tgt": coarse_out["feat_tgt"],
            "warped_affine": src_warp_affine,
            "warped_final": src_warp_final,
            "dvf": dvf,
            "gamma": evidential_out["gamma"],
            "eta": evidential_out["eta"],
            "kappa": evidential_out["kappa"],
            "rho": evidential_out["rho"],
        }


class LossManager(nn.Module):
    def __init__(
        self,
        w_sim: float = 1.0,
        w_evidential: float = 0.1,
        w_gcl: float = 0.05,
        ncc_window: int = 9,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.w_sim = w_sim
        self.w_evidential = w_evidential
        self.w_gcl = w_gcl
        self.ncc_window = ncc_window
        self.eps = eps

    def local_ncc_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Local NCC loss used for image similarity:
        L_sim = 1 - mean(local NCC).
        """
        b, c, _, _ = x.shape
        win = self.ncc_window
        pad = win // 2
        weight = torch.ones((c, 1, win, win), device=x.device, dtype=x.dtype)

        x2 = x * x
        y2 = y * y
        xy = x * y

        x_sum = F.conv2d(x, weight, padding=pad, groups=c)
        y_sum = F.conv2d(y, weight, padding=pad, groups=c)
        x2_sum = F.conv2d(x2, weight, padding=pad, groups=c)
        y2_sum = F.conv2d(y2, weight, padding=pad, groups=c)
        xy_sum = F.conv2d(xy, weight, padding=pad, groups=c)

        win_size = float(win * win)
        x_mean = x_sum / win_size
        y_mean = y_sum / win_size

        cross = xy_sum - y_mean * x_sum - x_mean * y_sum + x_mean * y_mean * win_size
        x_var = x2_sum - 2.0 * x_mean * x_sum + x_mean * x_mean * win_size
        y_var = y2_sum - 2.0 * y_mean * y_sum + y_mean * y_mean * win_size

        ncc = (cross * cross) / (x_var * y_var + self.eps)
        return 1.0 - ncc.mean()

    def evidential_nig_nll(
        self,
        gamma: torch.Tensor,
        eta: torch.Tensor,
        kappa: torch.Tensor,
        rho: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative Log-Evidence for NIG (per-pixel):
        using gamma as mean, eta/rho/kappa as uncertainty-aware parameters.
        """
        # Mapping to standard NIG naming:
        # mu=gamma, beta=eta, alpha=kappa (>1), v=rho (>0)
        mu = gamma
        beta = eta + self.eps
        alpha = kappa + self.eps
        v = rho + self.eps

        sq_error = (target - mu).pow(2)
        term1 = 0.5 * torch.log(math.pi / v)
        term2 = -alpha * torch.log(2.0 * beta * (1.0 + v))
        term3 = (alpha + 0.5) * torch.log(v * sq_error + 2.0 * beta * (1.0 + v))
        term4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        nll = term1 + term2 + term3 + term4
        return nll.mean()

    def grid_geometry_consistency(self, dvf: torch.Tensor) -> torch.Tensor:
        """
        Grid Geometry Consistency Loss (GCL):
        Total variation regularization over predicted DVF to preserve grid collinearity.
        """
        dx = torch.abs(dvf[:, :, :, 1:] - dvf[:, :, :, :-1])
        dy = torch.abs(dvf[:, :, 1:, :] - dvf[:, :, :-1, :])
        return dx.mean() + dy.mean()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        image_tgt: torch.Tensor,
        pseudo_gt_dvf: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        gamma = outputs["gamma"]
        eta = outputs["eta"]
        kappa = outputs["kappa"]
        rho = outputs["rho"]
        dvf = outputs["dvf"]
        warped_final = outputs["warped_final"]

        if pseudo_gt_dvf is None:
            pseudo_gt_dvf = torch.zeros_like(gamma)

        # Similarity loss (Local NCC)
        l_sim = self.local_ncc_loss(warped_final, image_tgt)

        # Uncertainty-aware evidential loss
        l_evidential = self.evidential_nig_nll(gamma, eta, kappa, rho, pseudo_gt_dvf)

        # Grid geometry consistency loss
        l_gcl = self.grid_geometry_consistency(dvf)

        total = self.w_sim * l_sim + self.w_evidential * l_evidential + self.w_gcl * l_gcl
        return {
            "total_loss": total,
            "L_sim": l_sim,
            "L_evidential": l_evidential,
            "L_GCL": l_gcl,
        }


if __name__ == "__main__":
    # Dummy test block for shape/runtime verification.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_src = torch.randn(2, 1, 256, 256, device=device)
    image_tgt = torch.randn(2, 1, 256, 256, device=device)

    model = Advanced_ADRNet(
        in_channels=1,
        loftr_feat_dim=128,
        loftr_blocks=2,
        loftr_heads=8,
        unet_base_channels=32,
    ).to(device)
    loss_manager = LossManager().to(device)

    outputs = model(image_src, image_tgt)
    pseudo_gt_dvf = torch.zeros_like(outputs["dvf"])
    loss_dict = loss_manager(outputs, image_tgt, pseudo_gt_dvf=pseudo_gt_dvf)

    loss_dict["total_loss"].backward()

    print("Forward successful.")
    print("theta shape:", outputs["theta"].shape)
    print("warped_final shape:", outputs["warped_final"].shape)
    print("dvf shape:", outputs["dvf"].shape)
    print({k: float(v.detach().cpu()) for k, v in loss_dict.items()})
