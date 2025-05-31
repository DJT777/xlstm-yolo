
class FusionMLPBase(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or 4 * dim

    def forward(self, x):
        raise NotImplementedError

class MLPBaseline(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class GEGLU(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.fc = nn.Linear(dim, self.hidden_dim * 2)
        self.proj = nn.Linear(self.hidden_dim, dim)

    def forward(self, x):
        x1, x2 = self.fc(x).chunk(2, dim=-1)
        return self.proj(F.gelu(x1) * x2)

class SwiGLU(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.fc = nn.Linear(dim, self.hidden_dim * 2)
        self.proj = nn.Linear(self.hidden_dim, dim)

    def forward(self, x):
        x1, x2 = self.fc(x).chunk(2, dim=-1)
        return self.proj(F.silu(x1) * x2)

class RGBlock(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        local_dim = self.hidden_dim * 2 // 3
        self.fc1 = nn.Conv2d(dim, local_dim * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(local_dim, local_dim, kernel_size=3, padding=1, groups=local_dim)
        self.fc2 = nn.Conv2d(local_dim, dim, kernel_size=1)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = F.gelu(self.dwconv(x) + x) * v
        return self.fc2(x)

class ConvMLP(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, self.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, dim, kernel_size=1)
        )

    def forward(self, x):
        return self.mlp(x)

class LoRAMLP(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None, rank=16):
        super().__init__(dim, hidden_dim)
        self.rank = min(rank, self.hidden_dim)
        self.down = nn.Linear(dim, self.rank)
        self.up = nn.Linear(self.rank, dim)

    def forward(self, x):
        return self.up(F.relu(self.down(x)))

class MLPMixer(FusionMLPBase):
    def __init__(self, dim, seq_len, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(seq_len, seq_len),
        )
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, dim)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # B, C, S
        x = self.token_mlp(x)
        x = x.transpose(1, 2)
        return self.channel_mlp(x)

class CrossAttentionMLP(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, dim)

    def forward(self, x1, x2):
        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x2)
        attn = F.softmax(q @ k.transpose(-2, -1) / (self.dim ** 0.5), dim=-1)
        return self.out(attn @ v)

class FiLMMLP(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, dim)
        )

    def forward(self, x, modulator):
        gamma = self.gamma(modulator)
        beta = self.beta(modulator)
        return self.ffn(x) * gamma + beta


# -------------------------------------------------------
# Registry of MLP Blocks (use dictionary for swappable logic)
# -------------------------------------------------------
MLP_REGISTRY = {
    "baseline": lambda dim, **kwargs: MLPBaseline(dim, **kwargs),
    "geglu": lambda dim, **kwargs: GEGLU(dim, **kwargs),
    "swiglu": lambda dim, **kwargs: SwiGLU(dim, **kwargs),
    "rgblock": lambda dim, **kwargs: RGBlock(dim, **kwargs),
    "convmlp": lambda dim, **kwargs: ConvMLP(dim, **kwargs),
    "lora": lambda dim, **kwargs: LoRAMLP(dim, **kwargs),
    "mixer": lambda dim, seq_len=64, **kw: MLPMixer(dim, seq_len=seq_len, **kw),
    "crossattn": lambda dim, **kwargs: CrossAttentionMLP(dim, **kwargs),
    "film": lambda dim, **kwargs: FiLMMLP(dim, **kwargs),
}

# -------------------------------------------------------
# FusionViLLayer Class
# -------------------------------------------------------
# - [-1, 1, FusionViLLayerBlock, [256, {
#     "proj_type": "conv",
#     "mlp_type": "swiglu",
#     "seq_len": 64,
#     "use_mlp": true
# }]]

class FusionViLLayer(nn.Module):
    def __init__(
        self,
        dim,
        direction="rowwise_from_top_left",
        mlp_type="baseline",
        mlp_hidden_dim=None,
        use_skip=True,
        use_mlp=True,
        conv_kind="2d",
        conv_kernel_size=3,
        proj_bias=True,
        norm_bias=True,
        seqlens=None,
        num_blocks=1,
        init_weights="original",
        seq_len=None,
        proj_type="linear",  # 'linear', 'conv', or 'sequenceconv'
    ):
        super().__init__()
        self.use_skip = use_skip
        self.use_mlp = use_mlp
        self.seq_len = seq_len
        self.proj_type = proj_type

        # Project + Normalize: supports 3 projection types
        if proj_type == "linear":
            self.input_proj = nn.Linear(dim * 2, dim)
        elif proj_type == "conv":
            self.input_proj = nn.Sequential(
                nn.Conv2d(dim * 2, dim, kernel_size=1, bias=proj_bias),
                nn.BatchNorm2d(dim),
                nn.SiLU()
            )
        elif proj_type == "sequenceconv":
            self.input_proj = SequenceConv2d(
                in_channels=dim * 2,
                out_channels=dim,
                kernel_size=1,
                padding=0,
                bias=proj_bias,
                seqlens=seqlens
            )
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)

        self.vilayer = ViLLayer(
            dim=dim,
            direction=direction,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            seqlens=seqlens,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            num_blocks=num_blocks,
            init_weights=init_weights,
        )

        self.residual_proj = nn.Identity() if not use_skip else nn.Linear(dim, dim)

        if use_mlp:
            self.norm2 = LayerNorm(ndim=dim)
            self.post_mlp = MLP_REGISTRY[mlp_type](
                dim, hidden_dim=mlp_hidden_dim or dim * 4, seq_len=seq_len
            )
        else:
            self.post_mlp = None

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        S = H * W

        if self.proj_type == "conv":
            x = torch.cat([x1, x2], dim=1)           # [B, 2C, H, W]
            x = self.input_proj(x)                  # [B, C, H, W]
            x_seq = rearrange(x, "b c h w -> b (h w) c")
        else:
            x1_seq = rearrange(x1, "b c h w -> b (h w) c")
            x2_seq = rearrange(x2, "b c h w -> b (h w) c")
            x = torch.cat([x1_seq, x2_seq], dim=-1)  # [B, S, 2C]
            x_seq = self.input_proj(x) if self.proj_type == "linear" else self.input_proj(x)

        fused = self.norm(x_seq)
        fused_out = self.vilayer(fused)

        if self.use_skip:
            fused_out = fused_out + self.residual_proj(x1_seq)

        if self.use_mlp:
            fused_out = fused_out + self.post_mlp(self.norm2(fused_out))

        return rearrange(fused_out, "b (h w) c -> b c h w", h=H, w=W)

class ViLLayerLite(nn.Module):
    def __init__(
        self,
        dim,
        direction,
        conv_kind="2d",
        conv_kernel_size=3,
        proj_bias=True,
        norm_bias=True,
        conv_bias=True,
        mlp_type="baseline",         # 🔄 Swappable via MLP_REGISTRY
        mlp_kwargs=None,             # 🔧 Extra kwargs like hidden_dim, rank, seq_len
        seqlens=None,
    ):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.conv_kind = conv_kind
        self.seqlens = seqlens or (14, 14)

        # Convolution selection
        if conv_kind == "causal1d":
            self.conv = CausalConv1d(dim=dim, kernel_size=conv_kernel_size, bias=conv_bias)
        elif conv_kind == "2d":
            self.conv = SequenceConv2d(
                in_channels=dim, out_channels=dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=dim, bias=conv_bias,
                seqlens=self.seqlens if len(self.seqlens) == 2 else None,
            )
        elif conv_kind == "3d":
            self.conv = SequenceConv3d(
                in_channels=dim, out_channels=dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=dim, bias=conv_bias,
                seqlens=self.seqlens if len(self.seqlens) == 3 else None,
            )
        else:
            raise NotImplementedError(f"conv_kind='{conv_kind}' not supported.")

        # q/k/v projections
        self.q_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.k_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.v_proj = nn.Linear(dim, dim, bias=proj_bias)

        self.mlstm_cell = MatrixLSTMCell(dim=dim, num_heads=1, norm_bias=norm_bias)
        self.learnable_skip = nn.Parameter(torch.ones(dim))

        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)

        # Instantiate MLP via registry
        mlp_kwargs = mlp_kwargs or {}
        if mlp_type == "mixer":
            seq_len = math.prod(self.seqlens)
            mlp_kwargs.setdefault("seq_len", seq_len)

        assert mlp_type in MLP_REGISTRY, f"Unknown MLP type: {mlp_type}"
        self.mlp = MLP_REGISTRY[mlp_type](dim=dim, **mlp_kwargs)

    def forward(self, x: torch.Tensor, seqlens=None) -> torch.Tensor:
        B, S, D = x.shape

        if seqlens and hasattr(self.conv, "seqlens"):
            self.conv.seqlens = seqlens

        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])

        x_conv = self.conv(x)
        x_conv_act = F.silu(x_conv)

        q = self.q_proj(x_conv_act)
        k = self.k_proj(x_conv_act)
        v = self.v_proj(x)

        h_tilde = self.mlstm_cell(q=q, k=k, v=v)
        h_tilde_skip = h_tilde + self.learnable_skip * x_conv_act

        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            h_tilde_skip = h_tilde_skip.flip(dims=[1])

        x_norm = self.norm(h_tilde_skip)

        # MLP forward with shape handling
        if isinstance(self.mlp, (ConvMLP, RGBlock)):
            H, W = self.seqlens if len(self.seqlens) == 2 else (int(S**0.5), int(S**0.5))
            x_img = rearrange(x_norm, "b (h w) d -> b d h w", h=H)
            x_out = self.mlp(x_img)
            x_out = rearrange(x_out, "b d h w -> b (h w) d")
        else:
            x_out = self.mlp(x_norm)

        return h_tilde_skip + x_out
