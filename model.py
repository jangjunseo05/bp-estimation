import torch
import torch.nn as nn

# Transformer Encoder Layer 정의
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        """
        하나의 Transformer 인코더 레이어 구성
        - d_model: 입력 및 출력 차원
        - n_heads: 멀티 헤드 수
        - d_ff: FeedForward 네트워크 내부 차원
        - dropout: 드롭아웃 비율
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(            # Position-wise Feedforward Network
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)  # 첫 번째 LayerNorm (Residual 연결 후)
        self.norm2 = nn.LayerNorm(d_model)  # 두 번째 LayerNorm (Feedforward 후)
        self.dropout = nn.Dropout(dropout)  # 드롭아웃

    def forward(self, x):
        """
        입력 x: (B,L,d_model)
        """
        attn_output, _ = self.attn(x, x, x)             # Self-attention (Q=K=V=x)
        x = self.norm1(x + self.dropout(attn_output))   # 첫 번째 residual + norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))     # 두 번째 residual + norm
        return x

# 전체 Transformer 기반 시계열 회귀 모델 (TSTModel)
class TSTModel(nn.Module):

    def __init__(self, input_dim, d_model, n_heads, d_ff, num_layers, dropout, output_length, max_len=50):
        """
        TST(Time Series Transformer) 모델
        - input_dim: MiniRocket feature 차원
        - d_model: Transformer 내부 임베딩 차원
        - n_heads: Attention head 수
        - d_ff: FeedForward hidden 차원
        - num_layers: Encoder 레이어 수
        - dropout: 드롭아웃 비율
        - max_len: PAT 시퀀스 길이 (default: 50)
        """
        super().__init__()
        self.proj = nn.Linear(1, d_model)   # 입력 차원 -> Transformer d_model 차원으로 투영
        self.pos_embedding = nn.Parameter(torch.randn(1,max_len,d_model))

        self.encoders = nn.Sequential(*[
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])                                     # 다층 Transformer 인코더
        
        self.head = nn.Linear(d_model+input_dim, output_length)     # 최종 출력틍 (SBP,DBP 회귀)

    def forward(self, pat_seq, rocket_feat):
        """
        - pat_seq: (B,1,L) - 시계열
        - rocket_feat: (B, D) - 벡터
        """
        x = pat_seq.permute(0, 2, 1)  # (B, C, L) -> (B, L, C): Transformer가 L을 시퀀스로 인식하도록 
        x = self.proj(x)        # (B, L, d_model): input_dim -> d_model 차원 변환
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.encoders(x)    # (B, L, d_model): Transformer 인코더 통과
        x = x.mean(dim=1)       # (B, d_model): 전체 시퀀스 평균 (global average pooling)

        combined = torch.cat([x, rocket_feat], dim=1)       # (B, d_model + D)
        return self.head(combined)                          # (B, output_length): 예측 ABP 파형
