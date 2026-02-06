import torch
import torch.nn as nn
import torch.nn.functional as F


from utils.utils import initialize_weights, softmax_one
from 


class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of attention channels
    """
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        module = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        if dropout:
            module.append(nn.Dropout(0.25))

        module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*module)

    def forward(self, x):
        """
        x: [N, L]
        returns:
            A: [N, n_classes]
            x: [N, L] (passthrough)
        """
        return self.module(x), x


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout
        n_classes: number of attention channels
    """
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()

        attention_a = [
            nn.Linear(L, D),
            nn.Tanh()
        ]

        attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid()
        ]

        if dropout:
            attention_a.append(nn.Dropout(0.25))
            attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*attention_a)
        self.attention_b = nn.Sequential(*attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        """
        x: [N, L]
        returns:
            A: [N, n_classes]
            x: [N, L]
        """
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a * b
        A = self.attention_c(A)
        return A, x


# ============================================================
# 2. SIMPLE CROSS-ATTENTION BLOCK
# ============================================================

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): First feature set of shape (seq_len, dim).
            x2 (torch.Tensor): Second feature set of shape (seq_len, dim).
        
        Returns:
            torch.Tensor: Fused feature representation of shape (seq_len, dim).
        """
        # Compute Q, K, V
        Q = self.query(x1)  # (seq_len, dim)
        K = self.key(x2)  # (seq_len, dim)
        V = self.value(x2)  # (seq_len, dim)

        # Compute scaled dot-product attention
        attn_weights = self.softmax(Q @ K.T / (x1.shape[-1] ** 0.5))  # (seq_len, seq_len)
        attn_output = attn_weights @ V  # (seq_len, dim)

        # Add & Norm
        output = self.norm(x1 + attn_output)  # (seq_len, dim)

        return output 


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        print(x.shape)
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
    

class CMA_MIL(nn.Module):
    """
    Multi-scale MIL model with:
      - Cross-magnification attention (5x->10x, 10x->20x, 20x->5x optionally)
      - Gated attention per magnification
      - Attribute-weighted bag-level logits
      - Instance-level supervision (top-k positive / negative patches)
 """
    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=0.2,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
        embed_dim=512,
        dim=128,
    ):
        super().__init__()

        # size_dict: [input_dim, hidden_dim, attn_hidden_dim]
        self.size_dict = {
            "small": [embed_dim, 128, 64],
            "big": [embed_dim, 256, 384],
        }
        size = self.size_dict[size_arg]

        self.n_classes = n_classes
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping

        self.classifiers = nn.Linear(size[1], n_classes)

        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(size[1], 2) for _ in range(n_classes)]
        )

        # Cross-magnification attention blocks
        self.cross_attn_5x_10x = CrossAttention(dim, dropout=dropout)
        self.cross_attn_10x_20x = CrossAttention(dim,dropout=dropout)
        self.cross_attn_20x_5x = CrossAttention(dim, dropout=dropout)

        # Adapt 5x/10x/20x features to shared latent dim
        self.adaptor_5x = nn.Sequential(nn.Linear(embed_dim, dim), nn.ReLU())
        self.adaptor_10x = nn.Sequential(nn.Linear(embed_dim, dim), nn.ReLU())
        self.adaptor_20x = nn.Sequential(nn.Linear(embed_dim, dim), nn.ReLU())

        # Gated attention over instances at each magnification
        self.attention_net = Attn_Net_Gated(L=dim, D=dim // 2, dropout=dropout, n_classes=1)

        # Classifier to produce per-instance class scores
        self.classifier = nn.Linear(dim, n_classes)

        self.logit_weights = nn.Parameter(torch.ones(3))  
        self.bias = nn.Parameter(torch.zeros(n_classes), requires_grad=True)

        
    # --------------------------------------------------------
    # Utility target creators
    # --------------------------------------------------------
    @staticmethod
    def create_targets(length, value, device):
        return torch.full((length,), value, device=device).long()

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # --------------------------------------------------------
    # Instance-level evaluation (top-k pos / neg)
    # --------------------------------------------------------
    def inst_eval(self, ss, A, h, classifier, positive=True):
        """
        ss: 1D or 2D tensor of scores used to rank instances (shape [N] or [1, N])
        A:  attention scores (shape [1, N])
        h:  instance features (shape [N, dim])
        classifier: nn.Linear(dim, 2)
        """
        device = h.device

        if ss.dim() == 1:
            ss = ss.view(1, -1)
        if A.dim() == 1:
            A = A.view(1, -1)

        k_selected = min(self.k_sample, A.shape[1])

        # Top-k positive / negative indices from ss
        top_p_ids = torch.topk(ss, k_selected, dim=1)[1][-1]   # [k]
        top_n_ids = torch.topk(-ss, k_selected, dim=1)[1][-1]  # [k]

        top_p = torch.index_select(h, dim=0, index=top_p_ids)  # [k, dim]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)  # [k, dim]

        p_targets = self.create_positive_targets(k_selected, device)
        n_targets = self.create_negative_targets(k_selected, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)        # [2k]
        all_instances = torch.cat([top_p, top_n], dim=0)              # [2k, dim]

        # Attention scores corresponding to the chosen instances
        top_p_A = A[0, top_p_ids]   # [k]
        top_n_A = A[0, top_n_ids]   # [k]
        all_A = torch.cat([top_p_A, top_n_A], dim=0)  # [2k]
        all_A = all_A.unsqueeze(0)                    # [1, 2k]

        # Raw instance logits (2-way classification)
        instance_score = classifier(all_instances).unsqueeze(0)  # [1, 2k, 2]

        # Expand attention weights for each class dimension
        A_expanded = all_A.unsqueeze(1).expand(-1, 2, -1).permute(0, 2, 1)  # [1, 2k, 2]

        # Attribute-weighted scores
        attribute_score = instance_score * torch.exp(A_expanded)            # [1, 2k, 2]
        attribute_score = attribute_score.squeeze(0)                        # [2k, 2]

        instance_loss = self.instance_loss_fn(attribute_score, all_targets)

        # Weighted logits (bag-level from these instances) if needed
        logits = torch.sum(attribute_score, dim=0, keepdim=True) / torch.sum(
            torch.exp(A_expanded.squeeze(0)), dim=0, keepdim=True
        )  # [1, 2]

        return instance_loss, logits, all_targets

	def inst_eval_out(self, ss, A, h, classifier, positive=True):
        """
        ss: 1D or 2D tensor of scores used to rank instances (shape [N] or [1, N])
        A:  attention scores (shape [1, N])
        h:  instance features (shape [N, dim])
        classifier: nn.Linear(dim, 2)
        """
        device = h.device

        if ss.dim() == 1:
            ss = ss.view(1, -1)
        if A.dim() == 1:
            A = A.view(1, -1)

        k_selected = min(self.k_sample, A.shape[1])

        top_p_ids = torch.topk(ss, k_selected, dim=1)[1][-1]   # [k]

        top_p = torch.index_select(h, dim=0, index=top_p_ids)  # [k, dim]

        p_targets = self.create_negative_targets(k_selected, device)

        # Attention scores corresponding to the chosen instances
        top_p_A = A[0, top_p_ids]   # [k]
        all_A = top_p_A.unsqueeze(0)                    # [1, k]

        # Raw instance logits (2-way classification)
        instance_score = classifier(top_p).unsqueeze(0)  # [1, 2k, 2]

        # Expand attention weights for each class dimension
        A_expanded = all_A.unsqueeze(1).expand(-1, 2, -1).permute(0, 2, 1)  # [1, 2k, 2]

        # Attribute-weighted scores
        attribute_score = instance_score * torch.exp(A_expanded)            # [1, 2k, 2]
        attribute_score = attribute_score.squeeze(0)                        # [2k, 2]

        instance_loss = self.instance_loss_fn(attribute_score, p_targets)

        # Weighted logits (bag-level from these instances) if needed
        logits = torch.sum(attribute_score, dim=0, keepdim=True) / torch.sum(
            torch.exp(A_expanded.squeeze(0)), dim=0, keepdim=True
        )  # [1, 2]

        return instance_loss, logits, p_targets

    
    def forward(
        self,
        h_5x,
        h_10x,
        h_20x,
        label=None,
        instance_eval=True
        attention_only =False
    ):
        """
        h_5x, h_10x, h_20x: [N_5, 512], [N_10, 512], [N_20, 512]
        label: scalar tensor with slide-level label
        instance_eval: whether to compute instance-level loss

        returns:
            if instance_eval:
                logits, total_inst_loss
            else:
                logits
        """

        h_5x = self.adaptor_5x(h_5x)   # [N_5, dim]
        h_10x = self.adaptor_10x(h_10x)  # [N_10, dim]
        h_20x = self.adaptor_20x(h_20x)  # [N_20, dim]

        # 2) Cross-magnification attention
        h_5x_10x = self.cross_attn_5x_10x(h_5x, h_10x)  
        h_10x_20x = self.cross_attn_10x_20x(h_10x, h_20x)
        h_20x_5x = self.cross_attn_20x_5x(h_20x, h_5x)

        if attention_only:
            # Return attention maps for the two main streams
            A_5x = self._get_attention(h_5x_10x)
            A_10x = self._get_attention(h_10x_20x)
	 		A_20x = self._get_attention(h_20x_5x)
            return A_5x, A_10x, A_20x

        # 3) Process each magnification stream with gated attention + attribute-weighted logits
        logits_5x, loss_5x = self._process_magnification(h_5x_10x, label, instance_eval)
        logits_10x, loss_10x = self._process_magnification(h_10x_20x, label, instance_eval)
        ogits_20x, loss_20x = self._process_magnification(h_20x_5x, label, instance_eval)

        # 4) Fuse logits from different magnifications
        weights = torch.softmax(self.logit_weights, dim=0)  
        logits = weights[0] * logits_5x + weights[1] * logits_10x + weights[2] * logits_20x
        logits = logits + self.bias  # [1, n_classes]

        if instance_eval:
            total_inst_loss = (loss_5x + loss_10x+loss_20x) / 3.0
            return logits, total_inst_loss
        else:
            return logits


    def _get_attention(self, h):
        """
        Returns attention map (after softmax over instances) for a given stream.
        """
        A, _ = self.attention_net(h)      # [N, 1]
        A = A.transpose(1, 0)             # [1, N]
        A = F.softmax(A, dim=1)           # [1, N]
        return A


    def _process_magnification(self, h, label, instance_eval):
        """
        h: [N, dim]
        label: scalar slide-level label
        """
        # 1) Attention over instances
        A, h_attn = self.attention_net(h)        # A: [N, 1], h_attn: [N, dim]
        A = F.softmax(A.transpose(1, 0), dim=1)  # [1, N]

        # 2) Per-instance class scores
        instance_score = self.classifier(h_attn).unsqueeze(0)  # [1, N, n_classes]

        # 3) Expand attention to all class dimensions
        A_expanded = A.unsqueeze(1).expand(-1, self.n_classes, -1).permute(0, 2, 1)  # [1, N, n_classes]

        # 4) Attribute-weighted scores and bag-level logits
        attribute_score = instance_score * torch.exp(A_expanded)             # [1, N, n_classes]
        logits = torch.sum(attribute_score, dim=1) / torch.sum(
            torch.exp(A_expanded), dim=1
        )  # [1, n_classes]

        total_inst_loss = 0.0

        if instance_eval and label is not None:
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # [n_classes]
            for i, classifier in enumerate(self.instance_classifiers):
                if inst_labels[i].item() == 1:
                    # In-the-class branch: rank by class-specific scores
                    class_score = attribute_score[0, :, i] 
                    loss, _, _ = self.inst_eval(class_score, A, h_attn, classifier, positive=True)
                elif self.subtyping:
                    # Out-of-the-class branch
                    class_score = attribute_score[0, :, i] 
                    loss, _, _ = self.inst_eval_out(class_score, A, h_attn, classifier, positive=True)
                else:
                    continue
                total_inst_loss += loss

        return logits, total_inst_loss


