import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminativeLoss(nn.Module):
    """
    Discriminative loss (De Brabandere et al. 2017).
    Pulls same-instance embeddings together, pushes different-instance
    embeddings apart, using hinged margins.

    delta_v = 0.5   pull margin  (penalise if pixel > delta_v from its mean)
    delta_d = 1.5   push margin  (penalise if two means < 2*delta_d apart)
    """
    def __init__(self, delta_v=0.5, delta_d=1.5):
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d

    def forward(self, embeddings, instance_labels, mask):
        """
        embeddings:      (B, 8, H, W)
        instance_labels: (B, H, W)  int, 0=background, 1=word1, 2=word2...
        mask:            (B, H, W)  binary, 1 where TCL pixels are

        Returns: loss_pull (scalar), loss_push (scalar)
        """
        B = embeddings.shape[0]
        total_pull, total_push, valid = 0.0, 0.0, 0

        for b in range(B):
            emb = embeddings[b]                         # (8, H, W)
            inst = instance_labels[b] * mask[b].long()  # zero non-TCL pixels
            ids = torch.unique(inst)
            ids = ids[ids != 0]
            K = len(ids)
            if K == 0:
                continue
            valid += 1
            means = []

            # Pull loss
            pull = 0.0
            for k in ids:
                px = (inst == k)                             # (H, W) bool
                e_k = emb[:, px]                             # (8, N_k)
                mu_k = e_k.mean(dim=1, keepdim=True)         # (8, 1)
                means.append(mu_k.squeeze())
                d = torch.norm(e_k - mu_k, dim=0)           # (N_k,)
                pull += torch.pow(torch.clamp(d - self.delta_v, min=0), 2).mean()
            total_pull += pull / K

            # Push loss
            if K > 1:
                push = 0.0
                M = torch.stack(means)   # (K, 8)
                n_pairs = 0
                for i in range(K):
                    for j in range(i + 1, K):
                        d = torch.norm(M[i] - M[j])
                        push += torch.pow(
                            torch.clamp(2.0 * self.delta_d - d, min=0), 2)
                        n_pairs += 1
                total_push += push / n_pairs

        if valid == 0:
            zero = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            return zero, zero

        return total_pull / valid, total_push / valid


class TextLoss(nn.Module):

    def __init__(self, delta_v=0.5, delta_d=1.5, lambda_embed=0.1):
        super().__init__()
        self.disc_loss = DiscriminativeLoss(delta_v=delta_v, delta_d=delta_d)
        self.lambda_embed = lambda_embed

    def ohem(self, predict, target, train_mask, negative_ratio=3.):
        pos = (target * train_mask).byte()
        neg = ((1 - target) * train_mask).byte()

        n_pos = pos.float().sum()

        if n_pos.item() > 0:
            loss_pos = F.cross_entropy(predict[pos], target[pos], reduction='sum')
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))
        else:
            loss_pos = 0.
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = 100
        loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    def forward(self, input, embedding, tr_mask, tcl_mask,
                sin_map, cos_map, radii_map, train_mask, instance_labels):
        """
        calculate textsnake loss + discriminative embedding loss
        :param input: (Variable), network predict, (BS, 7, H, W)
        :param embedding: (Variable), embedding output, (BS, 8, H, W)
        :param tr_mask: (Variable), TR target, (BS, H, W)
        :param tcl_mask: (Variable), TCL target, (BS, H, W)
        :param sin_map: (Variable), sin target, (BS, H, W)
        :param cos_map: (Variable), cos target, (BS, H, W)
        :param radii_map: (Variable), radius target, (BS, H, W)
        :param train_mask: (Variable), training mask, (BS, H, W)
        :param instance_labels: (Variable), instance ID map, (BS, H, W) int
        :return: total loss, loss dict
        """

        tr_pred = input[:, :2].permute(0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW, 2)
        tcl_pred = input[:, 2:4].permute(0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW, 2)

        # Handle potential NaNs/Infs in predictions for tr_loss and tcl_loss
        tr_pred = torch.nan_to_num(tr_pred, nan=0.0, posinf=1e5, neginf=-1e5)
        tcl_pred = torch.nan_to_num(tcl_pred, nan=0.0, posinf=1e5, neginf=-1e5)

        sin_pred = input[:, 4].contiguous().view(-1)  # (BSxHxW,)
        cos_pred = input[:, 5].contiguous().view(-1)  # (BSxHxW,)

        # Handle potential NaNs/Infs in predictions
        sin_pred = torch.nan_to_num(sin_pred, nan=0.0, posinf=1e5, neginf=-1e5)
        cos_pred = torch.nan_to_num(cos_pred, nan=0.0, posinf=1e5, neginf=-1e5)

        # regularize sin and cos: sum to 1
        scale = torch.sqrt(1.0 / (sin_pred ** 2 + cos_pred ** 2 + 1e-6))
        sin_pred = sin_pred * scale
        cos_pred = cos_pred * scale

        radii_pred = input[:, 6].contiguous().view(-1)  # (BSxHxW,)

        # Handle potential NaNs/Infs in ground truth masks
        train_mask = torch.nan_to_num(train_mask.view(-1), nan=0.0, posinf=1e5, neginf=-1e5)
        tr_mask = torch.nan_to_num(tr_mask.contiguous().view(-1), nan=0.0, posinf=1e5, neginf=-1e5)
        tcl_mask_flat = torch.nan_to_num(tcl_mask.contiguous().view(-1), nan=0.0, posinf=1e5, neginf=-1e5)

        radii_map = radii_map.contiguous().view(-1)

        # Handle potential NaNs/Infs in ground truth maps
        sin_map = torch.nan_to_num(sin_map.contiguous().view(-1), nan=0.0, posinf=1e5, neginf=-1e5)
        cos_map = torch.nan_to_num(cos_map.contiguous().view(-1), nan=0.0, posinf=1e5, neginf=-1e5)

        # --- existing 5 losses UNCHANGED ---
        loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())

        loss_tcl = 0.
        tr_train_mask = train_mask * tr_mask
        if tr_train_mask.sum().item() > 0:
            loss_tcl = F.cross_entropy(tcl_pred[tr_train_mask], tcl_mask_flat[tr_train_mask].long())

        # geometry losses
        loss_radii, loss_sin, loss_cos = 0., 0., 0.
        tcl_train_mask = train_mask * tcl_mask_flat
        if tcl_train_mask.sum().item() > 0:
            ones = radii_map.new(radii_pred[tcl_mask_flat].size()).fill_(1.).float()

            ##### these are the smoothed L1 losses as mentioned in the paper ####

            loss_radii = F.smooth_l1_loss(radii_pred[tcl_mask_flat] / (radii_map[tcl_mask_flat] + 1e-6), ones)
            loss_sin = F.smooth_l1_loss(sin_pred[tcl_mask_flat], sin_map[tcl_mask_flat])
            loss_cos = F.smooth_l1_loss(cos_pred[tcl_mask_flat], cos_map[tcl_mask_flat])

        # --- new embedding loss ---
        # tcl_mask is needed in 2D (B, H, W) for discriminative loss
        tcl_mask_2d = tcl_mask  # keep original 3D tensor — passed as arg before flattening
        loss_pull, loss_push = self.disc_loss(
            embedding, instance_labels, tcl_mask_2d)
        loss_embed = loss_pull + loss_push

        # total loss
        loss_total = loss_tr + loss_tcl + loss_radii + loss_sin + loss_cos + self.lambda_embed * loss_embed

        return loss_total, {
            'tr_loss': loss_tr,
            'tcl_loss': loss_tcl,
            'radii_loss': loss_radii,
            'sin_loss': loss_sin,
            'cos_loss': loss_cos,
            'pull_loss': loss_pull,
            'push_loss': loss_push,
            'embed_loss': loss_embed,
        }