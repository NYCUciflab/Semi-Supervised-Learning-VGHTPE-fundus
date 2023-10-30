# Code below are implementation of FixMatch and CoMatch

import torch


# CoMatch
def train_comatch(model, loss_label, metric, optimizer,
                  label_loader, unlabel_loader, threshold, num_class,
                  train_on_gpu=True):

    """
    citation:https://arxiv.org/pdf/2011.11183.pdf.
    Args:
        model (nn.Module).
        loss_label (callable):
            An objective function is any callable with
            loss = fn(output, label),
            where output denotes model's prediction,
            label denote ground truth values.
        metirc (callable):
            An objective function is any callable with
            result = fn(output, label),
            where output denotes model's prediction,
            label denote ground truth values.
        optimizer (torch.optim).
        label_loader (torch.utils.data.DataLoader):
            Dataloader for labeled data.
        unlabel_loader (torch.utils.data.DataLoader):
            Dataloader for unlabeled data.
        threshold (float): Threshold of Pseudo-Label.
        num_class (integer): number of class to be classify.
        train_on_gpu (boolean):
            train on gpu or not.
    """

    iterator = tqdm(label_loader)

    model.train()

    train_acc = 0
    labeled_loss = 0
    unlabeled_loss = 0
    train_loss = 0
    pseudolabel_percentage = 0
    queue_size =  160
    queue_ptr = 0
    probs_avg = 0
    queue_feats = torch.zeros(queue_size, 64).cuda()
    queue_probs = torch.zeros(queue_size, num_class).cuda()
    for batch_idx, batch in enumerate(
    zip(iterator, unlabel_loader)
    ):
        labeled_batch, unlabeled_batch = batch

        labeled_imgs, labels = labeled_batch
        labeled_imgs, labels = labeled_imgs.cuda(), labels.cuda()
        (unlabeled_w, unlabeled_s1, unlabeled_s2), _ = unlabeled_batch
        unlabeled_w, unlabeled_s1, unlabeled_s2 = unlabeled_w.cuda(), unlabeled_s1.cuda(), unlabeled_s2.cuda()


        optimizer.zero_grad()

        #Supervised Part
        labeled_logits, labeled_feature = model(labeled_imgs)
        Lx = loss_label(labeled_logits, labels)
        score = metric(labeled_logits, labels)
        train_acc += score

        #Semi-Supervised Part
        bt = labeled_imgs.size(0) # batch size of label data
        btu = unlabeled_w.size(0)# batch size of unlabel data

        with torch.no_grad():
            uw_logits, uw_feature = model(unlabeled_w)
            uw_logits, uw_feature, labeled_feature = uw_logits.detach(), uw_feature.detach(), labeled_feature.detach()

            probs = torch.softmax(uw_logits, dim=1)

            probs_orig = probs.clone()

            A = torch.exp(torch.mm(uw_feature, queue_feats.t())/0.2)
            A = A/A.sum(1,keepdim=True)
            probs = 0.9*probs + (1-0.9)*torch.mm(A, queue_probs)

            scores, labeled_u = torch.max(probs, dim=1)
            mask = scores.ge(threshold).float()

            feats_w = torch.cat([uw_feature,labeled_feature], dim=0)
            onehot_encode = torch.zeros(bt, num_class).cuda().scatter(1, labels.view(-1, 1), 1)
            probs_w = torch.cat([probs_orig, onehot_encode], dim=0)

            # updata memory bank
            n = bt + btu
            queue_feats[queue_ptr:queue_ptr + n,:] = feats_w
            queue_probs[queue_ptr:queue_ptr + n,:] = probs_w
            queue_ptr = (queue_ptr+n)%160

        pseudolabel_percentage+=(mask.mean().item())
        # Graph Similarity
        us1_logits, us1_feature = model(unlabeled_s1)
        us2_logits, us2_feature = model(unlabeled_s2)
        sim = torch.exp(torch.mm(us1_feature, us2_feature.t())/0.1)
        sim_probs = sim / sim.sum(1, keepdim=True)

        # pseudo-label graph with self-loop
        Q = torch.mm(probs, probs.t())
        Q.fill_diagonal_(1)
        pos_mask = (Q>=0.8).float()

        Q = Q * pos_mask
        Q = Q / Q.sum(1, keepdim=True)


        # Contrastive Loss
        loss_contrast = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
        loss_contrast = loss_contrast.mean()

        # Pseudo-Label Loss

        loss_pl = (loss_label(us1_logits, labeled_u) * mask).mean()

        loss = Lx + 0.5*loss_pl + 0.5*loss_contrast
        train_loss += loss.item()*bt
        loss.backward()

        iterator.set_description("train")
        iterator.set_postfix({'Label loss':'%.4f'%Lx.item(), 'Pseudo Label loss':'%.4f'%loss_pl.item(), 'Contrastive loss':'%.4f'%loss_contrast.item(),metric.__name__:'%.4f'%score})

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2) # prevent gradient explode/vanish
        optimizer.step()

    return train_acc/len(iterator.dataset()), labeled_loss/len(iterator.dataset()), unlabeled_loss, pseudolabel_percentage*100/len(iterator.dataset())


#FixMatch
def train_fixmatch(model, loss_label, metric, optimizer,
                  label_loader, unlabel_loader, threshold,
                  train_on_gpu=True):

    """
    citation:https://arxiv.org/abs/2001.07685.
    Args:
        model (nn.Module).
        loss_label (callable):
            An objective function is any callable with
            loss = fn(output, label),
            where output denotes model's prediction,
            label denote ground truth values.
        metirc (callable):
            An objective function is any callable with
            result = fn(output, label),
            where output denotes model's prediction,
            label denote ground truth values.
        optimizer (torch.optim).
        target_batch (int):
            Number of target batch for one training epoch.
        label_loader (torch.utils.data.DataLoader):
            Dataloader for labeled data.
        unlabel_loader (torch.utils.data.DataLoader):
            Dataloader for unlabeled data.
        num_class (integer): number of class to be classify.
        train_on_gpu (boolean):
            train on gpu or not.
    """

    model.train()

    iterator = tqdm(label_loader)

    train_acc = 0
    labeled_loss = 0
    unlabeled_loss = 0
    pseudolabel_percentage = 0


    for batch_idx, batch in enumerate(
    zip(iterator, unlabel_loader)
    ):
        labeled_batch, unlabeled_batch = batch

        labeled_imgs, labels = labeled_batch
        labeled_imgs, labels = labeled_imgs.cuda(), labels.cuda()
        (unlabeled_w, unlabeled_s), _ = unlabeled_batch
        unlabeled_w, unlabeled_s = unlabeled_w.cuda(), unlabeled_s.cuda()


        optimizer.zero_grad()


        labeled_logits = model(labeled_imgs)
        Lx = loss_label(labeled_logits, labels)
        #Lx.backward()

        score = metric(labeled_logits, labels)
        train_acc += score

        #Generate Pseudo-Label and Calculate how many image are used as pseudo-label
        with torch.no_grad():
            logits_uw = model(unlabeled_w)
            pseudo_labels = torch.softmax(logits_uw, dim=1)
            max_probs, labels_u = torch.max(pseudo_labels, dim=1)
            mask = max_probs.ge(threshold).float()

        pseudolabel_percentage+=(mask.mean().item())

        logits_us = model(unlabeled_s)

        Lu = (loss_label(logits_us, labels_u) * mask).mean()

        #Lu.backward()
        loss = Lx + 0.5 * Lu
        loss.backward()

        labeled_loss += Lx.item()
        unlabeled_loss += Lu.item()


        iterator.set_description("train")
        iterator.set_postfix({'Label loss':'%.4f'%Lx.item(), 'Unlabel loss':'%.4f'%Lu.item(), metric.__name__:'%.4f'%score})

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2) # prevent gradient explode/vanish
        optimizer.step()

    return train_acc/len(iterator.dataset()), labeled_loss/len(iterator.dataset()), unlabeled_loss, pseudolabel_percentage*100/len(iterator.dataset())

