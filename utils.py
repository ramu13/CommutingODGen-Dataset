def epoch_pass(loader, train=True):
    model.train() if train else model.eval()
    total_kl, total_obs = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            x, tgt, ori = flat_batch(batch, device)
            pred = model(x, ori)                         # (M,)
            loss = F.kl_div(pred.log(), tgt, reduction='batchmean')

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_kl  += loss.item() * tgt.shape[0]
            total_obs += tgt.shape[0]
    return total_kl / total_obs