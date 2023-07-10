import torch
from tqdm import tqdm
import numpy as np

def train(device,model,optim,crit,epoch_start,epoch_end,train_dl,test_dl,name):
    torch.cuda.empty_cache()
    pbar=tqdm(range(epoch_start,epoch_end))
    b_trn=10
    b_tst=10
    for epoch in pbar:
        model.train()
        train_loss=[]
        for data in train_dl:
            optim.zero_grad()
            if len(data)==5:
                md,fp,maccs,col,ris = data
                preds=model(md.to(device),fp.to(device),maccs.to(device),col.to(device))
            else:
                form,col,ris = data
                preds=model(form.to(device),col.to(device))
            loss=crit(preds.squeeze(),ris.to(device))
            loss.backward()
            optim.step()
            train_loss.append(loss.detach().cpu().mean())
        train_loss=np.average(train_loss)
        b_trn=min(b_trn,train_loss)

        model.eval()
        test_loss=[]
        for data in test_dl:
            with torch.no_grad():
                if len(data)==5:
                    md,fp,maccs,col,ris = data
                    preds=model(md.to(device),fp.to(device),maccs.to(device),col.to(device))
                else:
                    form,col,ris = data
                    preds=model(form.to(device),col.to(device))
                loss=crit(preds.squeeze(),ris.to(device))
                test_loss.append(loss.detach().cpu().mean())
        test_loss=np.average(test_loss)
        b_tst=min(b_tst,test_loss)

        pbar.set_postfix_str(f"{train_loss:.4f}/{b_trn:.4f}\t{test_loss:.4f}/{b_tst:.4f}")
        if test_loss<=b_tst:
            torch.save(model.state_dict(),f"{name}_model.pth")
            torch.save(optim.state_dict(),f"{name}_optim.pth")
    return b_trn,b_tst