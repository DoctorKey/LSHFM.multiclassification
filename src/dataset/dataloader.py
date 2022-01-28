import torch

def create_train_loader(train_dataset, args):
    train_loader = torch.utils.data.DataLoader(train_dataset,
             batch_size=args.batch_size,
             shuffle=True,
             num_workers=args.workers,
             pin_memory=True,
             drop_last=True)
    return train_loader

def create_eval_loader(eval_dataset, args):
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)
    return eval_loader

