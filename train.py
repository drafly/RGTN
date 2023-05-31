import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from tool.models.archs.restormer_mask.res_mask_de_detach import RestormerUfor_mask
from tool.models.losses.losses import PSNRLoss,CharbonnierLoss
from tool.models import lr_scheduler as lr_scheduler
import torch
import torch.nn.functional as F
from tool.utils.utils import parse_options,save_model,create_train_val_dataloader

from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True
opt = parse_options(opt_path='Options/basic.yml',is_train=True)

logdir= os.path.join(opt['log_dir'],opt['name'])
os.makedirs(logdir,exist_ok=True)

result = create_train_val_dataloader(opt)
train_loader, train_sampler, val_loader, total_epochs, total_iters = result

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda_count=torch.cuda.device_count()

net = RestormerUfor_mask(win_size=8).to(device)
#3net.load_state_dict(torch.load(opt['path']['pretrain_network_g'], map_location='cuda')['state_dict_net'])

train_opt = opt['train']
scheduler_type = train_opt['scheduler'].pop('type')
#########################        OPTIMIZER        #########################
optim_type = train_opt['optim_g'].pop('type')
if optim_type == 'Adam':
    optimizer = torch.optim.Adam(net.parameters(), **train_opt['optim_g'])
elif optim_type == 'AdamW':
    optimizer = torch.optim.AdamW(net.parameters(), **train_opt['optim_g'])
else:
    raise NotImplementedError(
        f'optimizer {optim_type} is not supperted yet.')


scheduler = lr_scheduler.CosineAnnealingRestartCyclicLR(optimizer, **train_opt['scheduler'])

train_loss = PSNRLoss().to(device)
val_loss=PSNRLoss().to(device)
mask_loss=CharbonnierLoss().to(device)


batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
gt_size = opt['datasets']['train'].get('gt_size')
iters_epoch=int(861/batch_size)

loss_min=0
current_iter=0
tensor_writer = SummaryWriter(logdir)
for epoch in range(0, 1200):
    avg_val_loss = 0
    step_val = 0
    if epoch > 1:
        scheduler.step()

    net.train()
    for iter, train_data in enumerate(train_loader, 1):
        current_iter+=1
        warmup_iter=opt['train'].get('warmup_iter', -1)

        if current_iter < warmup_iter:
            init_lr_g_l = []
            init_lr_g_l.append(  [v['initial_lr'] for v in optimizer.param_groups])

            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * current_iter for v in init_lr_g])

            for param_group, lr in zip(optimizer.param_groups, warm_up_lr_l):
                param_group['lr'] = lr

        optimizer.zero_grad()
        
        lq = train_data['lq'].to(device)
        gt = train_data['gt'].to(device)
        mask=1-abs(gt-lq).mean(dim=1).unsqueeze(dim=1)

        blur_masks = [
            F.interpolate(mask, scale_factor=0.25, mode='nearest'),
            F.interpolate(mask, scale_factor=0.5, mode='nearest'),
            mask,
        ]
        preds,masks = net(lq)

        total_loss=0

        loss_pred=train_loss(preds, gt)
        total_loss += loss_pred
        tensor_writer.add_scalar("loss on preds", loss_pred, iter+iters_epoch*epoch)
        for i_s, (gate_m, mask) in enumerate(zip(masks, blur_masks)):
            loss_mask=mask_loss(gate_m, mask)
            total_loss += loss_mask
            tensor_writer.add_scalar("loss on {} masks".format(i_s), loss_mask, iter+iters_epoch*epoch)
        total_loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        for idx, val_data in enumerate(val_loader):
            lq = val_data['lq'].to(device)
            gt = val_data['gt'].to(device)
            preds,_ = net(lq)
            mse_loss = val_loss(preds, gt)
            avg_val_loss += mse_loss
            step_val += 1
        avg_val_loss /= step_val
        tensor_writer.add_scalar('val_loss', avg_val_loss, epoch)

    if avg_val_loss<loss_min:
        loss_min=avg_val_loss
        save_model(net,epoch,avg_val_loss,logdir)
