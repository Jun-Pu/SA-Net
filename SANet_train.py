import os
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from data import get_loader, test_dataset, num_fs
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
from utils import print_network
from models.SANet import SANet
from utils import hybrid_e_loss


#set the device for training
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True

#build the model
model = SANet()
print_network(model, 'SANet')
if(opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load)
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

#set the path
image_root = opt.rgb_root
gt_root = opt.gt_root
depth_root = opt.depth_root
fs_root = opt.fs_root
test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
test_depth_root = opt.test_depth_root
test_fs_root = opt.test_fs_root
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
train_loader = get_loader(image_root, gt_root, depth_root, fs_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, test_depth_root, test_fs_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path+'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("SANet-Train")
logging.info("Config")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.
             format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.decay_rate,opt.load,save_path,opt.decay_epoch))

#set loss function
CE = torch.nn.BCEWithLogitsLoss()

step = 0
writer = SummaryWriter(save_path+'summary')
best_mae = 1
best_epoch = 0

#train function
def train(train_loader, model, optimizer, epoch,save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, depths, fss) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()
            for idx in range(num_fs):
                fss[idx] = fss[idx].unsqueeze(2).cuda()
            fss = torch.cat(fss, dim=2)
            pred, pL, pR = model(images, fss, depths)
            loss1, loss2, loss3 = hybrid_e_loss(pred, gts), hybrid_e_loss(pL, gts), hybrid_e_loss(pR, gts)
            loss = loss1 + loss2 + loss3
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                    format( epoch, opt.epoch, i, total_step, loss.data))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = pred[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('pred', torch.tensor(res), step,dataformats='HW')
        
        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path+'SANet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'SANet_epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise
        
#test function
def test(test_loader,model,epoch,save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, fss, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            for idx in range(num_fs):
                fss[idx] = fss[idx].unsqueeze(2).cuda()
            fss = torch.cat(fss, dim=2)
            res, _, _ = model(image, fss, depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])
        mae = mae_sum/test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path+'SANet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))
 
if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
