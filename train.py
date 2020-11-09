import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import os
import argparse
import yaml
from tqdm import tqdm

from data_loader import get_dataloader
from utils import prepare_dir, clean_ckpts, logger, AverageMeter
from models.decoder import AttentionDecoder
from models.encoder import CNN


class Trainer(object):
    def __init__(self, config_file: str):
        self.cfg = yaml.load(open(config_file), Loader=yaml.FullLoader)
        self.teach_forcing_prob = self.cfg['trainer']['teach_forcing_prob']

        # Set random seed
        np.random.seed(self.cfg['trainer']['seed'])
        random.seed(self.cfg['trainer']['seed'])
        torch.manual_seed(self.cfg['trainer']['seed'])
        torch.cuda.manual_seed_all(self.cfg['trainer']['seed'])

        dist.init_process_group(backend='nccl', init_method='env://')
        self.local_rank = torch.distributed.get_rank()

        self.encoder = CNN(**self.cfg['arch']['encoder'])
        self.decoder = AttentionDecoder(**self.cfg['arch']['decoder'])
        
        self.encoder = self.encoder.cuda(self.local_rank)
        self.decoder = self.decoder.cuda(self.local_rank)
        self.num_hidden = self.decoder.num_hidden

        process_group = torch.distributed.new_group(list(range(torch.cuda.device_count())))
        self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder, process_group)
        self.decoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.decoder, process_group)

        # broadcast_buffers==True if use sync bn
        self.encoder = DDP(self.encoder,
                           device_ids=[self.local_rank],
                           output_device=self.local_rank,
                           find_unused_parameters=True,
                           broadcast_buffers=True)
        self.decoder = DDP(self.decoder,
                           device_ids=[self.local_rank],
                           output_device=self.local_rank,
                           find_unused_parameters=True,
                           broadcast_buffers=True)

        self.criterion = torch.nn.NLLLoss()
        self.criterion = self.criterion.cuda(self.local_rank)

        self.params = [
            {'params': self.encoder.module.parameters()},
            {'params': self.decoder.module.parameters()}
        ]
        self.optimizer = getattr(torch.optim, self.cfg['optimizer']['type'])(
            self.params, **self.cfg['optimizer']['args']
        )

        # Load model from previous checkpoint
        self.step, self.epoch = 0, 0
        self.ckpt_path = self.cfg['trainer']['outputs'] + 'checkpoints'
        if self.cfg['trainer']['restore']:
            ckpt = os.path.join(self.ckpt_path, sorted(os.listdir(self.ckpt_path))[-1])
            ckpt = torch.load(ckpt)
            self.encoder.module.load_state_dict(ckpt['encoder'], strict=True)
            self.decoder.module.load_state_dict(ckpt['decoder'], strict=True)
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.step = ckpt['step']
            self.epoch = ckpt['epoch'] + 1
            #self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            if self.local_rank == 0:
                logger.info(f'Contatinue training from step {self.step} and epoch {self.epoch}.')
        elif self.local_rank == 0:
            prepare_dir(self.ckpt_path)
            logger.info(f"{self.ckpt_path} was cleaned.")
        
        if self.local_rank == 0:
            # Create tensorboard summary writer
            summary_dir = self.cfg['trainer']['outputs'] + 'summary'
            if not self.cfg['trainer']['restore']:
                prepare_dir(summary_dir)
                logger.info(f"{summary_dir} was cleaned.")
            self.writer = SummaryWriter(summary_dir)

            #add graph
            if self.cfg['trainer']['tensorboard']:
                self.writer.add_graph(self.encoder.module, torch.zeros(1, 1, 32, 224).cuda(self.local_rank))
                torch.cuda.empty_cache()
        
        # Create Dataloader
        self.train_data_loader, self.sampler = get_dataloader(self.cfg['dataset']['train'], distributed=True)
        # Wether use validate or not
        self.val_enable = 'val' in self.cfg['dataset']
        if self.val_enable and self.local_rank == 0:
            self.val_data_loader, _ = get_dataloader(self.cfg['dataset']['val'], distributed=False)
        
        if self.local_rank == 0:
            with open(self.cfg['trainer']['outputs'] + 'debug_configs.yaml', 'w') as f:
                yaml.dump(self.cfg, f, indent=1)
            
            self.train_loss = AverageMeter()

    def train(self):
        while self.epoch < self.cfg['trainer']['epochs']:
            self._train_step()
            if self.local_rank == 0 and self.val_enable:
                self._val_step()
            if self.local_rank == 0:
                self._save_network()
            self.epoch += 1

    def _train_step(self):
        if self.local_rank == 0:
            logger.info('----------------------------------------------------------')
            logger.info(f'Training for epoch {self.epoch} ...')
        torch.backends.cudnn.benchmark = self.cfg['trainer']['use_benchmark']
        self.sampler.set_epoch(self.epoch)
        self.encoder.train()
        self.decoder.train()
        for images, labels in self.train_data_loader:
            self.step += 1
            self.optimizer.zero_grad()
            images = images.cuda(self.local_rank)
            labels = labels.cuda(self.local_rank)

            # 教师强制：将目标label作为下一个输入
            teach_forcing = random.random() < self.teach_forcing_prob  # use teach forcing
            preds = self._step(images, teach_forcing, labels)
            loss = 0.
            for pred, label in zip(preds, labels[1:, :]):
                loss += self.criterion(pred, label)

            # backpropagation
            loss.backward()            
            torch.nn.utils.clip_grad_norm_(self.decoder.module.parameters(), self.cfg['trainer']['clip_grad_norm'])
            torch.nn.utils.clip_grad_norm_(self.encoder.module.parameters(), self.cfg['trainer']['clip_grad_norm'])
            self.optimizer.step()

            if self.local_rank == 0:
                self.train_loss.update(loss.item())
                if self.step % self.cfg['trainer']['log_iter'] == 0:
                    logger.info(f'step: {self.step}, loss: {self.train_loss.avg:.6f}')

                    if self.cfg['trainer']['tensorboard']:
                        self.writer.add_scalar('Train/loss', self.train_loss.avg, self.step)
                        self.writer.flush()

                    self.train_loss.reset()
        torch.cuda.empty_cache()
    
    def _val_step(self):
        logger.info('**********************************************************')
        logger.info(f'Validating for epoch {self.epoch} ...')
        torch.backends.cudnn.benchmark = False  # Since image shape is different
        self.encoder.eval()
        self.decoder.eval()
        val_loss = AverageMeter()
        num_correct, num_wrong = 0, 0
        with torch.no_grad():
            for images_cpu, labels_cpu in tqdm(self.val_data_loader):
                images = images_cpu.cuda(self.local_rank)
                labels = labels_cpu.cuda(self.local_rank)
                loss = 0.
                pred_labels = []
                # 预测的时候采用非强制策略，将前一次的输出，作为下一次的输入，直到标签为EOS_TOKEN时停止
                preds = self._step(images, False, labels)
                for pred, label in zip(preds, labels[1:]):
                    loss += self.criterion(pred, label)
                    pred_labels.append(pred.cpu().numpy())

                val_loss.update(loss.item())
                pred_labels = np.array(pred_labels)  # (length, N)
                gt_labels = labels_cpu.numpy()[1:]
                num_correct += np.sum(pred_labels == gt_labels)
                num_wrong += np.sum(pred_labels != gt_labels)

        torch.cuda.empty_cache()      
        logger.info(f'epoch: {self.epoch}, loss: {val_loss.avg:.6f}, '
                    f'accuracy: {num_correct / (num_correct + num_wrong):.4f}')

    def _step(self, images, teach_forcing, labels):
        length, bsize = labels.size()
        encoder_out = self.encoder(images)  # (56, 4, 256)
        
        decoder_hidden = torch.zeros(1, bsize, self.num_hidden, device=f'cuda:{self.local_rank}')
        outputs = []
        if teach_forcing:
            for decoder_in in labels[:-1]:  # decoder_in (N,)
                decoder_out, decoder_hidden, decoder_attn = self.decoder(
                    decoder_in, decoder_hidden, encoder_out
                )
                outputs.append(decoder_out)  # (N, 39)
        else:
            decoder_in = labels[0]
            for _ in range(1, length):
                decoder_out, decoder_hidden, decoder_attn = self.decoder(
                    decoder_in, decoder_hidden, encoder_out
                )
                decoder_in = torch.argmax(decoder_out, dim=1)
                outputs.append(decoder_out)
        return outputs

    def _save_network(self):
        state = {
            'encoder': self.encoder.module.state_dict(),
            'decoder': self.decoder.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'epoch': self.epoch
        }
        torch.save(state, self.ckpt_path + f'/model_epoch_{self.epoch}_step_{self.step}.pth')
        clean_ckpts(self.ckpt_path, num_max=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aocr')
    parser.add_argument('-c', '--config_file', default='config.yaml', type=str)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    trainer = Trainer(args.config_file)
    trainer.train()
