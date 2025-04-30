from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
sys.path.append('/home/kunanon.k/Sakon_works/faceDetections/RPNetsPlus')

import torch
from lib.opts_pose import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.train_factory import train_factory
from lib.datasets.sample.multi_pose import Multiposebatch


def main(opt, qtepoch=[0, ]):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    
    # Get dataset and update options
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    # Setup logger
    logger = Logger(opt)

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    print('Using device:', opt.device)
    
    print('Creating model...')
    # Check if using Vision Transformer
    if opt.arch.startswith('vit'):
        print(f"Creating Vision Transformer model: {opt.arch}")
        # For ViT models, we might want to adjust some settings
        # ViT models generally need more memory, so reduce batch size if necessary
        if opt.batch_size > 16 and not hasattr(opt, 'batch_size_adjusted'):
            original_batch_size = opt.batch_size
            opt.batch_size = min(16, opt.batch_size)
            print(f"Adjusted batch size from {original_batch_size} to {opt.batch_size} for ViT model")
            opt.batch_size_adjusted = True  # Flag to avoid repeated adjustments
            
        # ViT models might benefit from different learning rate settings
        if not hasattr(opt, 'lr_adjusted') and not opt.resume:
            original_lr = opt.lr
            opt.lr = 1e-4  # Typical starting LR for transformer models
            print(f"Adjusted learning rate from {original_lr} to {opt.lr} for ViT model")
            opt.lr_adjusted = True  # Flag to avoid repeated adjustments
    
    # Create model
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    
    # Initialize optimizer
    # For ViT models, AdamW is often better than Adam
    if opt.arch.startswith('vit'):
        print("Using AdamW optimizer with weight decay for ViT")
        optimizer = torch.optim.AdamW(model.parameters(), 
                                     lr=opt.lr, 
                                     weight_decay=0.01)  # Weight decay is important for transformers
    else:
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    
    # Load model if specified
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
        print(f"Loaded model from {opt.load_model}, starting from epoch {start_epoch}")
        print(f"Current learning rate: {opt.lr}")

    # Initialize trainer
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')
    # Validation data loader
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # Test mode - run validation and exit
    if opt.test:
        print("Running in test mode...")
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return

    # Training data loader
    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=Multiposebatch
    )

    print('Starting training...')
    best = 1e10
    
    # For ViT models, we might want to implement a warmup schedule
    if opt.arch.startswith('vit') and not hasattr(opt, 'no_warmup'):
        warmup_epochs = 5
        base_lr = opt.lr
        print(f"Using {warmup_epochs} epochs of learning rate warmup for ViT")
        opt.no_warmup = False
    else:
        warmup_epochs = 0
    
    # Training loop
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        qtepoch.append(epoch)
        mark = epoch if opt.save_all else 'last'
        
        # Apply warmup for ViT models
        if opt.arch.startswith('vit') and epoch <= warmup_epochs and not opt.no_warmup:
            warmup_lr = base_lr * (epoch / warmup_epochs)
            print(f"Warmup epoch {epoch}/{warmup_epochs}, LR = {warmup_lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Train for one epoch
        log_dict_train, _ = trainer.train(epoch, train_loader)
        
        # Log training metrics
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        
        # Validation and model saving
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            # Save checkpoint
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                      epoch, model, optimizer)
            
            # Run validation
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            
            # Log validation metrics
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            
            # Save best model
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                print(f"New best {opt.metric}: {best}")
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                          epoch, model)
        else:
            # Save latest model
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                      epoch, model, optimizer)
        
        logger.write('\n')
        
        # Learning rate decay
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                      epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    opt.load_model = '../exp/multi_pose/Hathai_13_04_2025_version1/light_hybrid_pretrain/model_last.pth'
    opt.scale = 1
    
    # Add ViT-specific options if they don't exist
    # Using hasattr instead of directly setting to avoid AttributeError
    if not hasattr(opt, 'batch_size_adjusted'):
        opt.batch_size_adjusted = False
    if not hasattr(opt, 'lr_adjusted'):
        opt.lr_adjusted = False
    if not hasattr(opt, 'no_warmup'):
        opt.no_warmup = False
    
    main(opt)