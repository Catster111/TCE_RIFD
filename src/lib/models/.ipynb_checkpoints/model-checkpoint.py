from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torchvision.models as models
import torch
#from torchsummaryX import summary
#from torchsummary import summary
import torch.nn as nn
import os
from .networks.msra_resnet import get_pose_net
# from .networks.dlav0 import get_pose_net as get_dlav0
# from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
# from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.large_hourglass import get_large_hourglass_net
# from .Backbone.mobilenetv2 import get_mobile_pose_netv2
# from .Backbone.mobilenet_v2 import get_mobile_net
# from .Backbone.centerface_mobilenet_v2 import get_mobile_net
from .Backbone.centerface_mobilenet_v2_fpn import get_mobile_net
from .Backbone.centerface_vit_fpn import get_vit_net
from .Backbone.centerface_hybrid_fpn import get_hybrid_net
from .Backbone.centerface_light_hybrid_fpn import get_lightweight_hybrid_net
from .Backbone.centerface_light_hybrid_fpn_pretrain import get_lightweight_hybrid_net_pretrain

_model_factory = {
  'res': get_pose_net,               # default Resnet with deconv
  # 'dlav0': get_dlav0,              # default DLAup
  # 'dla': get_dla_dcn,
  # 'resdcn': get_pose_net_dcn,
  'hourglass': get_large_hourglass_net,
  'mobilev2': get_mobile_net,
  'vit': get_vit_net,
  'hybrid': get_hybrid_net,  # Add this line# Vision Transformer implementation
  'light_hybrid': get_lightweight_hybrid_net,  # Add this line
  'light_hybrid_pretrain' : get_lightweight_hybrid_net_pretrain
}

def create_model(arch, heads, head_conv):
    """
    Create model based on architecture name
    """
    # Special handling for hybrid model which doesn't follow the num_layers pattern
    if arch == 'light_hybrid_pretrain':
        return get_lightweight_hybrid_net_pretrain(heads=heads, head_conv=head_conv)
        
    # For other models, extract num_layers if applicable
    try:
        num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    except ValueError:
        # If we can't convert to int, assume it's the base architecture without a number
        num_layers = 0
        
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    return model

def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
  """
  Load model weights from a checkpoint
  
  Args:
    model: Model to load weights into
    model_path: Path to the checkpoint file
    optimizer: Optimizer to load state (optional)
    resume: Whether to resume training
    lr: Learning rate (used when resuming)
    lr_step: Learning rate decay steps
  
  Returns:
    model: Model with loaded weights
    optimizer: Optimizer with loaded state (if provided)
    start_epoch: Epoch to start from
  """
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  start_epoch = checkpoint['epoch']
  print(lr)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # Convert data_parallel to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  
  model_state_dict = model.state_dict()
  
  # Check loaded parameters and created model parameters
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}.'.format(
          k, model_state_dict[k].shape, state_dict[k].shape))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k))
  
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k))
      state_dict[k] = model_state_dict[k]
  
  # Load state dict with flexible compatibility
  model.load_state_dict(state_dict, strict=False)
  
  # Resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  """
  Save model checkpoint
  
  Args:
    path: Path to save the checkpoint
    epoch: Current epoch number
    model: Model to save
    optimizer: Optimizer to save (optional)
  """
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  
  data = {
    'epoch': epoch,
    'state_dict': state_dict
  }
  
  if optimizer is not None:
    data['optimizer'] = optimizer.state_dict()
  
  torch.save(data, path)