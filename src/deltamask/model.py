"""Frozen pretrained weights with trainable stochastic masks.

Adapted from the recovered DeltaMask model/vit.py and network.py. The Bernoulli
straight-through estimator is explicitly identity in probability space; unlike
the recovered extra probability multiplier, sigmoid supplies its own derivative.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

class Bernoulli(torch.autograd.Function):
    @staticmethod
    def forward(ctx,p): return torch.bernoulli(p)
    @staticmethod
    def backward(ctx,gradient): return gradient

class MaskedLayer(nn.Module):
    def __init__(self,layer):
        super().__init__();self.layer=layer
        for p in layer.parameters(): p.requires_grad_(False)
        self.weight_mask=nn.Parameter(torch.zeros_like(layer.weight))
        if layer.bias is not None: self.bias_mask=nn.Parameter(torch.zeros_like(layer.bias))
        else: self.register_parameter('bias_mask',None)
    def forward(self,x):
        weight=self.layer.weight*Bernoulli.apply(self.weight_mask.sigmoid())
        bias=None if self.bias_mask is None else self.layer.bias*Bernoulli.apply(self.bias_mask.sigmoid())
        if isinstance(self.layer,nn.Conv2d): return self.layer._conv_forward(x,weight,bias)
        return nn.functional.linear(x,weight,bias)

def mask_children(module):
    for name,child in list(module.named_children()):
        if isinstance(child,(nn.Linear,nn.Conv2d)): setattr(module,name,MaskedLayer(child))
        elif not isinstance(child,MaskedLayer): mask_children(child)

class Model(nn.Module):
    def __init__(self,backbone,features,classes,blocks,kind):
        super().__init__();self.backbone=backbone;self.head=nn.Linear(features,classes)
        self.blocks=blocks;self.kind=kind
        for p in backbone.parameters():p.requires_grad_(False)
    def forward(self,x):
        if self.kind=='clip': x=self.backbone(pixel_values=x).image_embeds
        elif self.kind=='dino': x=self.backbone(pixel_values=x).last_hidden_state[:,0]
        else:x=self.backbone(x)
        return self.head(x)
    def enable_masks(self):
        if isinstance(self.head,MaskedLayer):return
        for block in self.blocks: mask_children(block)
        # CLIP projection and classifier were also masked in the recovered code.
        if self.kind=='clip':self.backbone.visual_projection=MaskedLayer(self.backbone.visual_projection)
        self.head=MaskedLayer(self.head)
        for n,p in self.named_parameters():p.requires_grad_(n.endswith('_mask'))

def make_model(cfg,classes):
    revisions={'clip-vit-b32':'3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268','clip-vit-l14':'32bd64288804d66eefd0ccbe215aa642df71cc41','dinov2-base':'f9e44c814b77203eaa57a6bdbbd535f21ede1415','dinov2-small':'ed25f3a31f01632728cabb09d1542f84ab7b0056'}
    revision=cfg['model-revision'] or revisions.get(cfg['model'])
    with torch.random.fork_rng():
        torch.manual_seed(cfg['seed'])
        name=cfg['model']
        if name=='tiny':
            # Offline correctness fixture, deliberately not a pretrained foundation model.
            backbone=nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,12),nn.ReLU())
            model=Model(backbone,12,classes,[backbone],'tiny')
        elif name.startswith('clip'):
            from transformers import CLIPVisionModelWithProjection
            identifier={'clip-vit-b32':'openai/clip-vit-base-patch32','clip-vit-l14':'openai/clip-vit-large-patch14'}[name]
            backbone=CLIPVisionModelWithProjection.from_pretrained(identifier,revision=revision,local_files_only=not cfg['allow-download'])
            model=Model(backbone,backbone.config.projection_dim,classes,list(backbone.vision_model.encoder.layers)[-5:],'clip')
        elif name.startswith('dinov2'):
            from transformers import AutoModel
            identifier={'dinov2-base':'facebook/dinov2-base','dinov2-small':'facebook/dinov2-small'}[name]
            backbone=AutoModel.from_pretrained(identifier,revision=revision,local_files_only=not cfg['allow-download'])
            model=Model(backbone,backbone.config.hidden_size,classes,list(backbone.encoder.layer)[-5:],'dino')
        elif name=='convmixer-768-32':
            import timm
            checkpoint=cfg.get('model-checkpoint','')
            if not cfg['allow-download'] and not checkpoint:
                raise ValueError('ConvMixer needs model-checkpoint or allow-download=true')
            backbone=timm.create_model('convmixer_768_32.in1k',pretrained=not bool(checkpoint),checkpoint_path=checkpoint)
            features=backbone.num_features;backbone.reset_classifier(0)
            model=Model(backbone,features,classes,list(backbone.blocks)[-5:],'convmixer')
        else:raise ValueError(f'Unsupported model: {name}')
    return model.to(cfg.get('device','cpu'))

def mask_scores(model):
    return np.concatenate([p.detach().cpu().numpy().ravel() for n,p in model.named_parameters() if n.endswith('_mask')])

def set_mask_scores(model,values):
    values=np.asarray(values);offset=0
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n.endswith('_mask'):
                count=p.numel();p.copy_(torch.as_tensor(values[offset:offset+count].reshape(p.shape),device=p.device));offset+=count
    if offset!=values.size:raise ValueError('Mask layout mismatch')

def _fit(model,ds,cfg,seed,head_only):
    torch.manual_seed(seed);model.eval() # Freeze backbone dropout/batchnorm; masks still sample.
    params=list(model.head.parameters()) if head_only else [p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(params,lr=cfg['probe-learning-rate' if head_only else 'learning-rate'])
    loader=DataLoader(ds,batch_size=cfg['batch-size'],shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
    losses=[];device=next(model.parameters()).device
    for _ in range(1 if head_only else cfg['local-epochs']):
        for x,y in loader:
            optimizer.zero_grad();loss=nn.functional.cross_entropy(model(x.to(device)),y.to(device));loss.backward();optimizer.step();losses.append(loss.item())
    return float(np.mean(losses))

def probe(model,ds,cfg,seed):return _fit(model,ds,cfg,seed,True)
def train_local(model,ds,cfg,seed):return _fit(model,ds,cfg,seed,False)

def evaluate(model,ds,cfg,seed):
    model.eval();torch.manual_seed(seed);correct=0;loss=0.;device=next(model.parameters()).device
    with torch.no_grad():
        for x,y in DataLoader(ds,batch_size=cfg['batch-size']):
            y=y.to(device);pred=model(x.to(device));correct+=int((pred.argmax(1)==y).sum());loss+=nn.functional.cross_entropy(pred,y,reduction='sum').item()
    return {'accuracy':correct/len(ds),'loss':loss/len(ds)}
