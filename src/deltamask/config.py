"""Explicit experiment defaults, shared by the CLI and Flower applications."""
import math

def defaults():
    return {'num-rounds':2,'num-clients':2,'participation':1.,'local-epochs':1,
        'batch-size':16,'learning-rate':.1,'probe-learning-rate':.1,'kappa':.8,
        'seed':42,'dataset':'synthetic','model':'tiny','data-root':'data',
        'allow-download':False,'alpha':10.,'samples-per-client':32,
        'output':'results/smoke.json','device':'cpu','model-revision':'','model-checkpoint':''}

def validate(cfg):
    for key in ['num-rounds','num-clients','local-epochs','batch-size','samples-per-client']:
        if type(cfg[key])!=int or cfg[key]<1: raise ValueError(f'{key} must be a positive integer')
    for key in ['learning-rate','probe-learning-rate','alpha']:
        if not math.isfinite(cfg[key]) or cfg[key]<=0: raise ValueError(f'{key} must be positive')
    if not 0<cfg['participation']<=1 or not 0<=cfg['kappa']<=1: raise ValueError('Invalid participation or kappa')
    if cfg['dataset']=='synthetic' and cfg['model']!='tiny': raise ValueError('Synthetic vectors require tiny model')
    if cfg['dataset']!='synthetic' and cfg['model']=='tiny': raise ValueError('Image datasets require a vision backbone')
    if cfg['seed']<0 or cfg['seed']>=2**32: raise ValueError('seed must be a uint32')
    return cfg

def kappa_at(cfg,round_id):
    # Recovered scheduler horizon (R-1)//16, clamped at its endpoint instead of cycling.
    horizon=max(1,(cfg['num-rounds']-1)//16)
    fraction=min(1.,max(0.,(round_id-1)/horizon))
    return 1-(1-cfg['kappa'])*.5*(1+math.cos(math.pi*fraction))
