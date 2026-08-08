import sys, torch, numpy as np
sys.path.insert(0,'/home/jetson/jjyoo3/EDCNO_DeepONet')
from src.data.mat_reader import read_mat_key
from src.data.datasets import ensure_nchw_last, _downsample_2d, _to_torch, _prepare_x_with_coords, _make_loader
from src.utils.normalizer import UnitGaussianNormalizer
from src.train.engine import evaluate_epoch
from src.utils.config import load_config
from src.utils.device import get_device

def norm_from(d):
    return UnitGaussianNormalizer(mean=d['mean'].clone(), std=d['std'].clone(), eps=float(d['eps']))

def go(tag, cfgpath, ckpt, builder, recorded):
    cfg=load_config(cfgpath); dev=get_device(None)
    ck=torch.load(ckpt,map_location='cpu',weights_only=False)
    xn=norm_from(ck['x_normalizer']); yn=norm_from(ck['y_normalizer'])
    res=tuple(cfg.data.resolution)
    X=_downsample_2d(ensure_nchw_last(read_mat_key('/home/jetson/data/piececonst_r421_N1024_smooth2.mat',cfg.data.input_key),spatial_dim=2),*res)
    Y=_downsample_2d(ensure_nchw_last(read_mat_key('/home/jetson/data/piececonst_r421_N1024_smooth2.mat',cfg.data.output_key),spatial_dim=2),*res)
    xt=_prepare_x_with_coords(xn.encode(_to_torch(X)),spatial_dim=2,add_coords=bool(cfg.data.add_coords))
    yt=yn.encode(_to_torch(Y))
    model=builder(cfg,xt.shape[-1],yt.shape[-1]).to(dev)
    model.load_state_dict(ck['model_state'])
    YN=yn.to(dev); bs=int(cfg.data.eval_batch_size)
    def run(a,b_):
        r=evaluate_epoch(model=model,loader=_make_loader(xt[a:b_],yt[a:b_],bs,False,0,False),
                         device=dev,y_normalizer=YN,amp_enabled=bool(cfg.train.amp))
        return r.rel_l2_mean,r.n_samples
    f=run(0,len(xt)); s=run(0,200)
    ok = abs(f[0]-recorded)<5e-7
    print("  %-9s full n=%d  %.8f  vs 기록 %.8f  %s" % (tag,f[1],f[0],recorded,"재현 ✓" if ok else "불일치 ✗"))
    print("  %-9s  200 n=%d  %.8f  %s" % ("",s[1],s[0],"" if ok else "(재현 실패로 무효)"))
    return ok,s[0]

from src.models.fno import build_fno_model
go("FNO","configs/resolution/darcy_fno_base_r141.yaml",
   "/home/jetson/jjyoo3/EDCNO/artifacts/checkpoints/darcy_fno_base_r141_seed0_best.pt",
   lambda c,i,o: build_fno_model(c,input_channels=i,output_channels=o), 0.015022320672869682)

from src.models.deeponet import build_deeponet_model
go("DeepONet","configs/deeponet/resolution/darcy_deeponet_base_r141.yaml",
   "/home/jetson/jjyoo3/EDCNO_DeepONet/artifacts/checkpoints/darcy_deeponet_base_r141_seed2_best.pt",
   lambda c,i,o: build_deeponet_model(c,input_channels=i,output_channels=o), 0.07332683354616165)
