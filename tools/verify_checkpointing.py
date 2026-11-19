"""Compare full-model losses, features, every gradient, buffers, and RNG; no updates."""
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import torch
from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.env import seed_all_rng
from fastreid.utils.events import EventStorage


def main():
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    seed_all_rng(0)
    cfg = get_cfg()
    cfg.merge_from_file('configs/CARGO/visa_official_single_gpu.yml')
    cfg.MODEL.HEADS.NUM_CLASSES = 2500
    model = build_model(cfg).train()
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    images = torch.rand(8,3,256,128)*255
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'batch_size':8,'optimizer_steps':0,'synthetic_data_only':True,
            'checkpoint_regions':'11 encoder blocks including token subtraction, plus final transformer/LayerNorm',
            'cases':[]}
    captured=[]
    handle=model.backbone.register_forward_hook(
        lambda module,inputs,outputs:captured.extend(t.detach().cpu().clone() for t in outputs))
    try:
        for amp in (False,True):
            reference=None
            for enabled in (False,True):
                model.zero_grad(set_to_none=True)
                model.load_state_dict(initial)
                model.backbone.activation_checkpointing=enabled
                model.backbone.base.activation_checkpointing=enabled
                captured.clear()
                seed_all_rng(123)
                batch={'images':images.cuda().clone(),
                       'targets':torch.tensor([0]*4+[1]*4,device='cuda'),
                       'camids':torch.tensor([0,1,5,6]*2,device='cuda'),
                       'viewids':['Aerial','Aerial','Ground','Ground']*2}
                with EventStorage():
                    with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                        losses=model(batch)
                        total=sum(losses.values())
                    assert torch.isfinite(total)
                    total.backward()
                torch.cuda.synchronize()
                buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
                rng=(torch.get_rng_state(),torch.cuda.get_rng_state().cpu())
                if not enabled:
                    gradients={k:None if p.grad is None else p.grad.detach().cpu().clone()
                               for k,p in model.named_parameters()}
                    assert all(g is None or torch.isfinite(g).all() for g in gradients.values())
                    reference={'losses':{k:v.detach().cpu().clone() for k,v in losses.items()},
                               'features':list(captured),'gradients':gradients,'buffers':buffers,'rng':rng}
                    print(f'Reference amp={amp} completed',flush=True)
                else:
                    for name,value in losses.items():
                        torch.testing.assert_close(value.detach().cpu(),reference['losses'][name],rtol=1e-5,atol=1e-6)
                    for value,expected in zip(captured,reference['features']):
                        torch.testing.assert_close(value,expected,rtol=1e-5,atol=1e-6)
                    assert len(captured)==len(reference['features'])
                    max_abs=0.; grad_count=0; none_count=0; bitwise_equal=True
                    for name,param in model.named_parameters():
                        expected=reference['gradients'][name]
                        assert (param.grad is None)==(expected is None),name
                        if expected is None:
                            none_count+=1
                            continue
                        value=param.grad.detach().cpu()
                        assert torch.isfinite(value).all(),name
                        torch.testing.assert_close(value,expected,rtol=1e-4,atol=1e-5,msg=name)
                        max_abs=max(max_abs,(value-expected).abs().max().item())
                        bitwise_equal=bitwise_equal and torch.equal(value,expected)
                        grad_count+=1
                    assert buffers.keys()==reference['buffers'].keys()
                    for name,value in buffers.items():
                        assert torch.equal(value,reference['buffers'][name]),name
                    assert all(torch.equal(a,b) for a,b in zip(rng,reference['rng']))
                    assert all(torch.equal(p.detach().cpu(),initial[name]) for name,p in model.named_parameters())
                    report['cases'].append({'amp_fp16':amp,'losses_and_all_backbone_outputs':'passed',
                                            'gradient_tensors_compared':grad_count,'gradient_none_tensors':none_count,
                                            'max_gradient_absolute_difference':max_abs,
                                            'all_gradients_bitwise_equal':bitwise_equal,
                                            'gradient_rtol':1e-4,'gradient_atol':1e-5,
                                            'buffers_equal_including_BN':True,'cpu_and_cuda_rng_equal':True,
                                            'parameters_unchanged':True})
                    print(f'Checkpoint comparison amp={amp} passed; max gradient difference={max_abs}',flush=True)
                del losses,total,buffers,rng,batch
            del reference,gradients
            model.zero_grad(set_to_none=True)
            gc.collect()
        report['status']='passed'
    except Exception as error:
        import traceback
        traceback.print_exc()
        report.update(status='failed',error=str(error))
    finally:
        handle.remove()
        out=ROOT/'output/audit/checkpointing_correctness.json'
        out.parent.mkdir(exist_ok=True,parents=True)
        out.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    sys.exit(main())
