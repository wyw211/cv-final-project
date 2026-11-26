"""Isolated 50-step native training test, checkpoint restore, no test-set evaluation."""
from datetime import datetime
from zoneinfo import ZoneInfo
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import statistics
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, hooks
from fastreid.utils.env import seed_all_rng
from fastreid.utils.events import EventStorage


class RNGState:
    def state_dict(self):
        state=np.random.get_state()
        return {'python':random.getstate(),'cpu':torch.get_rng_state(),
                'cuda':torch.cuda.get_rng_state_all(),
                'numpy':(state[0],torch.from_numpy(state[1].astype(np.int64)),*state[2:])}

    def load_state_dict(self,state):
        random.setstate(state['python'])
        torch.set_rng_state(state['cpu'])
        torch.cuda.set_rng_state_all(state['cuda'])
        n=state['numpy']
        np.random.set_state((n[0],n[1].numpy().astype(np.uint32),*n[2:]))


class ShortTrainer(DefaultTrainer):
    def build_hooks(self):
        # Keep the real warmup/cosine logic, but never create test/evaluation hooks.
        return [hooks.LRScheduler(self.optimizer,self.scheduler)]

    @classmethod
    def build_model(cls,cfg):
        # Avoid dumping the full model to the console.
        from fastreid.modeling import build_model
        return build_model(cfg)


def digest_state(value):
    digest=hashlib.sha256()
    def add(x):
        if isinstance(x,torch.Tensor):
            digest.update(str((x.dtype,tuple(x.shape))).encode())
            digest.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):
                digest.update(str(k).encode());add(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:add(v)
        else:digest.update(repr(x).encode())
    add(value)
    return digest.hexdigest()


def states(trainer):
    return {'model':digest_state(trainer.model.state_dict()),
            **{name:digest_state(obj.state_dict())
               for name,obj in trainer.checkpointer.checkpointables.items()}}


def close_trainer(trainer):
    trainer.data_loader.shutdown()
    # DataLoaderX keeps a reference to its BackgroundGenerator and prefetched batch.
    trainer._trainer._data_loader_iter=None
    trainer.model.zero_grad(set_to_none=True)


def summarize(report):
    """Validate AMP-aware progress, including legitimate overflow-skipped steps."""
    rows=report['rows']
    assert len(rows)==50
    assert all(np.isfinite(row['total_loss']) for row in rows)
    skipped=[r for r in rows if not r['head_parameters_updated']]
    assert all(r['amp_scale_after']<r['amp_scale_before'] for r in skipped), 'Unexplained missing update'
    assert all(r['head_parameters_updated'] for r in rows[-5:]), 'Final updates are not stable'
    assert report['checkpoint_restore']['all_state_hashes_equal']
    measured=[r['seconds'] for r in rows if 11<=r['iteration']<=40 or 44<=r['iteration']<=50]
    average=statistics.mean(measured)
    return dict(status='passed',successful_updates=sum(r['head_parameters_updated'] for r in rows),
                skipped_updates=len(skipped),amp_overflow_skip_iterations=[r['iteration'] for r in skipped],
                validation='All losses finite; every skipped update lowers AMP scale; last 5 updates succeed; restored states match',
                timed_iterations=len(measured),mean_iteration_seconds=average,
                median_iteration_seconds=statistics.median(measured),
                p95_iteration_seconds=float(np.percentile(measured,95)),
                projected_24000_iterations_hours=average*24000/3600,
                projection_excludes='final evaluations, checkpoint I/O, scheduling variability; early warmup sample only',
                peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in rows))


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-file',default='configs/CARGO/visa_official_checkpointing.yml')
    args=parser.parse_args()
    stamp=datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y%m%d_%H%M%S')
    output=ROOT/'logs'/f'bringup_{stamp}_{uuid.uuid4().hex[:6]}'
    output.mkdir(parents=True,exist_ok=False)
    report={'purpose':'isolated short training, not a baseline result',
            'output_dir':str(output),'iterations_requested':50,'test_evaluations':0,
            'dataset':'CARGO train only','batch_size':256,'loader_workers':8,
            'activation_checkpointing':True,'warmup_iterations':2000,
            'max_epoch_setting':120,'config_file':args.config_file,'rows':[]}
    (ROOT/'output/audit/latest_short_training.txt').write_text(str(output),encoding='utf-8')
    trainer=None
    try:
        torch.set_num_threads(8)
        torch.cuda.set_per_process_memory_fraction(1.0,0)
        torch.backends.cudnn.benchmark=True
        torch.cuda.reset_peak_memory_stats()
        report.update(gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),cuda=torch.version.cuda)
        cfg=get_cfg();cfg.merge_from_file(args.config_file)
        cfg.OUTPUT_DIR=str(output)
        assert cfg.SOLVER.IMS_PER_BATCH==256 and cfg.DATALOADER.NUM_WORKERS==8
        seed_all_rng(0)
        trainer=ShortTrainer(cfg)
        trainer.checkpointer.checkpointables['rng_state']=RNGState()
        (output/'config.yml').write_text(trainer.cfg.dump(),encoding='utf-8')
        initial_head=trainer.model.heads.weight.detach().cpu().clone()
        checkpoint_path=output/'short_iteration_40.pth'
        checkpoint_hashes=None
        started=time.perf_counter()
        for start,end in [(0,40),(40,50)]:
            with EventStorage(start) as storage:
                trainer.storage=storage
                trainer.epoch=0
                storage.epoch=0
                for iteration in range(start,end):
                    trainer.iter=iteration
                    head=trainer.model.heads.weight.detach().clone()
                    scale_before=trainer.grad_scaler.get_scale()
                    lr_before=trainer.optimizer.param_groups[0]['lr']
                    torch.cuda.synchronize()
                    tick=time.perf_counter()
                    trainer.before_step()
                    trainer.run_step()
                    trainer.after_step()
                    torch.cuda.synchronize()
                    duration=time.perf_counter()-tick
                    head_changed=not torch.equal(head,trainer.model.heads.weight.detach())
                    latest=storage.latest()
                    metrics={k:float(v[0]) for k,v in latest.items()
                             if k.startswith('loss_') or k in ('total_loss','load_balance_loss','data_time')}
                    assert np.isfinite(metrics['total_loss'])
                    row={'iteration':iteration+1,'seconds':duration,
                         'head_parameters_updated':head_changed,'lr_used':lr_before,
                         'amp_scale_before':scale_before,'amp_scale_after':trainer.grad_scaler.get_scale(),
                         'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,**metrics}
                    row['peak_reserved_GiB']=torch.cuda.max_memory_reserved()/1024**3
                    report['rows'].append(row)
                    with (output/'iterations.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(row)+'\n')
                    if (iteration+1)%5==0:
                        print(json.dumps(row),flush=True)
                    del head
            if end==40:
                assert not torch.equal(initial_head,trainer.model.heads.weight.detach().cpu())
                print('Saving iteration-40 checkpoint (model/optimizer/schedulers/AMP/RNG)',flush=True)
                save_started=time.perf_counter()
                trainer.checkpointer.save('short_iteration_40',completed_iterations=40,epoch=0,
                                          purpose='short bring-up only; not a baseline checkpoint')
                report['checkpoint_save_seconds']=time.perf_counter()-save_started
                checkpoint_hashes=states(trainer)
                close_trainer(trainer)
                del trainer
                trainer=None
                gc.collect();torch.cuda.empty_cache()
                print('Creating a fresh trainer and restoring checkpoint',flush=True)
                # Skip ImageNet load; the saved checkpoint supplies all model tensors.
                resume_cfg=cfg.clone();resume_cfg.MODEL.BACKBONE.PRETRAIN=False
                trainer=ShortTrainer(resume_cfg)
                trainer.checkpointer.checkpointables['rng_state']=RNGState()
                load_started=time.perf_counter()
                metadata=trainer.checkpointer.load(str(checkpoint_path))
                report['checkpoint_load_seconds']=time.perf_counter()-load_started
                assert metadata['completed_iterations']==40
                restored=states(trainer)
                assert restored==checkpoint_hashes, 'Checkpoint state hashes differ'
                trainer._trainer.param_wrapper.assert_buffer_is_valid()
                report['checkpoint_restore']={'all_state_hashes_equal':True,'states':restored,
                                              'continued_at_iteration':41,
                                              'sampler_worker_queue_restored':False}
                print('Checkpoint restore verified; continuing iterations 41-50',flush=True)
        rows=report['rows']
        report.update(summarize(report))
        report.update(total_wall_seconds=time.perf_counter()-started,
                      peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,
                      checkpoint_bytes=checkpoint_path.stat().st_size)
    except Exception as error:
        import traceback
        traceback.print_exc()
        report.update(status='failed',error=str(error))
    finally:
        if trainer is not None:close_trainer(trainer)
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    sys.exit(main())
