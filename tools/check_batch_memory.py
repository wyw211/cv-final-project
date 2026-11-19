"""Measure ViSA training memory using synthetic or CARGO train data; no step()."""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import torch
from fastreid.config import get_cfg
from fastreid.data import build_reid_train_loader
from fastreid.modeling import build_model
from fastreid.solver import build_optimizer
from fastreid.utils.env import seed_all_rng
from fastreid.utils.events import EventStorage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--activation-checkpointing', action='store_true')
    parser.add_argument('--passes', type=int, default=1)
    parser.add_argument('--real-training-data', action='store_true')
    args = parser.parse_args()
    assert args.batch_size % 4 == 0 and args.passes > 0
    torch.set_num_threads(8)
    seed_all_rng(0)
    # Windows can spill CUDA allocations into shared system RAM. Bound the
    # caching allocator to the GPU's physical capacity to test dedicated VRAM.
    torch.cuda.set_per_process_memory_fraction(1.0, 0)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = True
    free, total = torch.cuda.mem_get_info()
    report = {'timestamp_utc':datetime.now(timezone.utc).isoformat(),
              'gpu':torch.cuda.get_device_name(0), 'torch':torch.__version__,
              'cuda':torch.version.cuda, 'total_GiB':total/1024**3,
              'initial_free_GiB':free/1024**3,
              'batch_size':args.batch_size, 'shape':[args.batch_size,3,256,128],
              'precision':'official AMP fp16 with FP32 parameters',
              'allocator_fraction_of_physical_memory':1.0,
              'synthetic_data_only':not args.real_training_data, 'optimizer_steps':0,
              'passes_requested':args.passes, 'passes_completed':0,
              'activation_checkpointing':args.activation_checkpointing,
              'checkpoint_written':False, 'milestones':[]}
    phase = 'initialization'
    loader = None
    start = time.perf_counter()

    def milestone(name):
        nonlocal phase
        phase = name
        torch.cuda.synchronize()
        item={'phase':name,'allocated_GiB':torch.cuda.memory_allocated()/1024**3,
              'reserved_GiB':torch.cuda.memory_reserved()/1024**3,
              'elapsed_seconds':time.perf_counter()-start}
        report['milestones'].append(item)
        print(json.dumps(item),flush=True)

    try:
        cfg = get_cfg()
        cfg.merge_from_file('configs/CARGO/visa_official_single_gpu.yml')
        cfg.SOLVER.IMS_PER_BATCH = args.batch_size
        cfg.MODEL.BACKBONE.ACTIVATION_CHECKPOINTING = args.activation_checkpointing
        cfg.MODEL.HEADS.NUM_CLASSES = 2500
        phase = 'build_model'
        model = build_model(cfg).train()
        milestone('model_loaded')
        phase = 'build_official_optimizer_contiguous_buffers'
        optimizer, wrapper = build_optimizer(cfg, model)
        milestone('official_optimizer_created')
        # Reserve momentum BEFORE the forward to cover steady-state memory,
        # including later batches after the optimizer has initialized its state.
        for group in optimizer.param_groups:
            for param in group['params']:
                if group.get('momentum',0):
                    optimizer.state[param]['momentum_buffer'] = torch.zeros_like(param)
        report['momentum_buffer_GiB'] = sum(
            state['momentum_buffer'].numel()*state['momentum_buffer'].element_size()
            for state in optimizer.state.values() if 'momentum_buffer' in state)/1024**3
        assert report['momentum_buffer_GiB'] > 1.0
        milestone('momentum_buffers_reserved_no_step')
        batch={'images':torch.rand(args.batch_size,3,256,128,device='cuda')*255,
               'targets':torch.arange(args.batch_size//4,device='cuda').repeat_interleave(4),
               'camids':torch.tensor([0,1,5,6]*(args.batch_size//4),device='cuda'),
               'viewids':['Aerial','Aerial','Ground','Ground']*(args.batch_size//4)}
        scaler = torch.amp.GradScaler('cuda')
        if args.real_training_data:
            cfg.DATALOADER.NUM_WORKERS = 0
            report['loader_workers'] = 0
            loader = build_reid_train_loader(cfg)
            iterator = iter(loader)
            del batch
        milestone('training_loader_ready' if loader is not None else 'synthetic_batch_loaded')
        with EventStorage():
            for pass_index in range(args.passes):
                if loader is not None:
                    batch = next(iterator)
                elif pass_index > 0:
                    batch['images'] = torch.rand(args.batch_size,3,256,128,device='cuda')*255
                phase = 'training_forward_with_autograd'
                print(f'Pass {pass_index+1}: {phase}',flush=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    losses = model(batch)
                    loss = sum(losses.values())
                assert torch.isfinite(loss)
                milestone('forward_completed')
                phase = 'training_backward'
                optimizer.zero_grad(set_to_none=False)
                scaler.scale(loss).backward()
                wrapper.assert_buffer_is_valid()
                assert all(p.grad is not None for group in optimizer.param_groups for p in group['params'])
                milestone('backward_completed')
                # Record finiteness after unscaling, allowing the initial AMP
                # scale to adapt as in training, while never calling step().
                scale_before = scaler.get_scale()
                scaler.unscale_(optimizer)
                finite = all(torch.isfinite(p.grad).all().item()
                             for group in optimizer.param_groups for p in group['params'])
                report.setdefault('gradient_finite_per_pass',[]).append(finite)
                scaler.update()
                report.setdefault('amp_scales',[]).append({'before':scale_before,'after':scaler.get_scale()})
                report['passes_completed'] += 1
                del losses,loss
        report['status'] = 'forward_backward_and_state_reservation_fit'
    except torch.OutOfMemoryError as error:
        report.update(status='out_of_memory',failed_phase=phase,error=str(error))
        print(f'OUT OF MEMORY during {phase}',flush=True)
    except Exception as error:
        import traceback
        traceback.print_exc()
        report.update(status='error',failed_phase=phase,error=str(error))
    finally:
        if loader is not None:
            loader.shutdown()
        report['peak_allocated_GiB']=torch.cuda.max_memory_allocated()/1024**3
        report['peak_reserved_GiB']=torch.cuda.max_memory_reserved()/1024**3
        report['elapsed_seconds']=time.perf_counter()-start
        suffix = '_checkpointing' if args.activation_checkpointing else ''
        if args.real_training_data:
            suffix += '_cargo'
        output=ROOT/f'output/audit/batch_{args.batch_size}{suffix}_memory.json'
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)
    return 1 if report['status']=='error' else 0


if __name__ == '__main__':
    sys.exit(main())
