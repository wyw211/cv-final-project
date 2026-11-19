"""Test real checkpoint files and hooks on a tiny CPU model; no ViSA training."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, hooks
from fastreid.engine.train_loop import TrainerBase
from fastreid.utils.checkpoint import Checkpointer, PeriodicCheckpointer
from fastreid.utils.env import RandomState

OUT=ROOT/'output/audit'/('checkpoint_verification_'+uuid.uuid4().hex[:8])
OUT.mkdir(parents=True,exist_ok=False)


def objects(directory):
    directory.mkdir(parents=True,exist_ok=True)
    model=torch.nn.Linear(2,1)
    optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
    for p in model.parameters():
        optimizer.state[p]['momentum_buffer']=torch.full_like(p,.25)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=110)
    scaler=torch.amp.GradScaler('cpu',init_scale=1024)
    cp=Checkpointer(model,str(directory),optimizer=optimizer,lr_sched=scheduler,
                    grad_scaler=scaler,random_state=RandomState())
    return model,optimizer,scheduler,scaler,cp


class CheckpointTests(unittest.TestCase):
    def test_schedule_120_epochs_without_metrics(self):
        class Recorder:
            def __init__(self):self.calls=[]
            def save(self,name,**state):self.calls.append((name,state))
        recorder=Recorder()
        periodic=PeriodicCheckpointer(recorder,5,120)
        for epoch in range(120):periodic.step(epoch)
        latest=[s['completed_epochs'] for n,s in recorder.calls if n=='model_latest']
        final=[s['completed_epochs'] for n,s in recorder.calls if n=='model_final']
        self.assertEqual(latest,list(range(5,121,5)))
        self.assertEqual(final,[120])
        self.assertNotIn('model_best',[n for n,s in recorder.calls])
        recorder2=Recorder()
        for epoch in range(7):
            PeriodicCheckpointer(recorder2,5,7).step(epoch,metric=100-epoch)
        self.assertEqual([(n,s['completed_epochs']) for n,s in recorder2.calls],
                         [('model_latest',5),('model_latest',7),('model_final',7)])

    def test_real_files_fresh_restore_and_epoch_resume(self):
        directory=OUT/'roundtrip'
        model,opt,sched,scaler,cp=objects(directory)
        periodic=PeriodicCheckpointer(cp,5,120)
        with torch.no_grad():
            for p in model.parameters():p.fill_(5)
        for epoch in range(5):periodic.step(epoch,completed_iterations=(epoch+1)*200)
        self.assertEqual(sorted(p.name for p in directory.glob('*.pth')),['model_latest.pth'])
        expected=[p.detach().clone() for p in model.parameters()]
        expected_random=(random.random(),np.random.rand(3),torch.rand(3))
        _,opt2,sched2,scaler2,cp2=objects(directory)
        fresh=SimpleNamespace(checkpointer=cp2,cfg=SimpleNamespace(MODEL=SimpleNamespace(WEIGHTS='')),start_epoch=0)
        DefaultTrainer.resume_or_load(fresh,resume=True)
        self.assertEqual(fresh.start_epoch,5)  # completed epoch 5 -> start epoch 6 (zero-based 5)
        for p,w in zip(cp2.model.parameters(),expected):torch.testing.assert_close(p,w,rtol=0,atol=0)
        self.assertEqual(opt2.param_groups[0]['lr'],opt.param_groups[0]['lr'])
        for p in cp2.model.parameters():
            torch.testing.assert_close(opt2.state[p]['momentum_buffer'],torch.full_like(p,.25),rtol=0,atol=0)
        self.assertEqual(sched.state_dict(),sched2.state_dict())
        self.assertEqual(scaler.state_dict(),scaler2.state_dict())
        actual_random=(random.random(),np.random.rand(3),torch.rand(3))
        self.assertEqual(expected_random[0],actual_random[0])
        np.testing.assert_array_equal(expected_random[1],actual_random[1])
        torch.testing.assert_close(expected_random[2],actual_random[2],rtol=0,atol=0)
        # Latest overwrites instead of accumulating 24 full checkpoint files.
        for epoch in range(5,120):periodic.step(epoch,completed_iterations=(epoch+1)*200)
        self.assertEqual(sorted(p.name for p in directory.glob('*.pth')),['model_final.pth','model_latest.pth'])
        self.assertEqual((directory/'last_checkpoint').read_text(),'model_final.pth')
        metadata=cp2.load(str(directory/'model_final.pth'))
        self.assertEqual(metadata['epoch'],119)
        self.assertEqual(metadata['completed_epochs'],120)
        self.assertEqual(metadata['completed_iterations'],24000)

    def test_failed_write_keeps_previous_latest(self):
        directory=OUT/'atomic_failure'
        model,opt,sched,scaler,cp=objects(directory)
        cp.save('model_latest',epoch=4)
        path=directory/'model_latest.pth'
        original=hashlib.sha256(path.read_bytes()).hexdigest()
        pointer=(directory/'last_checkpoint').read_text()
        def failed_save(data,stream):
            stream.write(b'incomplete checkpoint')
            raise OSError('simulated interrupted write')
        with patch('torch.save',side_effect=failed_save):
            with self.assertRaises(OSError):cp.save('model_latest',epoch=9)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),original)
        self.assertEqual((directory/'last_checkpoint').read_text(),pointer)
        self.assertEqual(cp.load(str(path))['epoch'],4)
        self.assertFalse(list(directory.glob('*.tmp')))

    def test_final_checkpoint_precedes_failed_evaluation(self):
        directory=OUT/'eval_failure'
        model,opt,sched,scaler,cp=objects(directory)
        cfg=get_cfg();cfg.merge_from_file(str(ROOT/'configs/CARGO/visa_official_checkpointing.yml'))
        def fail_eval(*args):raise RuntimeError('simulated final evaluation failure')
        fake=SimpleNamespace(cfg=cfg,model=model,optimizer=opt,scheduler={'lr_sched':sched},
                             checkpointer=cp,build_writers=lambda:[],test=fail_eval,
                             max_epoch=120,epoch=119,iter=24000)
        built=DefaultTrainer.build_hooks(fake)
        periodic=next(h for h in built if isinstance(h,hooks.PeriodicCheckpointer))
        evaluation=next(h for h in built if isinstance(h,hooks.EvalHook))
        self.assertLess(built.index(periodic),built.index(evaluation))
        periodic.trainer=fake;evaluation.trainer=fake
        periodic.before_train()
        periodic.after_epoch()
        with self.assertRaises(RuntimeError):evaluation.after_epoch()
        self.assertTrue((directory/'model_final.pth').exists())
        metadata=cp.load(str(directory/'model_final.pth'))
        self.assertEqual(metadata['completed_iterations'],24000)
        self.assertEqual((directory/'last_checkpoint').read_text(),'model_final.pth')

    def test_actual_epoch_loop_records_completed_iterations(self):
        _,_,_,_,cp=objects(OUT/'epoch_loop')
        class DryRun(TrainerBase):
            def __init__(self):
                super().__init__()
                self.max_epoch=5
                self.register_hooks([hooks.PeriodicCheckpointer(cp,5)])
            def run_step(self):
                pass  # Only exercise hook scheduling, not any model training.
        DryRun().train(0,5,2)
        data=cp.load(str(OUT/'epoch_loop/model_final.pth'))
        self.assertEqual(data['epoch'],4)
        self.assertEqual(data['completed_epochs'],5)
        self.assertEqual(data['completed_iterations'],10)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CheckpointTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    summary={'status':'passed' if result.wasSuccessful() else 'failed',
             'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'viSA_training_started':False,'output_directory':str(OUT)}
    (OUT/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (ROOT/'output/audit/latest_checkpoint_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
