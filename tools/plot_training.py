"""Plot recorded training metrics only; never infer validation/test curves."""
from pathlib import Path
from collections import defaultdict
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'logs/CARGO/ViSA_official_checkpointing/metrics.json'
OUTPUT=ROOT/'output/figures'


def main():
    records=[json.loads(line) for line in SOURCE.read_text(encoding='utf-8').splitlines() if line.strip()]
    records=sorted({r['iteration']:r for r in records}.values(),key=lambda r:r['iteration'])
    assert records[-1]['iteration']==23999
    assert all(np.isfinite(r['total_loss']) for r in records)
    grouped=defaultdict(list)
    for row in records:grouped[row['iteration']//200+1].append(row)
    keys=['total_loss','loss_cls_id','loss_cls_local','loss_triplet_id','loss_triplet_local',
          'loss_cls_view','loss_oreg','load_balance_loss','cls_accuracy','lr']
    epochs=[]
    for epoch,rows in sorted(grouped.items()):
        epochs.append({'epoch':epoch,'logged_records':len(rows),
                       **{key:float(np.mean([r[key] for r in rows])) for key in keys},
                       'lr_last':rows[-1]['lr']})
    assert len(epochs)==120
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'cargo_training_epoch_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(epochs[0]))
        writer.writeheader();writer.writerows(epochs)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                         'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#ccd5df','axes.labelcolor':'#26364a',
                         'xtick.color':'#526174','ytick.color':'#526174','svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12.8,8.4),dpi=180)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=.075,right=.97,top=.83,bottom=.14,hspace=.42,wspace=.24)
    fig.text(.075,.945,'CARGO / ViSA training curves',fontsize=21,fontweight='bold',color='#142b45')
    fig.text(.075,.903,'120 epochs  |  Batch 256  |  Activation checkpointing + AMP',fontsize=11,color='#526174')
    x=np.array([r['epoch'] for r in epochs])
    logged_x=np.array([(r['iteration']+1)/200 for r in records])
    for ax in axes.flat:
        ax.set_facecolor('#fafcfe')
        ax.grid(axis='y',color='#e5eaf0',linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_xlim(0,120)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.set_xlabel('Epoch')
    ax=axes[0,0]
    ax.plot(logged_x,[r['total_loss'] for r in records],color='#84b1d8',alpha=.4,linewidth=.7,label='Logged values')
    ax.plot(x,[r['total_loss'] for r in epochs],color='#175b94',linewidth=2.1,label='Epoch mean')
    ax.set_title('A   Total training loss',loc='left',fontweight='bold',pad=12)
    ax.set_ylabel('Loss');ax.set_ylim(bottom=0)
    ax.legend(frameon=False,fontsize=9)
    ax=axes[0,1]
    for key,label,color,style in [
        ('loss_cls_id','Global classification','#175b94','-'),
        ('loss_cls_local','Local classification','#cf7556','-'),
        ('loss_triplet_id','Global triplet','#24877f','--'),
        ('loss_triplet_local','Local triplet','#8b6fa8','--')]:
        ax.plot(x,[r[key] for r in epochs],color=color,linestyle=style,linewidth=1.8,label=label)
    ax.set_title('B   Identity loss components',loc='left',fontweight='bold',pad=12)
    ax.set_ylabel('Loss (epoch mean)');ax.set_ylim(bottom=0)
    ax.legend(frameon=False,fontsize=8.5,ncol=2,columnspacing=1)
    ax=axes[1,0]
    ax.plot(logged_x,[100*r['cls_accuracy'] for r in records],color='#9bc7bc',alpha=.45,linewidth=.7)
    ax.plot(x,[100*r['cls_accuracy'] for r in epochs],color='#177d66',linewidth=2.1)
    ax.set_title('C   Training identity classification accuracy',loc='left',fontweight='bold',pad=12)
    ax.set_ylabel('Accuracy (epoch mean)');ax.set_ylim(0,101)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax=axes[1,1]
    ax.plot(logged_x,[r['lr'] for r in records],color='#9b6536',linewidth=2)
    ax.axvspan(0,10,color='#f3eadc',alpha=.8,zorder=0)
    ax.text(12,.0077,'Warmup → cosine decay',fontsize=9,color='#765638',va='top')
    ax.set_title('D   Recorded learning rate',loc='left',fontweight='bold',pad=12)
    ax.set_ylabel('Learning rate');ax.set_ylim(bottom=0)
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,-3))
    fig.text(.075,.062,f'Source: {len(records):,} logged records (every 20 steps). Epoch means average the recorded values.',
             fontsize=9,color='#526174')
    fig.text(.075,.034,'Classification accuracy is a training metric, not retrieval Rank-1. No validation curve was recorded.',
             fontsize=9,color='#526174')
    for extension in ('png','svg'):
        fig.savefig(OUTPUT/f'cargo_training_curves.{extension}',facecolor='white',dpi=180)
    plt.close(fig)
    print(json.dumps({'epochs':len(epochs),'logged_records':len(records),
                      'first_epoch_mean_loss':epochs[0]['total_loss'],
                      'last_epoch_mean_loss':epochs[-1]['total_loss'],
                      'last_epoch_mean_training_accuracy_percent':100*epochs[-1]['cls_accuracy'],
                      'outputs':[str(OUTPUT/f'cargo_training_curves.{extension}') for extension in ('png','svg')]},indent=2))


if __name__=='__main__':main()
