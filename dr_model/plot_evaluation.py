"""Plot saved evaluation results for the exact ordinal model bundled in the app.

Reads existing artifacts only. No training, model inference, or test-set tuning.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from dr_model.metrics import metrics
from dr_model.paths import PROJECT_ROOT, new_output_path

LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
ORANGE, BLACK, GREY, LIGHT = '#BD4100', '#171717', '#777777', '#F3F1EF'


def roc_points(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    if not labels.any() or labels.all():
        raise ValueError('ROC requires positive and negative examples')
    order = np.argsort(-scores, kind='stable')
    sorted_scores, positives = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores)-1]
    tp = np.cumsum(positives)[ends]
    fp = ends + 1 - tp
    return np.r_[0, fp / (~labels).sum()], np.r_[0, tp / labels.sum()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Original ordinal training run')
    parser.add_argument('--output', type=Path, help='New directory; defaults to outputs/evaluation/<timestamp>')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--app-assets', type=Path, default=PROJECT_ROOT/'android-app/app/src/main/assets')
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error('Use at least 72 DPI')
    # Import only when plotting, so source/tests can be checked without Matplotlib.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import PercentFormatter

    model_meta = json.loads((args.app_assets/'model_metadata.json').read_text())
    checkpoint = args.run/'checkpoints/best.pt'
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != model_meta['checkpoint_sha256']:
        raise ValueError('This run is not the checkpoint used in the app')
    app_model = args.app_assets/'visionfirst.tflite'
    if hashlib.sha256(app_model.read_bytes()).hexdigest() != model_meta['sha256']:
        raise ValueError('Bundled model hash does not match metadata')
    with (args.run/'predictions/test_predictions.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Repeated prediction IDs')
    labels = np.array([int(r['true_grade']) for r in rows])
    cumulative = np.array([[float(r[f'p_gt_{i}']) for i in range(4)] for r in rows])
    predictions = (cumulative > .5).sum(1)
    np.testing.assert_array_equal(predictions, [int(r['predicted_grade']) for r in rows])
    if not np.isfinite(cumulative).all() or np.any((cumulative < 0) | (cumulative > 1)):
        raise ValueError('Invalid saved probabilities')
    report = metrics(labels, cumulative)
    saved = json.loads((args.run/'metrics/test_metrics.json').read_text())
    for key in ['accuracy','qwk','sensitivity','specificity']:
        if not np.isclose(report[key], saved[key], atol=1e-10):
            raise ValueError(f'Saved predictions and metric disagree: {key}')
    np.testing.assert_array_equal(report['confusion_matrix_counts'], saved['confusion_matrix_counts'])
    history = json.loads((args.run/'logs/history.json').read_text())
    selected = max(history, key=lambda row: row['val_qwk'])['epoch']
    with (args.run/'metadata/splits.csv').open(newline='') as f:
        splits = list(csv.DictReader(f))
    manifest_test = {r['id']: int(r['grade']) for r in splits if r['split']=='test'}
    if manifest_test != {r['id']: int(r['true_grade']) for r in rows}:
        raise ValueError('Predictions do not match saved test split')
    output = args.output or new_output_path('evaluation')
    output.mkdir(parents=True, exist_ok=False)
    for name in ['png','svg','tables']:(output/name).mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':17,
        'axes.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,
        'text.color':BLACK,'axes.labelcolor':BLACK,'xtick.color':BLACK,'ytick.color':BLACK,
        'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none'})
    captions = []
    scope = f'Held-out test set · n = {len(labels)} · ordinal model · checkpoint epoch {selected}'
    def save(fig, name, title, caption, subtitle=scope, left=.08):
        fig.suptitle(title, x=.055, y=.98, ha='left', fontweight='bold', fontsize=20)
        fig.text(.055,.89,subtitle,fontsize=10,color=GREY)
        fig.subplots_adjust(top=.80,bottom=.18,left=left,right=.97,wspace=.28)
        fig.text(.055,.025,'VisionFirst | DSC4109 · Research evaluation; not clinical validation',fontsize=9,color=GREY)
        fig.savefig(output/'png'/f'{name}.png',dpi=args.dpi,facecolor='white')
        fig.savefig(output/'svg'/f'{name}.svg',facecolor='white')
        captions.append(f'{name}: {caption}')
        plt.close(fig)

    # 1. Summary with score units clearly distinguished.
    fig,ax=plt.subplots(figsize=(13.33,7.5))
    names=['Accuracy','Macro-F1','QWK','Sensitivity¹','Specificity¹']
    values=[report[k] for k in ['accuracy','macro_f1','qwk','sensitivity','specificity']]
    bars=ax.bar(names,values,color=[ORANGE,BLACK,ORANGE,BLACK,GREY],width=.58)
    ax.set_ylim(0,1.12);ax.set_ylabel('Metric value (0–1)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for bar,value,name in zip(bars,values,names):
        label=f'{value:.3f}' if name in ['QWK','Macro-F1'] else f'{value:.1%}'
        ax.text(bar.get_x()+bar.get_width()/2,value+.025,label,ha='center',fontweight='bold',fontsize=16)
    ax.text(0,-.18,'¹ Screening endpoint: referable DR = grade ≥ 2. Macro-F1 averages the five class F1 scores.',transform=ax.transAxes,fontsize=10)
    save(fig,'01_metrics_summary','Held-out model performance','Exact-grade accuracy, macro-F1, and QWK; sensitivity/specificity use grade ≥ 2.')

    # 2–3. Flat-color matrices: no gradients; rows are actual grades.
    cm=np.array(report['confusion_matrix_counts'])
    for normalized,name in [(False,'02_confusion_counts'),(True,'03_confusion_normalized')]:
        fig,ax=plt.subplots(figsize=(13.33,7.5))
        for i in range(5):
            for j in range(5):
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=ORANGE if i==j else LIGHT,edgecolor='white',linewidth=3))
                text=f'{cm[i,j]/cm[i].sum():.1%}' if normalized else str(cm[i,j])
                ax.text(j,i,text,ha='center',va='center',color='white' if i==j else BLACK,fontsize=16,fontweight='bold')
        ax.set(xlim=(-.5,4.5),ylim=(4.5,-.5),xticks=range(5),yticks=range(5),xlabel='Predicted grade',ylabel='Actual grade')
        ax.set_xticklabels([f'{i}\n{label}' for i,label in enumerate(LABELS)],fontsize=10)
        ax.set_yticklabels([f'{i} · {label} (n={cm[i].sum()})' for i,label in enumerate(LABELS)],fontsize=10)
        for spine in ax.spines.values():spine.set_visible(False)
        fig.subplots_adjust(left=.24)
        save(fig,name,'Confusion matrix — '+('row percentages' if normalized else 'image counts'),'Rows are true grades; columns are predictions. Orange cells are exact matches.', left=.24)
        # save() uses a common margin; long labels are kept within the wide figure.

    # 4. All grades, including difficult/rare classes.
    fig,ax=plt.subplots(figsize=(13.33,7.5));x=np.arange(5);width=.24
    for offset,key,color in [(-1,'precision',BLACK),(0,'recall',ORANGE),(1,'f1',GREY)]:
        values=[r[key] for r in report['per_class']]
        bars=ax.bar(x+offset*width,values,width,label=key.title(),color=color)
        for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+.015,f'{value:.2f}',ha='center',fontsize=9)
    ax.set_xticks(x,[f'{i} · {label}\nn={cm[i].sum()}' for i,label in enumerate(LABELS)],fontsize=10)
    ax.set_ylim(0,1.15);ax.set_ylabel('Score (0–1)');ax.legend(frameon=False,ncol=3);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    save(fig,'04_per_class_metrics','Performance across severity grades','Precision, recall, and F1 for each grade; includes class support.')

    # 5. Threshold ROC, not one-vs-rest class ROC.
    fig,ax=plt.subplots(figsize=(13.33,7.5))
    for i,(color,style) in enumerate([(BLACK,'-'),(ORANGE,'-'),(BLACK,'--'),(ORANGE,':')]):
        fpr,tpr=roc_points(labels>i,cumulative[:,i]);auc=float(np.trapezoid(tpr,fpr))
        if not np.isclose(auc,saved['threshold_auc'][str(i+1)]):raise ValueError('ROC AUC mismatch')
        ax.plot(fpr,tpr,color=color,linestyle=style,lw=2.5,label=f'Grade ≥ {i+1}: AUC {auc:.3f}')
    ax.plot([0,1],[0,1],color=GREY,lw=1,linestyle=':');ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='False-positive rate',ylabel='True-positive rate')
    ax.legend(frameon=False,loc='lower right');ax.grid(alpha=.15)
    save(fig,'05_ordinal_roc','ROC curves at four severity thresholds','ROC curves use cumulative probabilities P(grade ≥ threshold), not class one-vs-rest probabilities.')

    # 6. Validation selected the checkpoint; do not present this as test history.
    epochs=[r['epoch'] for r in history]
    fig,axes=plt.subplots(1,2,figsize=(13.33,7.5))
    axes[0].plot(epochs,[r['train_loss'] for r in history],color=BLACK,marker='o',label='Training BCE')
    axes[0].plot(epochs,[r['val_loss'] for r in history],color=ORANGE,marker='o',label='Validation BCE')
    axes[0].set(xlabel='Epoch',ylabel='Ordinal binary cross-entropy');axes[0].legend(frameon=False)
    axes[1].plot(epochs,[r['val_qwk'] for r in history],color=ORANGE,marker='o')
    axes[1].set(xlabel='Epoch',ylabel='Validation QWK',ylim=(.7,1))
    for ax in axes:
        ax.axvline(selected,color=GREY,linestyle='--',label=f'Selected epoch {selected}');ax.grid(alpha=.15);ax.set_xticks(epochs)
    axes[1].legend(frameon=False)
    save(fig,'06_training_history','Training and validation history',f'Epoch {selected} was selected by validation QWK; training ended after epoch {epochs[-1]}.',subtitle=f'Training and validation only · selected checkpoint epoch {selected}')

    # 7. Magnitude/direction of grading mistakes.
    fig,ax=plt.subplots(figsize=(13.33,7.5));errors=predictions-labels
    bins=np.arange(-4,5);counts=np.array([(errors==e).sum() for e in bins])
    bars=ax.bar(bins,counts,color=[ORANGE if e==0 else BLACK for e in bins],width=.65)
    for bar,count in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,count+3,f'{count}\n{count/len(labels):.1%}',ha='center',fontsize=10)
    ax.set(xlabel='Predicted grade − actual grade (negative = undergrading; positive = overgrading)',ylabel='Number of test images',xticks=bins,ylim=(0,max(counts)*1.18))
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    save(fig,'07_grade_error_distribution','How far are incorrect predictions from the label?','Error 0 is exact; ±1 is an adjacent-grade error. Direction distinguishes undergrading from overgrading.')

    # 8. Retained split composition, not original raw counts.
    fig,ax=plt.subplots(figsize=(13.33,7.5));x=np.arange(5)
    for k,(split,color) in enumerate([('train',BLACK),('validation',GREY),('test',ORANGE)]):
        values=[sum(r['split']==split and int(r['grade'])==g for r in splits) for g in range(5)]
        ax.bar(x+(k-1)*.24,values,.24,color=color,label=f'{split.title()} (n={sum(values):,})')
    ax.set_xticks(x,LABELS);ax.set_ylabel('Images retained after conflicting-label exclusions');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    save(fig,'08_split_distribution','Dataset composition by split','Pixel-identical groups stay together; 62 conflicting-label images were excluded. Patient IDs are unavailable.',subtitle='Retained data n = 3,600 · train / validation / test · original data n = 3,662')

    # Tabular sources and provenance accompany the PPT figures.
    (output/'tables/metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'tables/per_class_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['grade','support','precision','recall','f1']);writer.writeheader();writer.writerows(report['per_class'])
    (output/'captions.txt').write_text('\n\n'.join(captions)+'\n\nScope: saved held-out PyTorch checkpoint evaluation for the original ordinal model deployed via LiteRT. These are not classification-experiment results. Android conversion/device parity does not constitute a second full-test evaluation. Test metrics are already observed; use validation for any subsequent tuning. Pixel grouping does not guarantee patient independence.\n')
    (output/'provenance.json').write_text(json.dumps(dict(run=str(args.run.resolve()),checkpoint_sha256=checkpoint_hash,
        app_model_sha256=model_meta['sha256'],split='test',n=len(labels),selected_epoch=selected,
        model_bytes=model_meta['size_bytes'],matplotlib=matplotlib.__version__,numpy=np.__version__,
        note='Saved original-checkpoint test predictions; app uses parity-checked LiteRT conversion. No inference rerun.'),indent=2)+'\n')
    print('Saved 8 figures in PNG and SVG, metrics tables, captions, and provenance to:',output.resolve())
    print('Test metrics:', {k:report[k] for k in ['accuracy','qwk','macro_f1','sensitivity','specificity']})


if __name__=='__main__':main()
