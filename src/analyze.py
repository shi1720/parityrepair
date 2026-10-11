"""Generate every manuscript number, table and figure from saved raw results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':8.5,'axes.labelsize':8.5,'legend.fontsize':7.5,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
COL={'Unrepaired':'#888888','CORAL':'#c2935c','Marginal':'#8a62aa','P1':'#7c8c9e','P2':'#207e9c','P3':'#b54e42','ParityRepair':'#19734a','Oracle':'#222222','KnownSigns':'#222222'}
NAMES={'banknote':'Banknote','cancer':'WDBC','ionosphere':'Ionosphere','red':'Wine-red','white':'Wine-white'}

def se(x): return x.std(ddof=1)/np.sqrt(len(x))

def main():
 root=Path('.'); paper=root/'paper'; fig=root/'figures'; fig.mkdir(exist_ok=True); paper.mkdir(exist_ok=True)
 x=pd.read_csv('results/final/real.csv'); x['dataset']=pd.Categorical(x.dataset,categories=list(NAMES),ordered=True)
 primary=x[(x.n==32)&(x.model=='MLP')]
 grouped=primary.groupby(['dataset','seed','method'],observed=True).accuracy.mean().unstack('method')
 methods=['Unrepaired','Marginal','CORAL','P2','P3','ParityRepair','Oracle']
 means=grouped.groupby('dataset',observed=True).mean(); errors=grouped.groupby('dataset',observed=True).agg(se)
 lines=[r'\begin{table}[t]',r'\caption{MLP accuracy (\%) at 32 unlabeled calibration examples, averaged over six masks. Parentheses give standard errors across twenty splits after mask aggregation. PR denotes source-selected ParityRepair.}\label{tab:accuracy}',r'\centering\setlength{\tabcolsep}{2.5pt}\small',r'\begin{tabular}{lrrrrrrr}\toprule',r'Dataset & Unrep. & Marg. & CORAL & P2 & P3 & PR & Oracle\\\midrule']
 for ds in NAMES:
  vals=[f'\\shortstack{{{100*means.loc[ds,m]:.1f}\\\\({100*errors.loc[ds,m]:.1f})}}' for m in methods]
  lines.append(NAMES[ds]+' & '+' & '.join(vals)+r'\\')
 lines += [r'\bottomrule\end{tabular}',r'\end{table}']
 (paper/'accuracy_table.tex').write_text('\n'.join(lines))
 meta=json.loads(Path('results/data_manifest.json').read_text()); datalines=[r'\begin{table}[t]\centering',r'\caption{Public datasets after exact-record deduplication. All tasks are binary.}\label{tab:data}',r'\begin{tabular}{lrrl}\toprule Dataset & Records & Features & Target\\\midrule']
 for ds in NAMES:
  target='Quality $\\ge6$' if ds in ['red','white'] else ('Good radar return' if ds=='ionosphere' else ('Benign diagnosis' if ds=='cancer' else 'Dataset class'))
  datalines.append(f'{NAMES[ds]} & {meta[ds]["n_after_dedup"]} & {meta[ds]["d"]} & {target}'+r'\\')
 datalines += [r'\bottomrule\end{tabular}\end{table}'];(paper/'data_table.tex').write_text('\n'.join(datalines))
 # Dataset-equal summaries, not row-count weighted summaries.
 summary=means.mean().to_dict(); deltas={}
 for m in ['Unrepaired','Marginal','CORAL','P2','P3']:
  per=grouped['ParityRepair']-grouped[m]; per=per.groupby('seed').mean()
  deltas[m]={'delta_pp':100*per.mean(),'se_pp':100*se(per)}
 lr=x[(x.n==32)&(x.model=='LR')].groupby(['dataset','method'],observed=True).accuracy.mean().groupby('method').mean()
 variants=primary.groupby(['dataset','method'],observed=True)[['accuracy','sign_accuracy','exact']].mean()
 variants.to_csv('results/final/primary_summary.csv');grouped.to_csv('results/final/split_summary.csv')
 text=(f'Across the five datasets, giving each dataset equal weight, mean MLP accuracy is {100*summary["ParityRepair"]:.1f}\\% for \\PR{{}}, {100*summary["Unrepaired"]:.1f}\\% without repair, {100*summary["Marginal"]:.1f}\\% for marginal matching, {100*summary["P2"]:.1f}\\% for P2, and {100*summary["P3"]:.1f}\\% for P3. The paired PR--P2 difference is {deltas["P2"]["delta_pp"]:.1f} percentage points (split standard error {deltas["P2"]["se_pp"]:.1f}); PR--P3 is {deltas["P3"]["delta_pp"]:.1f} points ({deltas["P3"]["se_pp"]:.1f}). These descriptive comparisons do not establish universal dominance. On ionosphere, P3 reaches the known-sign reference, while P2 leaves more orientation errors. On WDBC, P3 is worse than P2; the source-only selector recovers part of that loss. Logistic regression yields dataset-equal means of {100*lr["ParityRepair"]:.1f}\\% for PR, {100*lr["P2"]:.1f}\\% for P2, and {100*lr["P3"]:.1f}\\% for P3. Full metrics and the P1/P2+U3 controls are released.\n')
 (paper/'results_paragraph.tex').write_text(text)
 # Synthetic, seed is unit of uncertainty.
 syn=pd.read_csv('results/synthetic.csv'); g=syn.groupby(['seed','n','degree'])[['accuracy','exact','rank']].mean().reset_index()
 f,axs=plt.subplots(1,3,figsize=(4.8,1.85),constrained_layout=True)
 for ax,metric,label in zip(axs,['exact','accuracy','rank'],['Exact recovery','Classifier accuracy','Confidence rank']):
  for k in ([3] if metric=='rank' else [1,2,3]):
   h=g[g.degree==k].groupby('n')[metric];mu=h.mean();err=h.agg(se)
   ax.plot(mu.index,mu.values,'o-',ms=3,label=f'P{k}',color=COL['P'+str(k)]);ax.fill_between(mu.index,mu-err,mu+err,alpha=.15,color=COL['P'+str(k)])
  ax.set_xscale('log',base=2);ax.set_xlabel('Calibration examples');ax.set_ylabel(label);ax.set_xticks([16,256,4096],['16','256','4096']);ax.grid(alpha=.2);ax.legend(frameon=False,loc='lower right')
 axs[0].set_ylim(0,1.04);axs[1].set_ylim(.3,1.04);axs[2].set_ylim(0,4.15)
 f.savefig(fig/'synthetic.pdf');f.savefig(fig/'synthetic.png',dpi=220);plt.close(f)
 f,axs=plt.subplots(1,3,figsize=(4.8,1.85),constrained_layout=True)
 for ax,ds in zip(axs,['banknote','cancer','ionosphere']):
  for m in ['Marginal','P2','P3','ParityRepair','Oracle']:
   h=x[(x.dataset==ds)&(x.model=='MLP')&(x.method==m)].groupby('n').accuracy.mean()
   ax.plot(h.index,h.values,'o',ms=2,lw=1,label=m,color=COL[m],linestyle='--' if m=='Oracle' else '-')
  ax.set_title(NAMES[ds]);ax.set_ylim(.45,1.02);ax.set_xscale('log',base=2);ax.set_xticks(h.index,[str(int(n)) for n in h.index]);ax.set_xlim(h.index.min()*.85,h.index.max()*1.18);ax.set_xlabel('Batch size');ax.grid(alpha=.2)
 axs[0].set_ylabel('MLP accuracy');handles,labels=axs[-1].get_legend_handles_labels();f.legend(handles,labels,loc='outside upper center',ncol=5,frameon=False)
 f.savefig(fig/'sample_size.pdf');f.savefig(fig/'sample_size.png',dpi=220);plt.close(f)
 stress=pd.read_csv('results/stress.csv');f,axs=plt.subplots(1,2,figsize=(4.8,1.85),constrained_layout=True)
 sm={}
 for ax,fault in zip(axs,['offset','noise']):
  part=stress[stress.fault==fault];h=part.groupby(['seed','level','method']).accuracy.mean().reset_index()
  for m in ['Unrepaired','ParityRepair','KnownSigns']:
   q=h[h.method==m].groupby('level').accuracy;mu=q.mean();err=q.agg(se);ax.plot(mu.index,mu.values,'o-',ms=3,color=COL[m],label='Known signs' if m=='KnownSigns' else m);ax.fill_between(mu.index,mu-err,mu+err,color=COL[m],alpha=.12)
   sm[fault+'_'+m]=mu.to_dict()
  ax.set_xlabel('Offset magnitude' if fault=='offset' else 'Noise standard deviation');ax.set_ylabel('MLP accuracy');ax.set_ylim(.4,1);ax.grid(alpha=.2);ax.legend(frameon=False)
 f.savefig(fig/'stress.pdf');f.savefig(fig/'stress.png',dpi=220);plt.close(f)
 para=(f'With an added unit offset, dataset-equal mean MLP accuracy is {100*sm["offset_ParityRepair"][1.]:.1f}\\% after PR and {100*sm["offset_KnownSigns"][1.]:.1f}\\% after known-sign inversion. With unit Gaussian noise, the corresponding values are {100*sm["noise_ParityRepair"][1.]:.1f}\\% and {100*sm["noise_KnownSigns"][1.]:.1f}\\%. These tests expose sensitivity to a misspecified transformation rather than extending the theorem. Offset drift changes odd moments and can produce an incorrect orientation even when the class mechanism remains unchanged.\n')
 (paper/'stress_paragraph.tex').write_text(para)
 report={'primary_mlp_equal_dataset_accuracy':summary,'paired_split_differences':deltas,'lr_equal_dataset_accuracy':lr.to_dict(),'rows_final':len(x),'unique_final_splits':int(x[['dataset','seed']].drop_duplicates().shape[0]),'synthetic_rows':len(syn),'stress_rows':len(stress),'stress':sm,'exact_solver_audit':pd.read_csv('results/solver_audit.csv').exact_objective.mean()}
 Path('results/summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
