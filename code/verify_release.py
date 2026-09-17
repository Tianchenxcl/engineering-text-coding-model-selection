from pathlib import Path
import csv,json,math,hashlib,argparse

ROOT=Path(__file__).resolve().parents[1]
def rows(rel):
    with (ROOT/rel).open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))

def main():
    checks={}
    data=rows('results/step6_performance_prediction/diagnostics/prediction_unified_analysis_data.csv')
    checks['600_unique_model_task_records']=len(data)==600 and len({r['record_id'] for r in data})==600
    checks['40_label_combinations']=len({r['combination_id'] for r in data})==40
    checks['480_supervised_120_llm_means']=sum(r['model_variant'] in ['SVM','DeBERTa'] for r in data)==480
    checks['240_supervised_tasks']=len(rows('manifests/source_manifests/supervised_task_manifest_240.csv'))==240
    checks['360_llm_runs']=len(rows('data/task_records/llm_performance_long.csv'))==360
    results={}
    for stage,filename,algorithm,expected in [
        ('step6_record_stratified_10fold','record_10fold_oof_predictions.csv','gradient_boosting',0.08413100480560769),
        ('step6_performance_prediction','prediction_oof_records.csv','ridge',0.11041358299504746),
    ]:
        selected=[r for r in rows(f'results/{stage}/predictions/{filename}') if r['outcome']=='kappa' and r['algorithm']==algorithm]
        rmse=math.sqrt(sum((float(r['actual'])-float(r['prediction']))**2 for r in selected)/len(selected))
        checks[stage+'_600_unique_predictions']=len(selected)==600 and len({r['record_id'] for r in selected})==600
        checks[stage+'_reported_rmse']=math.isclose(rmse,expected,abs_tol=1e-12)
        if stage=='step6_performance_prediction':
            folds={}
            for r in selected:folds.setdefault(r['combination_id'],set()).add(r['outer_fold'])
            checks['label_combinations_held_out_together']=all(len(v)==1 for v in folds.values())
        results[stage]={'algorithm':algorithm,'kappa_rmse':rmse}
    for p in ROOT.rglob('*.py'):
        if '.venv' not in p.parts and 'reproduced' not in p.parts:compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
    checks['python_syntax']=True
    manifest=ROOT/'SHA256SUMS.csv'
    if manifest.exists():
        checks['release_file_hashes']=all((ROOT/r['path']).is_file() and hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in rows('SHA256SUMS.csv'))
    report={'checks':checks,'prediction_results':results,'all_passed':all(checks.values())}
    dest=ROOT/'reproduced/verification.json';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not report['all_passed']:raise SystemExit(1)

if __name__=='__main__':main()
