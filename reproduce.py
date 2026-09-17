from pathlib import Path
import argparse,subprocess,sys,shutil

ROOT=Path(__file__).resolve().parent
STAGES={
    '1':('prepare_step1_analysis.py','step1_data_quality'),
    '2':('describe_step2_reliability.py','step2_descriptive_reliability'),
    '3':('fit_step3_supervised_complexity_models.py','step3_supervised_complexity'),
    '4':('fit_step4_llm_prompt_models.py','step4_llm_prompt_analysis'),
    '5':('fit_step5_common24_cross_model.py','step5_common24_cross_model'),
    '6-grouped':('fit_step6_performance_prediction.py','step6_performance_prediction'),
    '6-record':('fit_step6_record_stratified_10fold.py','step6_record_stratified_10fold'),
    '7':('build_step7_diagnostic_guidance.py','step7_diagnostic_guidance'),
    '8':('run_step8_robustness_checks.py','step8_robustness'),
}

def run(args):
    print('Running: '+' '.join(str(x) for x in args),flush=True)
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)

def main():
    parser=argparse.ArgumentParser(description='Verify or reproduce analyses from public task-level records. No text-model training or API requests.')
    parser.add_argument('--steps',nargs='+',choices=['verify','figures','fractional','all',*STAGES],default=['verify'])
    args=parser.parse_args();steps=list(STAGES)+['fractional','figures'] if 'all' in args.steps else args.steps
    dest=ROOT/'reproduced/statistical_analysis'
    if any(s in STAGES for s in steps) and not dest.exists():shutil.copytree(ROOT/'results',dest)
    for step in steps:
        if step=='verify':run([ROOT/'code/verify_release.py']);continue
        if step=='figures':run([ROOT/'code/figures/redraw_optimized_figures.py']);continue
        if step=='fractional':run([ROOT/'code/run_fractional_logit_robustness.py']);continue
        script,out=STAGES[step];cmd=[ROOT/'code'/script]
        if step=='1':cmd+=['--input-dir',ROOT/'data/task_records','--package-root',ROOT,'--skip-deep-test-checks']
        elif step in ['7','8']:
            cmd+=['--analysis-root',dest,'--step6-dir',dest/'step6_record_stratified_10fold']
            if step=='8':cmd+=['--step7-dir',dest/'step7_diagnostic_guidance']
        else:cmd+=['--input-dir',dest/'step1_data_quality']
        cmd+=['--output-dir',dest/out]
        run(cmd)

if __name__=='__main__':main()
