import json,math,re
from pathlib import Path
root=Path('/root/gpufree-data/pi05-repro')
out=root/'results/train-smoke'
rows=[]
for step,rest in re.findall(r'Step (\d+): ([^\r\n]+)',(root/'logs/train-smoke.log').read_text()):
 vals={k:float(v) for k,v in re.findall(r'(grad_norm|loss|param_norm)=([-+0-9.eE]+|nan|inf)',rest)}
 assert len(vals)==3 and all(math.isfinite(x) for x in vals.values()),(step,rest)
 rows.append({'step_index':int(step),**vals})
assert len(rows)==100 and [r['step_index'] for r in rows]==list(range(100))
assert all(r['grad_norm']>0 for r in rows)
trained=json.loads((out/'training-verification.json').read_text())
restored=json.loads((out/'restore-verification.json').read_text())
assert trained['training_steps']==100 and trained['checkpoint_save_completed']
assert trained['sampled_lora_updated'] and trained['sampled_frozen_unchanged']
assert restored['restored_step']==100 and restored['parameter_samples_identical']
(out/'metrics-from-log.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
summary={'steps_with_finite_metrics':len(rows),'first_step':rows[0],'last_step':rows[-1],'mean_loss_first10':sum(r['loss'] for r in rows[:10])/10,'mean_loss_last10':sum(r['loss'] for r in rows[-10:])/10,'precision':'parsed from upstream logs rounded to four decimals','all_gradient_norms_positive':True}
(out/'metrics-summary.json').write_text(json.dumps(summary,indent=2))
print('SHORT_TRAINING_CHECKS_PASS',json.dumps(summary))
