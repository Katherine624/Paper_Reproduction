import hashlib,json,subprocess,tarfile
from pathlib import Path
root=Path('/root/gpufree-data/pi05-repro')
train=root/'results/train-smoke';ev=root/'results/eval-libero-spatial-seed7-trials10'
s=json.loads((ev/'summary.json').read_text());records=[json.loads(x) for x in (ev/'episodes.jsonl').read_text().splitlines()]
t=json.loads((train/'training-verification.json').read_text());r=json.loads((train/'restore-verification.json').read_text());metrics=json.loads((train/'metrics-summary.json').read_text())
assert s['completed'] and len(records)==s['episodes']==100
assert s['successes']==sum(int(x['success']) for x in records)
assert [x['episode_index'] for x in records]==list(range(100))
assert len(s['per_task'])==10 and all(x['episodes']==10 for x in s['per_task'].values())
assert not (ev/'infrastructure-error.txt').exists()
assert t['training_steps']==r['restored_step']==metrics['steps_with_finite_metrics']==100
assert t['sampled_lora_updated'] and t['sampled_frozen_unchanged'] and t['checkpoint_save_completed'] and r['parameter_samples_identical']
for record in records:
 p=(ev/record['video']).resolve();assert p.is_relative_to(ev.resolve()) and p.is_file() and p.stat().st_size>0
assert len(list((ev/'videos').glob('*.mp4')))==100
protocol=json.loads((root/'results/experiment-protocol.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'openpi',text=True).strip()==protocol['openpi_commit']
for name,digest in protocol['scripts_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
checks={'training_verified':True,'fresh_process_restore_verified':True,'finite_training_metrics':100,'evaluation_episodes':100,'evaluation_successes':s['successes'],'videos_verified':100,'evaluation_infrastructure_errors':0,'experiment_source_hashes_match_protocol':True}
(root/'results/final-checks.json').write_text(json.dumps(checks,indent=2))
selected=[
 'results/train-smoke','results/experiment-protocol.json','results/final-checks.json','results/eval-libero-spatial-seed7-trials10','results/first-libero-success.mp4','results/first-libero-success-final-frame.png',
 'assets/pi05_libero_lora_smoke','config/libero','config/libero-requirements-resolved.txt','env.sh','env-model.sh','env-libero.sh',
 'scripts/pi05_experiment.py','scripts/serve_libero_policy.py','scripts/evaluate_libero.py','scripts/check-training-results.py','scripts/run-train-smoke.sh','scripts/run-policy-server.sh','scripts/run-libero-evaluation.sh','scripts/download-server-ranges.py','scripts/verify-official-checkpoint.py','scripts/export-results.py',
 'logs/openpi-commit.txt','logs/openpi-packages.txt','logs/libero-packages.txt','logs/train-norms.log','logs/train-smoke.log','logs/policy-server.log','logs/libero-evaluation.log','logs/pi05_base-asset-list.json','logs/pi05_libero-asset-list.json','logs/checkpoint-server-ranges.log','logs/gpu-final-status.txt',
 'data/libero-subset/subset-manifest.json','data/libero-subset/meta'
]
files=[]
for relative in selected:
 p=root/relative
 files.extend(sorted(x for x in p.rglob('*') if x.is_file()) if p.is_dir() else [p])
manifest=[]
for p in files:
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 manifest.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':digest})
manifest_path=root/'results/artifact-manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2));files.append(manifest_path)
archive=root/'results/reproduction-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in files:tar.add(p,arcname=p.relative_to(root),recursive=False)
print('FINAL_CHECKS_PASS',json.dumps(checks),flush=True)
print('EXPORT_READY',archive,archive.stat().st_size,flush=True)
