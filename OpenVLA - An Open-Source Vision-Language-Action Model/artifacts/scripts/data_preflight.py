import json,os,time
from pathlib import Path
import train_smoke as m
root=Path(os.environ["OPENVLA_ROOT"])
m.AutoConfig.register("openvla",m.OpenVLAConfig)
m.AutoImageProcessor.register(m.OpenVLAConfig,m.PrismaticImageProcessor)
m.AutoProcessor.register(m.OpenVLAConfig,m.PrismaticProcessor)
processor=m.AutoProcessor.from_pretrained(root/"models/base",trust_remote_code=False)
cfg=m.AutoConfig.from_pretrained(root/"models/base",trust_remote_code=False)
transform=m.RLDSBatchTransform(m.ActionTokenizer(processor.tokenizer),processor.tokenizer,image_transform=processor.image_processor.apply_transform,prompt_builder_fn=m.PurePromptBuilder)
start=time.perf_counter()
dataset=m.RLDSDataset(root/"data","libero_spatial_no_noops",transform,resize_resolution=tuple(cfg.image_sizes),shuffle_buffer_size=256,image_aug=True)
collator=m.PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,processor.tokenizer.pad_token_id,padding_side="right")
batch=next(iter(m.DataLoader(dataset,batch_size=2,collate_fn=collator,num_workers=0)))
info={"status":"passed","seconds":time.perf_counter()-start,"shapes":{k:list(v.shape) for k,v in batch.items() if isinstance(v,m.torch.Tensor)},"supervised_tokens":(batch["labels"]!=-100).sum(dim=1).tolist(),"statistics_keys":list(dataset.dataset_statistics)}
assert all(x>0 for x in info["supervised_tokens"])
assert m.torch.isfinite(batch["pixel_values"]).all().item()
(root/"results/data_preflight.json").write_text(json.dumps(info,indent=2))
print(json.dumps(info),flush=True)
