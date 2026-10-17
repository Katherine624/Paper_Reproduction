"""Short OpenVLA LoRA run using the official model, RLDS transforms and loss."""
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from torch.utils.data import DataLoader

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--steps",type=int,default=100)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--resume")
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--attention",default="flash_attention_2")
    args=p.parse_args()
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed);tf.random.set_seed(args.seed)
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    AutoConfig.register("openvla",OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig,PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig,PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig,OpenVLAForActionPrediction)
    processor=AutoProcessor.from_pretrained(args.model,trust_remote_code=False)
    base=AutoModelForVision2Seq.from_pretrained(args.model,torch_dtype=torch.bfloat16,
        attn_implementation=args.attention,low_cpu_mem_usage=True,trust_remote_code=False).to("cuda")
    if args.resume:
        model=PeftModel.from_pretrained(base,args.resume,is_trainable=True)
    else:
        model=get_peft_model(base,LoraConfig(r=32,lora_alpha=16,lora_dropout=0.0,
            target_modules="all-linear",init_lora_weights="gaussian"))
    model.print_trainable_parameters()
    params=[x for x in model.parameters() if x.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=5e-4)
    start=0
    if args.resume:
        state=torch.load(Path(args.resume)/"training_state.pt",map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        start=state["global_step"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"]);random.setstate(state["python_rng"])
        assert start<args.steps
        print(json.dumps({"event":"resumed","global_step":start,"optimizer_entries":len(optimizer.state),
            "data_stream":"restarted; exact iterator continuation is not claimed"}),flush=True)
    tokenizer=ActionTokenizer(processor.tokenizer)
    transform=RLDSBatchTransform(tokenizer,processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,prompt_builder_fn=PurePromptBuilder)
    dataset=RLDSDataset(Path(args.data),"libero_spatial_no_noops",transform,
        resize_resolution=tuple(base.config.image_sizes),shuffle_buffer_size=256,image_aug=True)
    save_dataset_statistics(dataset.dataset_statistics,root)
    collator=PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,padding_side="right")
    loader=iter(DataLoader(dataset,batch_size=args.batch_size,collate_fn=collator,num_workers=0))
    config=vars(args)|{"lora_rank":32,"lora_alpha":16,"learning_rate":5e-4,
        "trainable_parameters":sum(x.numel() for x in params),"dataset":"libero_spatial_no_noops"}
    (root/("config_resume.json" if args.resume else "config.json")).write_text(json.dumps(config,indent=2))
    sample_name,sample_param=next((n,x) for n,x in model.named_parameters() if x.requires_grad and "lora_B" in n)
    before=sample_param.detach().float().cpu().clone()
    step_times=[];first_batch=None
    log=(root/("resume_metrics.jsonl" if args.resume else "metrics.jsonl")).open("w",buffering=1)
    torch.cuda.reset_peak_memory_stats();model.train()
    def save(step):
        target=root/f"checkpoint-{step}";target.mkdir(exist_ok=True)
        model.save_pretrained(target);processor.save_pretrained(target)
        torch.save({"global_step":step,"optimizer":optimizer.state_dict(),"torch_rng":torch.get_rng_state(),
            "cuda_rng":torch.cuda.get_rng_state_all(),"numpy_rng":np.random.get_state(),
            "python_rng":random.getstate()},target/"training_state.pt")
        save_dataset_statistics(dataset.dataset_statistics,target)
        print(json.dumps({"event":"checkpoint","step":step,"path":str(target)}),flush=True)
    for step in range(start+1,args.steps+1):
        tic=time.perf_counter();batch=next(loader)
        if first_batch is None:
            first_batch={k:v.detach().cpu().clone() for k,v in batch.items() if isinstance(v,torch.Tensor)}
        inputs={"input_ids":batch["input_ids"].cuda(),"attention_mask":batch["attention_mask"].cuda(),
            "pixel_values":batch["pixel_values"].to("cuda",dtype=torch.bfloat16),"labels":batch["labels"].cuda()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):loss=model(**inputs).loss
        assert torch.isfinite(loss).item(),"Nonfinite loss"
        loss.backward()
        grads=[x.grad for x in params if x.grad is not None]
        assert grads and all(torch.isfinite(g).all().item() for g in grads),"Invalid gradients"
        grad_norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in grads])).item()
        assert grad_norm>0,"Zero gradient norm"
        optimizer.step();torch.cuda.synchronize()
        seconds=time.perf_counter()-tic;step_times.append(seconds)
        record={"step":step,"loss":loss.item(),"grad_norm":grad_norm,"seconds":seconds,
            "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated()}
        log.write(json.dumps(record)+"\n")
        if step%10==0 or step==start+1:print(json.dumps(record),flush=True)
        if step%50==0 or step==args.steps:save(step)
    difference=(sample_param.detach().float().cpu()-before).abs().max().item()
    assert difference>0,"No LoRA parameter update"
    model.eval()
    verify={k:v.cuda() for k,v in first_batch.items() if k in ["input_ids","attention_mask","pixel_values","labels"]}
    verify["pixel_values"]=verify["pixel_values"].to(torch.bfloat16)
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):verify_loss=model(**verify).loss.item()
    assert np.isfinite(verify_loss)
    # Generate a real action from the image and prompt, excluding supervised action tokens.
    prompt_mask=(first_batch["labels"][0]==-100) & first_batch["attention_mask"][0].bool()
    prompt_ids=first_batch["input_ids"][0][prompt_mask].unsqueeze(0).cuda()
    assert prompt_ids.shape[1]>0
    model.get_base_model().norm_stats=dataset.dataset_statistics
    norm_key=next(k for k in dataset.dataset_statistics if k.startswith("libero_spatial"))
    with torch.inference_mode():
        action=model.predict_action(input_ids=prompt_ids,
            pixel_values=first_batch["pixel_values"][:1].to("cuda",dtype=torch.bfloat16),
            unnorm_key=norm_key,do_sample=False)
    assert np.asarray(action).shape==(7,) and np.isfinite(action).all()
    summary={"start_step":start,"end_step":args.steps,"optimizer_steps_completed":args.steps-start,
        "tracked_parameter":sample_name,"max_parameter_update":difference,"verification_loss":verify_loss,
        "median_step_seconds":float(np.median(step_times)),"peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated(),
        "resumed_optimizer":bool(args.resume),"generated_action":np.asarray(action).tolist()}
    (root/("resume_summary.json" if args.resume else "summary.json")).write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    log.close()
if __name__=="__main__":main()
