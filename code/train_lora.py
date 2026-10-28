"""Train task-specific LoRA plus task-specific classification head."""
import argparse
import hashlib
import time
from pathlib import Path

from common import dataset_for, dataset_manifest, environment, metrics, model_for, read_config, seed_all, tokenizer_for, write_json


def main():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--head-only", action="store_true", help="Freeze LoRA at zero and train only the task head")
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.task not in cfg["tasks"]:
        parser.error("Task must be listed in config")
    out = Path(args.output)
    if (out / "adapter").exists():
        raise FileExistsError(f"Refusing to overwrite existing adapter: {out / 'adapter'}")
    tokenizer = tokenizer_for(cfg)
    model = model_for(cfg)
    seed_all(args.seed)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=cfg["rank"],
        lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"], target_modules=cfg["target_modules"],
        bias="none", modules_to_save=["score"]))
    if args.head_only:
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(False)
    model.print_trainable_parameters()
    train = dataset_for(args.task, cfg, tokenizer, "train", cfg["train_samples"])
    selection_split = cfg.get("selection_split", "validation")
    selection_limit = cfg.get("calibration_eval_samples", cfg["eval_samples"])
    selection = dataset_for(args.task, cfg, tokenizer, selection_split, selection_limit)
    train_manifest = dataset_manifest(args.task, cfg, "train", cfg["train_samples"])
    selection_manifest = dataset_manifest(args.task, cfg, selection_split, selection_limit)
    if selection_split == "calibration" and set(train_manifest["sample_ids"]) & set(selection_manifest["sample_ids"]):
        raise ValueError("Training/calibration IDs overlap")
    write_json(out / "data-manifest.json", {"train": train_manifest, "selection": selection_manifest})
    diagnose = cfg.get("diagnostics", False)
    select_best = cfg.get("select_best_on_calibration", False)
    if select_best and selection_split != "calibration":
        raise ValueError("Best-checkpoint selection is permitted only on calibration")
    options = TrainingArguments(output_dir=str(out / "checkpoints"), seed=args.seed, data_seed=args.seed,
        per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
        num_train_epochs=cfg["epochs"], max_steps=args.max_steps,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="epoch", save_total_limit=2,
        eval_strategy="epoch" if select_best else "no", logging_steps=cfg.get("logging_steps", 1),
        report_to="none", dataloader_num_workers=0,
        warmup_ratio=cfg.get("warmup_ratio", 0.0),
        load_best_model_at_end=select_best, metric_for_best_model="loss", greater_is_better=False)
    optimizer_pair = (None, None)
    if "head_learning_rate" in cfg:
        groups = []
        for is_head, learning_rate in [(True, cfg["head_learning_rate"]), (False, cfg["learning_rate"])]:
            params = [p for name, p in model.named_parameters() if p.requires_grad and (("score" in name) == is_head)]
            if params:
                groups.append({"params": params, "lr": learning_rate, "weight_decay": cfg["weight_decay"]})
        optimizer_pair = (torch.optim.AdamW(groups), None)
    trainer = Trainer(model=model, args=options, train_dataset=train, eval_dataset=selection,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics,
                      processing_class=tokenizer, optimizers=optimizer_pair)
    initial_selection = trainer.evaluate(metric_key_prefix="initial") if diagnose else None
    initial_parameters = {name: p.detach().float().cpu().clone() for name, p in model.named_parameters()
                          if p.requires_grad} if diagnose else {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=args.resume)
    seconds = time.perf_counter() - start
    evaluation = trainer.evaluate()
    train_evaluation = trainer.evaluate(eval_dataset=train.select(range(min(128, len(train)))),
                                         metric_key_prefix="train_probe") if diagnose else None
    parameter_change = {"head_l2": 0.0, "lora_l2": 0.0}
    for name, before in initial_parameters.items():
        difference = model.get_parameter(name).detach().float().cpu() - before
        key = "head_l2" if "score" in name else "lora_l2"
        parameter_change[key] += float(difference.square().sum())
    parameter_change = {key: value ** .5 for key, value in parameter_change.items()} if diagnose else None
    trainer.save_model(str(out / "adapter"))
    tokenizer.save_pretrained(out / "adapter")
    write_json(out / "run.json", {"task": args.task, "seed": args.seed, "config": cfg,
        "max_steps": args.max_steps, "head_only": args.head_only, "train_samples": len(train),
        "selection_split": selection_split, "selection_samples": len(selection),
        "train_seconds": seconds, "train": result.metrics, "selection": evaluation,
        "initial_selection": initial_selection, "train_probe": train_evaluation,
        "parameter_change_from_initialization": parameter_change,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "validation": evaluation if selection_split == "validation" else None,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ["common.py", "train_lora.py"]},
        "environment": environment(), "classification_head": "task-specific; never merged"})
    print(f"RUN_COMPLETE: {out / 'run.json'}")


if __name__ == "__main__":
    main()
