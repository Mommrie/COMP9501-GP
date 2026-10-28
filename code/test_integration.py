"""Offline tiny-Qwen test: train/save/load/merge/probe without model downloads."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class TinyPipeline(unittest.TestCase):
    def test_training_and_materialized_merge(self):
        from datasets import Dataset
        from peft import PeftModel
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForSequenceClassification
        import train_lora
        from common import model_for
        from evaluate import load_evaluation_model
        from merge import adapter_deltas, merge_tensors
        from probe import collect

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix="comp9501-tiny-") as temp:
            root = Path(temp)
            base_path = root / "base"
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
                {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "good": 3, "bad": 4}, unk_token="[UNK]")),
                unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
            tokenizer.save_pretrained(base_path)
            base = Qwen2ForSequenceClassification(Qwen2Config(vocab_size=8, hidden_size=32,
                intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, num_labels=2, pad_token_id=2, eos_token_id=2))
            base.save_pretrained(base_path)
            cfg = {"model": str(base_path), "base_seed": 9501, "tasks": ["sst2"], "rank": 2,
                "alpha": 4, "dropout": 0., "target_modules": ["q_proj", "v_proj"],
                "train_samples": 8, "eval_samples": 8, "batch_size": 2,
                "gradient_accumulation_steps": 1, "epochs": 1, "learning_rate": .001,
                "head_learning_rate": .002, "weight_decay": 0., "selection_split": "calibration",
                "select_best_on_calibration": True, "diagnostics": True}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(cfg), encoding="utf-8")
            dataset = Dataset.from_dict({"input_ids": [[3, 4, 3], [4, 3, 4]] * 4,
                "attention_mask": [[1, 1, 1]] * 8, "labels": [1, 0] * 4})
            output = root / "run"
            def manifest(task, config, split, limit):
                return {"sample_ids": list(range(8)) if split == "train" else list(range(8, 16))}
            with patch.object(train_lora, "dataset_for", return_value=dataset), patch.object(
                train_lora, "dataset_manifest", side_effect=manifest), patch("sys.argv", [
                "train_lora.py", "--task", "sst2", "--config", str(config_path),
                "--output", str(output), "--max-steps", "1"]):
                train_lora.main()
            adapter = output / "adapter"
            run = json.loads((output / "run.json").read_text())
            self.assertEqual(run["train"]["epoch"], .25)
            self.assertEqual(run["selection_split"], "calibration")
            self.assertIsNone(run["validation"])
            self.assertIsNotNone(run["best_checkpoint"])
            self.assertGreater(run["parameter_change_from_initialization"]["head_l2"], 0.)
            self.assertGreater(run["parameter_change_from_initialization"]["lora_l2"], 0.)
            deltas = adapter_deltas(adapter)
            self.assertGreater(sum(float(t.abs().sum()) for t in deltas.values()), 0.)
            merged_file = root / "merged.pt"
            torch.save({"deltas": {k: merge_tensors([v, -v]) for k, v in deltas.items()},
                        "metadata": {"base_model": cfg["model"]}}, merged_file)
            merged = load_evaluation_model(cfg, str(adapter), str(merged_file)).cpu()
            individual = PeftModel.from_pretrained(model_for(cfg), str(adapter)).merge_and_unload().cpu()
            original = model_for(cfg).cpu()
            for key, value in deltas.items():
                torch.testing.assert_close(dict(merged.named_parameters())[key],
                                           dict(original.named_parameters())[key], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(merged.score.weight, individual.score.weight)
            batch = {"input_ids": torch.tensor([[3, 4, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
            features = collect(merged, [{**batch}], "cpu")
            self.assertEqual(tuple(features.shape), (1, 32))
            self.assertTrue(torch.isfinite(features).all())
            head_output = root / "head-only"
            with patch.object(train_lora, "dataset_for", return_value=dataset), patch.object(
                train_lora, "dataset_manifest", side_effect=manifest), patch("sys.argv", [
                "train_lora.py", "--task", "sst2", "--config", str(config_path),
                "--output", str(head_output), "--max-steps", "1", "--head-only"]):
                train_lora.main()
            head_run = json.loads((head_output / "run.json").read_text())
            self.assertGreater(head_run["parameter_change_from_initialization"]["head_l2"], 0.)
            self.assertTrue(all(torch.count_nonzero(v) == 0 for v in adapter_deltas(head_output / "adapter").values()))


if __name__ == "__main__":
    unittest.main()
