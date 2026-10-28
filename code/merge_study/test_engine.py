"""State replacement and independent evaluator equivalence on a tiny Qwen."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from engine import DeltaEvaluator


class StateReplacement(unittest.TestCase):
    def test_no_drift_or_head_carryover(self):
        from transformers import Qwen2Config, Qwen2ForSequenceClassification
        torch.manual_seed(7)
        model = Qwen2ForSequenceClassification(Qwen2Config(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, pad_token_id=0, num_labels=2))
        key = "model.layers.0.self_attn.q_proj.weight"
        original = dict(model.named_parameters())[key].detach().clone()
        a, b = torch.full_like(original, .1), torch.full_like(original, -.2)
        heads = {"a": torch.ones_like(model.score.weight), "b": -torch.ones_like(model.score.weight)}
        engine = DeltaEvaluator(model, {"a": {key: a}, "b": {key: b}}, heads, "cpu")
        for _ in range(50):
            engine.set_state("a", {key:a})
            engine.set_state("b", {key:b})
        torch.testing.assert_close(dict(model.named_parameters())[key], original + b, rtol=0, atol=0)
        torch.testing.assert_close(model.score.weight, heads["b"], rtol=0, atol=0)
        engine.set_state("a", {key: torch.zeros_like(a)})
        torch.testing.assert_close(dict(model.named_parameters())[key], original, rtol=0, atol=0)
        torch.testing.assert_close(model.score.weight, heads["a"], rtol=0, atol=0)
        average = engine.merge({"method":"average","scale":1.,"density":1.})
        summed = engine.merge({"method":"task_arithmetic","scale":.5,"density":1.})
        torch.testing.assert_close(average[key], summed[key])


if __name__ == "__main__":
    unittest.main()
