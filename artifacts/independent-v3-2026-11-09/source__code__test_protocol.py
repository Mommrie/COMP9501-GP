"""Protect sentence roles under truncation and preserve archived protocol behavior."""
import unittest
from common import encode_example


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))

    def num_special_tokens_to_add(self, pair=False):
        return 0

    def prepare_for_model(self, ids, **kwargs):
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def __call__(self, *texts, truncation=True, max_length=128):
        return self.prepare_for_model(self.encode("".join(texts))[:max_length])


class InputProtocol(unittest.TestCase):
    def setUp(self):
        self.tok = CharacterTokenizer()
        self.cfg = {"input_format": "sentence_fields_v2", "max_length": 64}

    def decode(self, result):
        return "".join(map(chr, result["input_ids"]))

    def test_pair_has_roles_and_boundary(self):
        result = encode_example("rte", self.cfg, self.tok,
                                {"sentence1": "A cat sleeps.", "sentence2": "An animal sleeps."})
        self.assertEqual(self.decode(result), "Sentence A: A cat sleeps.\nSentence B: An animal sleeps.")

    def test_long_first_sentence_cannot_erase_second(self):
        result = encode_example("mrpc", self.cfg, self.tok,
                                {"sentence1": "a" * 1000, "sentence2": "b" * 1000})
        text = self.decode(result)
        self.assertEqual(len(result["input_ids"]), 64)
        self.assertIn("Sentence A: a", text)
        self.assertIn("\nSentence B: b", text)
        self.assertLessEqual(abs(text.count("a") - text.count("b")), 1)

    def test_missing_format_reproduces_legacy(self):
        result = encode_example("rte", {"max_length": 64}, self.tok,
                                {"sentence1": "left", "sentence2": "right"})
        self.assertEqual(self.decode(result), "leftright")

    def test_single_sentence_and_small_budget(self):
        result = encode_example("sst2", self.cfg, self.tok, {"sentence": "good"})
        self.assertEqual(self.decode(result), "Text: good")
        with self.assertRaises(ValueError):
            encode_example("rte", {**self.cfg, "max_length": 5}, self.tok,
                           {"sentence1": "a", "sentence2": "b"})


if __name__ == "__main__":
    unittest.main()
