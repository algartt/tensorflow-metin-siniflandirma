import unittest

from data import SAMPLES, split_samples


class DataTests(unittest.TestCase):
    def test_dataset_is_balanced(self):
        positives = sum(label for _, label in SAMPLES)
        self.assertEqual(positives, len(SAMPLES) - positives)

    def test_split_has_no_overlap(self):
        train, validation = split_samples(seed=7)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(len(train) + len(validation), len(SAMPLES))

    def test_split_keeps_both_classes(self):
        train, validation = split_samples()
        self.assertEqual({label for _, label in train}, {0, 1})
        self.assertEqual({label for _, label in validation}, {0, 1})

    def test_invalid_ratio(self):
        with self.assertRaises(ValueError):
            split_samples(1.0)


if __name__ == "__main__":
    unittest.main()
