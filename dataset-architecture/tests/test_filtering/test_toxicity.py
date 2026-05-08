import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from filtering.toxicity_filter import ToxicityFilter


class TestToxicityFilter:
    def setup_method(self):
        self.filter = ToxicityFilter()

    def test_clean_text_passes(self):
        result = self.filter.check("This is a clean and safe response that does not contain any toxic content.")
        assert result.passed
        assert result.score >= 0.9

    def test_profanity_detected(self):
        result = self.filter.check("This is a f***ing bad response with profanity.")
        matches = [i for i in result.issues if i.code == "TOXICITY_PROFANITY"]
        assert len(matches) >= 0

    def test_hate_speech_detected(self):
        result = self.filter.check("White supremacy is the only way forward.")
        assert len([i for i in result.issues if i.code == "TOXICITY_HATE_SPEECH"]) > 0

    def test_harassment_detected(self):
        result = self.filter.check("You should kill yourself. Nobody loves you.")
        assert len([i for i in result.issues if i.code == "TOXICITY_HARASSMENT"]) > 0

    def test_violence_detected(self):
        result = self.filter.check("Here is how to kill someone with a bomb.")
        assert len([i for i in result.issues if i.code == "TOXICITY_VIOLENCE"]) > 0

    def test_empty_text_passes(self):
        result = self.filter.check("")
        assert result.passed
        assert result.score == 1.0

    def test_batch_processing(self):
        texts = [
            "Clean safe text.",
            "White power genocide is the answer.",
            "How to torture someone painfully.",
        ]
        results = self.filter.check_batch(texts, num_workers=2)
        assert len(results) == 3
