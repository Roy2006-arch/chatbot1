import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from filtering.repetition_detector import RepetitionDetector


class TestRepetitionDetector:
    def setup_method(self):
        self.detector = RepetitionDetector()

    def test_clean_text_passes(self):
        result = self.detector.check("This is a diverse text with many different words and phrases that do not repeat themselves unnecessarily.")
        assert result.passed
        assert result.score >= 0.7

    def test_ngram_repetition_detected(self):
        text = "the cat in the hat the cat in the hat the cat in the hat the cat in the hat the cat in the hat"
        result = self.detector.check(text)
        assert len([i for i in result.issues if i.code == "REPETITION_NGRAM"]) > 0

    def test_filler_words_detected(self):
        text = "um so like basically you know um well i mean like actually so anyway um"
        result = self.detector.check(text)
        assert len([i for i in result.issues if i.code == "REPETITION_FILLERS"]) > 0

    def test_sentence_repetition_detected(self):
        text = "This is the first sentence. This is the first sentence. This is the first sentence."
        result = self.detector.check(text)
        assert len([i for i in result.issues if i.code == "REPETITION_SENTENCE"]) > 0

    def test_chunk_repetition_detected(self):
        text = (
            "This paragraph is repeated multiple times across different sections of the text.\n\n"
            "This paragraph is different and unique content here.\n\n"
            "This paragraph is repeated multiple times across different sections of the text.\n\n"
            "This paragraph is repeated multiple times across different sections of the text.\n\n"
            "This paragraph is repeated multiple times across different sections of the text."
        )
        result = self.detector.check(text)
        assert len([i for i in result.issues if i.code == "REPETITION_CHUNK"]) > 0

    def test_empty_text(self):
        result = self.detector.check("")
        assert not result.passed

    def test_batch_processing(self):
        texts = [
            "Clean text with variety.",
            "Repeat repeat repeat repeat repeat",
            "Um like so you know basically filler words",
        ]
        results = self.detector.check_batch(texts, num_workers=2)
        assert len(results) == 3

    def test_short_text_no_repetition(self):
        result = self.detector.check("Hello world")
        assert result.passed
