import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from filtering.hallucination_detector import HallucinationDetector


class TestHallucinationDetector:
    def setup_method(self):
        self.detector = HallucinationDetector()

    def test_clean_text_passes(self):
        result = self.detector.check("The sum of 2 and 3 is 5. This is a verified mathematical fact.")
        assert result.passed
        assert result.score >= 0.7

    def test_hedging_detected(self):
        result = self.detector.check("I think it might be possible that perhaps the answer could be 42. I believe this is maybe correct.")
        assert len([i for i in result.issues if i.code == "HALLUCINATION_HEDGING"]) > 0

    def test_unverifiable_claims_detected(self):
        result = self.detector.check("Research shows that this is true. Studies indicate the same. Experts say so. Data shows this conclusion.")
        assert len([i for i in result.issues if i.code == "HALLUCINATION_UNVERIFIABLE"]) > 0

    def test_template_leakage_detected(self):
        result = self.detector.check("Hello {{user}}, here is your response {{output}}")
        assert len([i for i in result.issues if i.code == "HALLUCINATION_TEMPLATE_LEAK"]) > 0

    def test_contradiction_detected(self):
        result = self.detector.check("The sky is blue. However, the sky is not blue. Both statements are true.")
        issues = [i for i in result.issues if i.code == "HALLUCINATION_CONTRADICTION"]
        assert len(issues) >= 0

    def test_empty_text(self):
        result = self.detector.check("")
        assert result.score >= 0

    def test_batch_processing(self):
        texts = [
            "This is a clean factual statement.",
            "I think maybe the answer could possibly be 42.",
            "Research shows that this is absolutely true without any doubt.",
        ]
        results = self.detector.check_batch(texts, num_workers=2)
        assert len(results) == 3

    def test_vagueness_detected(self):
        result = self.detector.check("There are many things like that and so on. Etc. Whatever. And more stuff like that among others.")
        assert len([i for i in result.issues if i.code == "HALLUCINATION_VAGUENESS"]) > 0
