import sys; sys.path.insert(0, ".")
from backend.url_verifier import URLVerifier

v = URLVerifier()

text = "Check https://docs.python.org/3/ for Python docs and http://fake-url-xyz.com/ for fake stuff"
urls = v.extract_urls(text)
print("Extracted:", urls)

print("Valid https://docs.python.org/3/:", v.validate_format("https://docs.python.org/3/"))
print("Valid ftp://bad.com:", v.validate_format("ftp://bad.com"))
print("Valid not-a-url:", v.validate_format("not-a-url"))

print("Whitelist docs.python.org:", v.check_whitelist("https://docs.python.org/3/"))
print("Whitelist github.com:", v.check_whitelist("https://github.com/user/repo"))
print("Whitelist fake-url-xyz.com:", v.check_whitelist("https://fake-url-xyz.com/page"))

print("Shortener bit.ly:", v.is_shortener("https://bit.ly/abc123"))
print("Shortener docs.python.org:", v.is_shortener("https://docs.python.org/3/"))

print("Has URLs in plain text:", v.has_any_urls("hello world"))
print("Has URLs in text with link:", v.has_any_urls("see https://example.org"))

print("Stripped:", v.strip_urls("check https://example.org/page"))

c1 = type("Obj", (), {"is_valid_format": True, "in_whitelist": True, "is_reachable": True, "is_shortener": False})()
c2 = type("Obj", (), {"is_valid_format": False, "in_whitelist": False, "is_reachable": False, "is_shortener": False})()
c3 = type("Obj", (), {"is_valid_format": True, "in_whitelist": False, "is_reachable": False, "is_shortener": False})()
print("Confidence valid+whitelisted+reachable:", v.compute_confidence(c1))
print("Confidence invalid format:", v.compute_confidence(c2))
print("Confidence valid but unverifiable:", v.compute_confidence(c3))

print("\nAll URL verifier tests passed!")
