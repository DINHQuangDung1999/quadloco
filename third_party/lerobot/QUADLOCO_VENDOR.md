# Vendored LeRobot

This directory is a source-vendored copy of Hugging Face LeRobot `0.4.4`.

- Upstream repository: `https://github.com/huggingface/lerobot`
- Upstream commit: `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`
- License: Apache-2.0; see `LICENSE`

Quadloco carries a compatibility fix in
`src/lerobot/policies/pi05/modeling_pi05.py` that maps the converted PI0.5
`lm_head.weight` tensor to PaliGemma's tied token-embedding key. This lets the
pinned PI0.5 checkpoint load strictly under LeRobot 0.4.4.

The vendored files are ordinary Quadloco repository files. This directory has
no nested Git metadata and does not require a separate LeRobot clone.
