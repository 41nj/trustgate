# Third-party notices

The MIT license in `LICENSE` covers the original TrustGate source code only. The following
third-party material is included or referenced and remains under its own terms.

## MMLU-Pro (data/mmlu_pro_sample.json, question text in evaluations/)

- Source: TIGER-Lab, *MMLU-Pro*, https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
  (revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`), described in Wang et al. (2024),
  "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark",
  https://arxiv.org/abs/2406.01574.
- License: MIT, as declared in the dataset card metadata (`license: mit`). The dataset repository
  contains no separate license file or copyright line; copyright remains with the dataset authors
  (TIGER-Lab). The MIT license text applies with that attribution:

  > Permission is hereby granted, free of charge, to any person obtaining a copy of this software
  > and associated documentation files (the "Software"), to deal in the Software without
  > restriction, including without limitation the rights to use, copy, modify, merge, publish,
  > distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
  > Software is furnished to do so, subject to the following conditions: The above copyright
  > notice and this permission notice shall be included in all copies or substantial portions of
  > the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  > IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
  > PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
  > LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
  > OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  > DEALINGS IN THE SOFTWARE.

- MMLU-Pro aggregates questions from MMLU, TheoremQA, SciBench and STEM websites. Review the
  dataset card before reuse beyond research and evaluation.
- The separate MMLU-Pro *code* repository (github.com/TIGER-AI-Lab/MMLU-Pro) is Apache-2.0
  licensed. TrustGate does not include or derive from that code.

## Gemma model outputs (evaluations/)

The predictions in `evaluations/` were generated with Google's Gemma model and are subject to the
Gemma Terms of Use (https://ai.google.dev/gemma/terms). Only the answer letter and the stated
confidence are published; full model responses are not.
