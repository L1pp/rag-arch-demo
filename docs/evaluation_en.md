# Existing Results and Metric Boundaries

[English](evaluation_en.md) | [简体中文](evaluation.md)

These figures are excerpts from existing run and evaluation summaries; the public demo has not rerun the experiments. Only anonymized aggregate results are published here. Original standards, plans, questions, reference answers, and model requests are not included. The sample demo does not reproduce the evaluations below, and the public code does not automatically inherit these historical results.

| Evaluation target | Fixed scope and method | Representative results | What the results support |
|---|---|---|---|
| Clause retrieval | Historical regression set of 60 annotated questions; BM25/Dense Top-30 each, RRF constant 60, 30 fused candidates, and cloud reranking over the same candidate pool | Recall@8 0.8444→0.8722; nDCG@8 0.8012→0.8185; complete evidence after enrichment 44/60→47/60 | Improved Top-8 evidence ranking and coverage on this question set; the set was used during system selection, and this is a single-round result |
| Public benchmark Q&A, AECBench | Fixed 240 questions, same model, single round; scored by decodable answer choice | Baseline 146/240 (60.83%), RAG 184/240 (76.67%); 57 questions changed from wrong to right, 19 from right to wrong | Benefit from retrieval-assisted answer-choice selection on the selected tasks; this is not whole-plan report accuracy |
| Plan-review development comparison | Five plans and a fixed set of 59 development reference items; Flash/Max under A/B/C conditions | Flash A/B/C: 34/59, 39/59, 40/59; Max A/B/C: 32/59, 40/59, 39/59 | Complete-hit coverage of the fixed reference items, not the detection rate for all defects or the correctness rate of the full report |

Do not report only favorable results: complete evidence at Top-4 fell from 40/60 to 39/60 in the same retrieval comparison. Of the 480 AEC outputs, 37 did not strictly follow the required format but still had a decodable answer choice. If strict JSON and a correct answer are both required, the baseline and RAG results are 140/240 and 179/240, respectively. This 240-question subset was observed over multiple rounds; it is not a new independent blind test set.

The whole-plan review used Qwen3.8 Flash/Max. A is the native first review; B is a native independent review of the same A; C is a local-standards-enhanced independent review of that same A. B is not an input to C. Observe the RAG increment between C and B for the same model; do not attribute the entire change from A to C to RAG. In this batch, C used 30 candidates, RRF constant 60, and reranking Top-4; it is separate from the Top-8 retrieval regression above. Real reviews still have issues such as object confusion, omissions across sections, and evidence being retrieved but not correctly used in the final recommendation. Results cannot be compared directly with papers or other systems unless model, prompt, samples, context, and scoring conditions are aligned.

## Source Register for the Public Summary

- Retrieval: retrieval-regression summary in the 2026-09-28 core materials v2.
- Q&A: AEC fixed-240-question summary in the 2026-10-09 four-benchmark results package.
- Whole-plan review: 59-item development-reference basis in the 2026-10-12 five-plan factual summary; kept separate from the older 88-item batch.

The original supporting materials remain on the author’s local machine. The public version contains only the readable figures above, so the scores cannot be independently recalculated or audited from this repository. The authored demo materials verify the input-to-report execution path; they do not verify historical task-quality improvements.