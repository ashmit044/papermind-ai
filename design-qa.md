# Paper workspace visual QA

## Reference and implementation

- Selected visual direction: option 2, bright research workbench.
- Reference image: `assets/screenshots/design_reference_option_2.png` (1487 × 1058 px; selected direction image shared in the design conversation).
- Implementation evidence: `assets/screenshots/paper_question_sources.png` (1440 × 1600 px), captured from the running Streamlit app using the explicitly approved Playwright verification.
- Side-by-side comparison: `assets/screenshots/design_comparison.png`.
- Browser: Chrome, desktop viewport 1440 × 1600 px; responsive check at 390 × 844 px.
- State: synthetic demo paper, submitted question, local offline synthesizer, three retrieved page/section citations.

## Comparison and iteration

The implementation follows the chosen direction's light paper-like surfaces, restrained blue accent, serif display headings, quiet borders, whitespace, and digest-plus-source-rail hierarchy. The initial implementation put Q&A on a separate tab; visual review identified that as a mismatch, so the digest, question form, answer, and source rail now share one Paper workspace.

The reference depicts a ready-made environmental paper and populated sources before a question. Those contents are illustrative and are not generated from PaperMind. To avoid presenting invented research or retrieval results as real, the implementation instead labels its sample as synthetic and leaves the source rail empty until a question is submitted. The selected visual structure is retained while those example data are intentionally not copied.

## Checks

- Desktop question submission: answer rendered and three source expanders displayed with page and section labels.
- Phone viewport: document width remained 390 px at a 390 px viewport; no horizontal overflow.
- Concept graph: rendered in the separate graph tab during the preceding UI check.
- Browser JavaScript and console errors: none observed.
- Open P0/P1/P2 visual issues: none. Existing upstream deprecation notices in the app's PDF and Plotly calls are outside this UI change.

Final result: passed
