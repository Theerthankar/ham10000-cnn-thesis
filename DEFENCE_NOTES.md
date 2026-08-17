# Defence notes

An honest read of where this thesis is strong, where it can be pushed on, and what to say
when it is. Written against the twelve completed runs, not against intentions.

## The three questions most likely to be asked

### "Your ablation beat your main experiment. Why isn't EfficientNet-B3 the answer?"

This is the sharpest question available and it should be met head-on rather than
deflected. E11 does produce the highest macro F1 in the study (0.738 against the best
confirmatory run's 0.681).

Three things to say, in order:

1. At matched input resolution, B3 and MobileNetV2 are **not statistically
   distinguishable**. All three of those McNemar comparisons fail to reject, two at
   p = 1.000. E11's advantage comes from the 300-pixel input, and the resolution pairs
   confirm it: E8 to E11 is the only significant resolution effect in the ablation.
2. That advantage costs **8.9x MobileNetV2's inference time on ARM** and 4.8x its disk.
   The thesis asks what a deployable architecture gives up. B3 answers a different
   question well.
3. E7 to E12 were added after the confirmatory design was fixed and run. They live in a
   separate statistical family for exactly that reason. Promoting a post-hoc addition
   over a pre-registered design is the failure mode the two-family structure prevents.

The concession to make freely: **for a server-side deployment, B3 at 300px is the right
choice.** Saying so makes the rest of the argument more credible, not less.

### "You only trained each configuration once."

Concede immediately; it is the single largest weakness. The McNemar tests establish that
two *specific trained models* differ on per-image correctness. They do not establish that
two *configurations* differ in expectation across seeds.

What makes it defensible: twelve runs at a single seed was the budget available, and the
alternative was fewer conditions with more seeds. The factorial structure was judged the
more informative use of the compute, because it answers a question (does the best
imbalance strategy depend on the architecture) that no number of seeds on a smaller
design would reach.

Multiple seeds is the first item under Future Work for this reason.

### "Your accuracy is low compared to published HAM10000 results."

Do not get defensive. The best accuracy here is 0.801 against published claims of 93 to
99 per cent.

The answer is the lesion-level split. HAM10000 photographs 1,956 of its 7,470 lesions more
than once. Splitting on `image_id` puts near-duplicates on both sides of the train/test
boundary, and many published studies either do this or do not say which they did. This
thesis splits on `lesion_id`, which is stricter and produces lower, better-founded
numbers.

Then make the positive point: cross-paper accuracy comparison on HAM10000 is unreliable,
which is precisely why every comparison in this thesis is internal to one pipeline.

## Where the thesis is genuinely strong

- **The controlled protocol is real, not claimed.** One split verified by hash across all
  twelve runs, one optimiser, one schedule, one GPU. The split hash appears in every
  `config.json` and can be checked live.
- **Findings that a one-factor-at-a-time study cannot produce.** The RQ2 result
  (imbalance remediation is not architecture-neutral) requires the factorial. So does the
  resolution result.
- **Self-criticism that costs something.** Section 8.3 documents that checkpoint selection
  on validation loss cost up to 0.067 macro F1 and disadvantaged ResNet-50 specifically,
  which is the architecture the thesis concludes *against*. Volunteering a caveat that
  weakens your own conclusion is the strongest signal of good faith available.
- **Every number is generated.** Tables and figures come from the run artefacts by script.
  Offer to regenerate them live if anyone doubts a figure.

## Weaknesses to raise before anyone else does

Raising these first converts them from attacks into evidence of rigour.

| Weakness | What to say |
|---|---|
| Single seed | Largest limitation, acknowledged, first future-work item |
| Checkpoint selection cost up to 0.067 macro F1 | Found it, quantified it, did not change the protocol mid-study because that would be worse |
| E4 never converged (hit the 50-epoch ceiling) | Reported as an unconverged run, not silently included |
| Unmatched dropout (0.2 / 0.3 / none) | Inherited from reference implementations, favours the models the thesis prefers, declared |
| No weight decay or LR schedule | Uniform across runs so no comparison is confounded, but likely the cause of ResNet-50's instability |
| DF has 13 test images, VASC 15 | One prediction moves DF recall by 7.7 points; rare-class figures carry no argumentative weight |
| No skin-tone stratification | HAM10000 has no Fitzpatrick data. For a tool meant to widen access to screening this is a real gap, not a footnote |
| E2's notebook lost its execution record | Results intact in `runs/E2/`; disclosed in the notebook itself |

## Numbers worth having memorised

- Split: 8,012 / 979 / 1,024 images from 5,976 / 747 / 747 lesions, hash `f35ac1de18678182`
- Class imbalance: NV 66.95%, DF 1.15%, ratio 58.3:1; class weights span 0.214 to 12.578
- Naive always-NV classifier: 66.95% accuracy, zero melanomas detected
- RQ1: E3 vs E2, melanoma F1 0.442 vs 0.480, gap **0.037** against a 0.05 threshold
- RQ1 robustness: under macro-F1-best (E5 vs E2) the gap is 0.027 and reverses direction
- RQ2: weighted CE for MobileNetV2 (0.648), augmentation-only for ResNet-50 (0.516)
- RQ3: 2.23M vs 23.52M params, 9.07 vs 94.30 MB, 14.2 vs 31.1 ms on ARM
- Statistics: 5 of 9 confirmatory significant, 4 of 12 exploratory
- E1 vs E2 and E3 vs E4 both **p = 1.000** — indistinguishable at a 10x size difference

## The one-sentence version

A network a tenth the size of ResNet-50 is statistically indistinguishable from it on
melanoma detection under two of three imbalance strategies, and the best imbalance
strategy turns out to depend on which architecture you apply it to.
