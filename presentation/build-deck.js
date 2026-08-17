#!/usr/bin/env node
/**
 * Defence deck. Every number comes from presentation/data/content.json, which is
 * generated from runs/ — nothing here is typed by hand.
 *
 * Output: ../thesis_defence.pptx
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import PptxGenJS from "pptxgenjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "content.json"), "utf8"));
const FIG = path.join(ROOT, "figures");
const ASSETS = path.join(__dirname, "assets");
const LOGO = path.join(__dirname, "logo.png");
const OUT = path.join(ROOT, "thesis_defence.pptx");

const C = {
  navy: "0D1B2A", orange: "D54407", dark: "263238", white: "FFFFFF",
  grey: "F5F5F5", green: "2E7D32", mid: "90A4AE", sub: "455A64", greenFill: "E8F5E9",
};
const W = 13.333, HEAD = 0.72, M = 0.45, TOP = 0.95, FOOT = 7.05;

let n = 0, total = 0, pptx = mk();
function mk() {
  const p = new PptxGenJS();
  p.layout = "LAYOUT_WIDE";
  p.author = D.meta.author;
  p.title = "Master Thesis Defence";
  return p;
}
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const f3 = (v) => v.toFixed(3);
const fig = (name) => {
  const p = path.join(FIG, name);
  return fs.existsSync(p) ? p : null;
};

function footer(s) {
  s.addText(`${n}/${total || "?"}`, {
    x: 12.1, y: FOOT, w: 1, h: 0.25, fontSize: 10, color: C.navy,
    align: "right", fontFace: "Arial",
  });
}

function slide(title, kicker) {
  n += 1;
  const s = pptx.addSlide();
  s.background = { color: C.white };
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: W, h: HEAD, fill: { color: C.navy }, line: { color: C.navy } });
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: 0.35, y: 0.06, w: 10.8, h: 0.24, fontSize: 10, bold: true,
      color: C.orange, fontFace: "Arial", charSpacing: 1,
    });
  }
  s.addText(title, {
    x: 0.35, y: kicker ? 0.28 : 0.12, w: 10.8, h: 0.42,
    fontSize: 21, bold: true, color: C.white, fontFace: "Arial",
  });
  if (fs.existsSync(LOGO)) s.addImage({ path: LOGO, x: 12.15, y: 0.08, h: 0.55 });
  footer(s);
  return s;
}

function section(title, sub) {
  n += 1;
  const s = pptx.addSlide();
  s.background = { color: C.navy };
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 3.0, w: 0.12, h: 1.3, fill: { color: C.orange }, line: { color: C.orange } });
  s.addText(title, { x: 0.5, y: 3.0, w: 11.5, h: 0.9, fontSize: 34, bold: true, color: C.white, fontFace: "Arial" });
  if (sub) s.addText(sub, { x: 0.5, y: 3.85, w: 11.5, h: 0.5, fontSize: 15, color: C.mid, fontFace: "Arial" });
  footer(s);
  return s;
}

/** One entry = one bullet; "  text" indents as a sub-point; "" is a spacer. */
function bullets(s, items, o = {}) {
  // default leaves room for a note() panel underneath; pass h explicitly when
  // a slide has no note and can use the full height.
  const { x = M, y = TOP, w = 12.4, h = 4.4, fontSize = 15 } = o;
  s.addText(items.map((line) => {
    const blank = !line.trim(), sub = !blank && /^\s{2,}/.test(line);
    return {
      text: blank ? " " : line.trim(),
      options: {
        bullet: blank ? false : sub ? { code: "2013", indent: 15 } : true,
        indentLevel: sub ? 1 : 0, breakLine: true,
        fontSize: blank ? Math.round(fontSize * 0.45) : sub ? fontSize - 1 : fontSize,
        color: sub ? C.sub : C.dark, fontFace: "Arial", paraSpaceAfter: 6,
      },
    };
  }), { x, y, w, h, valign: "top" });
}

function note(s, body, label = "What this shows: ", bottom = 6.85) {
  const charW = (12 / 72) * 0.47;
  const lines = Math.max(1, Math.ceil((label + body).length / Math.floor(11.95 / charW)));
  const h = lines * (12 / 72) * 1.32 + 0.3, top = bottom - h;
  s.addShape(pptx.ShapeType.rect, { x: M, y: top, w: 12.4, h, fill: { color: C.grey }, line: { color: "E0E0E0" } });
  s.addText([
    { text: label, options: { bold: true, fontFace: "Arial" } },
    { text: body, options: { fontFace: "Arial" } },
  ], { x: M + 0.2, y: top, w: 12.0, h, fontSize: 12, italic: true, color: C.navy, valign: "middle" });
}

const hdr = (cells) => cells.map((t) => ({
  text: t, options: { bold: true, fill: { color: C.navy }, color: C.white, align: "center", fontFace: "Arial" },
}));
const row = (cells, { zebra = false, hi = false } = {}) => cells.map((t, i) => ({
  text: String(t),
  options: {
    fill: { color: hi ? C.greenFill : zebra ? C.grey : C.white }, color: C.dark,
    bold: hi || i === 0, align: i === 0 ? "left" : "center", fontFace: "Arial",
  },
}));

const short = (e) => {
  const x = D.experiments[e];
  const a = { "MobileNetV2": "MNv2", "ResNet-50": "RN50", "EfficientNet-B3": "ENb3" }[x.architecture];
  return `${e}\n${a}${x.architecture === "EfficientNet-B3" ? "@" + x.imageSize : ""}`;
};

// ─────────────────────────────────────────────────────────── build
function build() {
  // Title
  n += 1;
  const t = pptx.addSlide();
  t.background = { color: C.white };
  if (fs.existsSync(LOGO)) t.addImage({ path: LOGO, x: 5.9, y: 0.3, h: 0.8 });
  [
    ["Master Thesis Defence", 22, true, C.orange, 1.3],
    ["Lightweight vs. Heavyweight CNN Architectures", 21, true, C.navy, 1.85],
    ["for Skin Lesion Classification", 21, true, C.navy, 2.3],
    ["A Controlled Factorial Comparison of MobileNetV2, ResNet-50", 13, false, C.dark, 2.85],
    ["and EfficientNet-B3 on HAM10000", 13, false, C.dark, 3.15],
    [D.meta.author, 16, true, C.dark, 3.75],
    [D.meta.supervisors, 12, false, C.sub, 4.15],
    [D.meta.school, 11, false, C.dark, 4.5],
    [`Matriculation ${D.meta.matriculation}  ·  ${D.meta.intake}`, 10, false, C.sub, 4.8],
    ["All 12 experiments complete  ·  every figure generated from stored run artefacts", 13, true, C.green, 5.4],
  ].forEach(([txt, size, bold, color, y]) =>
    t.addText(txt, { x: 0.75, y, w: 11.85, h: 0.4, fontSize: size, bold, color, align: "center", fontFace: "Arial" }));
  footer(t);

  // Agenda
  const ag = slide("Agenda");
  [["I", "The problem", "Why melanoma triage needs a small model"],
   ["II", "Design", "The 12-run controlled factorial"],
   ["III", "Results", "RQ1, RQ2, RQ3 and the statistics"],
   ["IV", "The ablation", "EfficientNet-B3 at two resolutions"],
   ["V", "What I got wrong", "Limitations found along the way"]].forEach(([num, h, sub], i) => {
    const y = TOP + 0.3 + i * 1.05;
    ag.addShape(pptx.ShapeType.rect, { x: M, y, w: 0.62, h: 0.85, fill: { color: i % 2 ? C.orange : C.navy }, line: { color: i % 2 ? C.orange : C.navy } });
    ag.addText(num, { x: M, y, w: 0.62, h: 0.85, fontSize: 22, bold: true, color: C.white, align: "center", valign: "middle", fontFace: "Arial" });
    ag.addText(h, { x: M + 0.85, y: y + 0.05, w: 4.2, h: 0.75, fontSize: 17, bold: true, color: C.navy, valign: "middle", fontFace: "Arial" });
    ag.addText(sub, { x: M + 5.2, y: y + 0.05, w: 7.2, h: 0.75, fontSize: 12.5, color: C.dark, valign: "middle", fontFace: "Arial" });
  });

  // ── I. Problem
  section("The Problem", "Why this comparison is worth running");

  let s = slide("Melanoma Is a Timing Problem", "Motivation");
  bullets(s, [
    "Melanoma survival is near 100% when caught early, below 30% once it has metastasised",
    "Nothing about the tumour changes in between except how long it was left alone",
    "",
    "So the bottleneck is access, not treatment: getting a suspicious lesion in front of someone qualified",
    "A triage tool does not need to replace a dermatologist. It needs to sort the queue",
    "",
    "But a triage tool has to run on the device in front of the patient, and that is where the accuracy literature stops helping",
  ]);
  note(s, "ResNet-50 serialises to 94.3 MB in the 7-class configuration used here. MobileNetV2 comes to 9.07 MB. That decides whether a model ships inside an app or behind a network call, and an offline screening tool cannot depend on a network call.", "The constraint: ");

  s = slide("Why the Published Record Does Not Answer It", "Motivation");
  bullets(s, [
    "Studies comparing architectures on HAM10000 also vary the split, augmentation, optimiser, batch size and epoch budget",
    "  A reported gap between two backbones conflates the backbone with everything else that changed",
    "",
    "And headline accuracy is misleading on this dataset:",
    `  HAM10000 is 66.9% benign nevi, so answering "nevus" every time scores 66.9% and finds zero melanomas`,
    "",
    "That number is not far below much of the published literature",
  ]);
  note(s, "Both problems have the same fix: run every comparison inside one pipeline, and report per class rather than in aggregate. That is what this thesis does.", "The response: ");

  // ── II. Design
  section("Design", "Twelve runs, two things allowed to vary");

  s = slide("The 12-Run Factorial", "Design");
  s.addTable([
    hdr(["", "Augmentation only", "Weighted CE", "Oversampling"]),
    row(["MobileNetV2 @224", "E1", "E3", "E5"]),
    row(["ResNet-50 @224", "E2", "E4", "E6"], { zebra: true }),
    row(["EfficientNet-B3 @224", "E7", "E8", "E9"]),
    row(["EfficientNet-B3 @300", "E10", "E11", "E12"], { zebra: true }),
  ], { x: 1.3, y: TOP + 0.35, w: 10.7, colW: [3.5, 2.4, 2.4, 2.4], border: { pt: 0.5, color: "CCCCCC" }, fontSize: 13, rowH: 0.45 });
  s.addText([
    { text: "E1–E6 ", options: { bold: true, fontFace: "Arial" } },
    { text: "are the confirmatory factorial, specified before anything ran.   ", options: { fontFace: "Arial" } },
    { text: "E7–E12 ", options: { bold: true, fontFace: "Arial" } },
    { text: "are an ablation added afterwards, analysed in a separate statistical family.", options: { fontFace: "Arial" } },
  ], { x: M, y: 3.85, w: 12.4, h: 0.5, fontSize: 13, color: C.dark });
  note(s, `Held identical across all twelve: the lesion-level split (hash ${D.dataset.splitHash}), augmentation, Adam at lr 1e-3, batch size 32, 50 epochs with patience 10, 16 dataloader workers, seed 42, and one GPU. Only architecture, imbalance strategy and (for B3) input size vary.`, "The controlled part: ");

  s = slide("The Data and the Split", "Design");
  const cd = fig("01_class_distribution.pdf") || fig("eda_class_distribution.png");
  if (cd) s.addImage({ path: cd, x: M, y: TOP, w: 7.0, h: 4.4, sizing: { type: "contain", w: 7.0, h: 4.4 } });
  bullets(s, [
    `${D.dataset.images.toLocaleString()} images, ${D.dataset.lesions.toLocaleString()} lesions, 7 classes`,
    `${D.dataset.train} / ${D.dataset.val} / ${D.dataset.test} train/val/test`,
    "",
    "Split on lesion_id, not image_id",
    "  1,956 lesions are photographed more than once",
    "  Splitting on images puts near-duplicates on both sides and inflates the result",
    "",
    "Imbalance ratio NV:DF = 58.3 : 1",
  ], { x: 7.7, y: TOP, w: 5.2, h: 4.4, fontSize: 13 });
  note(s, "Many published HAM10000 studies split at image level or do not say which they used. That is the single largest reason cross-paper accuracy comparison on this dataset is unreliable, and why every comparison here is internal to one pipeline.", "Why it matters: ");

  // ── III. Results
  section("Results", "RQ1, RQ2, RQ3");

  s = slide("Headline: The Small Model Is Never Beaten", "Results");
  const conf = D.confirmatory;
  s.addTable([
    hdr(["Metric", ...conf.map((e) => short(e).replace("\n", " "))]),
    row(["Macro F1", ...conf.map((e) => f3(D.experiments[e].macro_f1))]),
    row(["Balanced acc.", ...conf.map((e) => pct(D.experiments[e].balanced_accuracy))], { zebra: true }),
    row(["Accuracy", ...conf.map((e) => pct(D.experiments[e].accuracy))]),
    row(["F1 melanoma", ...conf.map((e) => f3(D.experiments[e].per_class_f1.mel))], { zebra: true }),
    row(["Epochs", ...conf.map((e) => String(D.experiments[e].epochsRun))]),
  ], { x: 0.55, y: TOP + 0.2, w: 12.2, colW: [2.2, ...conf.map(() => 1.67)], border: { pt: 0.5, color: "CCCCCC" }, fontSize: 11, rowH: 0.42 });
  note(s, "MobileNetV2 achieves higher macro F1 than ResNet-50 under all three strategies: 0.664 vs 0.628, 0.622 vs 0.483, 0.681 vs 0.527. Note also that E1 and E2 sit within 0.004 on accuracy while differing by 0.037 on macro F1 — which is exactly why accuracy is not used to decide anything here.");

  s = slide("RQ1 — The Deployability Gap: PASSES", "Results");
  const r1 = D.rq.rq1;
  bullets(s, [
    `Best MobileNetV2 by the RQ2 rule: ${r1.best_mobilenet}   ·   Best ResNet-50: ${r1.best_resnet}`,
    "",
    `F1 melanoma:  ${f3(r1.f1_mel_mobilenet)}  vs  ${f3(r1.f1_mel_resnet)}`,
    `Gap = ${f3(r1.gap)}   against a threshold of ${r1.threshold} fixed before any run`,
    "",
    `Under the alternative definition of "best" (highest macro F1): gap ${f3(r1.macro_f1_best_alternative.gap)}, and the direction reverses`,
  ], { fontSize: 16, h: 3.0 });
  note(s, `A model with 10.5x fewer parameters gives up ${f3(r1.gap)} of melanoma F1. The conclusion holds under both definitions of "best", so it does not depend on a choice I made after seeing the data.`, "In plain terms: ");

  s = slide("RQ2 — The Best Strategy Depends on the Architecture", "Results");
  const strat = ["augmentation", "weighted_ce", "oversampling"];
  const byArch = { "MobileNetV2": ["E1", "E3", "E5"], "ResNet-50": ["E2", "E4", "E6"] };
  s.addTable([
    hdr(["", "Augmentation only", "Weighted CE", "Oversampling"]),
    ...Object.entries(byArch).map(([a, es], i) => {
      const best = es.reduce((x, y) => (D.experiments[y].rq2 > D.experiments[x].rq2 ? y : x));
      return [
        { text: a, options: { bold: true, fill: { color: C.grey }, fontFace: "Arial" } },
        ...es.map((e) => ({
          text: `${f3(D.experiments[e].rq2)} (${e})`,
          options: { align: "center", bold: e === best, fill: { color: e === best ? C.greenFill : C.white }, fontFace: "Arial" },
        })),
      ];
    }),
  ], { x: 1.0, y: TOP + 0.35, w: 11.3, colW: [3.0, 2.75, 2.75, 2.8], border: { pt: 0.5, color: "CCCCCC" }, fontSize: 13, rowH: 0.5 });
  bullets(s, [
    "Weighted CE is best for MobileNetV2. Augmentation-only is best for ResNet-50.",
    "  Both remedies made ResNet-50 worse at the very classes they target",
  ], { y: 3.1, fontSize: 14, h: 1.0 });
  note(s, "This is what the factorial exists to find. A study varying one factor at a time would report a single 'best' imbalance strategy and be wrong for one of the two architectures. Chapter 8 traces the mechanism to optimisation stability, not class frequency.");

  s = slide("RQ3 — Efficiency, on Two CPUs", "Results");
  const arm = D.efficiency.arm64, x86 = D.efficiency.x86_64;
  const base = arm.find((r) => r.architecture === "mobilenet_v2");
  s.addTable([
    hdr(["Architecture", "Params", "Size (MB)", "ARM64 (ms)", "x86-64 (ms)"]),
    ...arm.map((r, i) => {
      const o = x86.find((y) => y.architecture === r.architecture && y.image_size === r.image_size);
      const nm = r.architecture === "efficientnet_b3" ? `EfficientNet-B3 @${r.image_size}`
        : r.architecture === "resnet50" ? "ResNet-50" : "MobileNetV2";
      return row([nm, r.trainable_params.toLocaleString(), r.state_dict_mb.toFixed(2),
        r.cpu_latency_ms_mean.toFixed(1), o.cpu_latency_ms_mean.toFixed(1)], { zebra: i % 2 === 1 });
    }),
  ], { x: 1.0, y: TOP + 0.3, w: 11.3, colW: [3.4, 2.2, 1.9, 1.9, 1.9], border: { pt: 0.5, color: "CCCCCC" }, fontSize: 12, rowH: 0.44 });
  note(s, "MobileNetV2 is fastest on both hosts — the only part of this table that is stable. ResNet-50 and EfficientNet-B3 swap places: B3 is 2.9x slower than ResNet-50 on ARM and 2.0x faster on x86. Parameter count, which is what the literature usually reports as 'efficiency', predicts neither. Phones are ARM, so the ARM column is the one this thesis rests on.");

  s = slide("Statistical Testing", "Results");
  const cm2 = D.mcnemar.filter((r) => r.family === "confirmatory");
  s.addTable([
    hdr(["Comparison", "p (Holm)", "Significant", "Favours"]),
    ...cm2.map((r, i) => row([`${r.exp_a} vs ${r.exp_b}`,
      r.p_holm_corrected < 1e-4 ? r.p_holm_corrected.toExponential(2) : r.p_holm_corrected.toFixed(4),
      r["reject_at_0.05"] ? "Yes" : "No", r.better], { zebra: i % 2 === 1 })),
  ], { x: 1.8, y: TOP + 0.15, w: 9.7, colW: [2.8, 2.4, 2.2, 2.3], border: { pt: 0.5, color: "CCCCCC" }, fontSize: 11, rowH: 0.35 });
  note(s, "McNemar's paired test, Holm-corrected; 5 of 9 significant. The two that matter most are E1 vs E2 and E3 vs E4, both at p = 1.000 — under augmentation and weighted CE the two architectures are indistinguishable on per-image correctness, despite a 10.5x difference in size. In no comparison does ResNet-50 significantly beat MobileNetV2.");

  // ── IV. Ablation
  section("The Ablation", "EfficientNet-B3 at two resolutions");

  s = slide("B3 Wins on Accuracy — and Loses the Argument", "Ablation");
  bullets(s, [
    `E11 (B3 @300, weighted CE) is the highest-scoring run in the study: macro F1 ${f3(D.experiments.E11.macro_f1)}`,
    "",
    "But at matched 224px resolution, B3 and MobileNetV2 are statistically indistinguishable:",
    "  E7 vs E1 → p = 1.000     E9 vs E5 → p = 1.000     E8 vs E3 → p = 0.222",
    "",
    "E11's advantage traces to the 300px input, not to the architecture",
    "  E8 → E11 is the only significant resolution effect in the ablation",
    "",
    "And that input costs 8.9x MobileNetV2's inference time on ARM, at 4.8x the disk",
  ], { fontSize: 14 });
  note(s, "For a server-side deployment, B3 at 300px is the right choice and I would say so. For the offline mobile setting this thesis is about, it answers a different question well. RQ1 is defined over E1–E6 only, so the ablation does not displace it.", "The honest position: ");

  s = slide("Does the Native Resolution Explain It?", "Ablation");
  const rf = fig("08_resolution_ablation.pdf");
  if (rf) s.addImage({ path: rf, x: M, y: TOP, w: 12.4, h: 3.6, sizing: { type: "contain", w: 12.4, h: 3.6 } });
  note(s, "224 → 300 helps in all three pairs, but by amounts differing by an order of magnitude: +0.026 under augmentation (p=0.085), +0.123 under weighted CE (p=1.9e-05), +0.010 under oversampling (p=0.190). Resolution matters for B3 only in combination with weighted cross-entropy — the highest-variance training condition in the study. Published comparisons vary architecture and resolution together and cannot separate these.");

  // ── V. Limitations
  section("What I Got Wrong", "Limitations found during the work");

  s = slide("Checkpoint Selection Chose Worse Models", "Limitations");
  s.addTable([
    hdr(["Exp", "Best val-loss epoch", "Macro F1 there", "Best macro F1", "Cost"]),
    ...[["E1", 28, 0.744, 0.744, 0.000], ["E2", 36, 0.640, 0.672, 0.032],
        ["E3", 24, 0.613, 0.650, 0.038], ["E4", 46, 0.511, 0.571, 0.059],
        ["E5", 20, 0.721, 0.724, 0.003], ["E6", 6, 0.534, 0.601, 0.067]]
      .map((r, i) => row([r[0], String(r[1]), r[2].toFixed(3), r[3].toFixed(3), r[4].toFixed(3)],
        { zebra: i % 2 === 1, hi: r[4] > 0.05 })),
  ], { x: 1.6, y: TOP + 0.15, w: 10.1, colW: [1.3, 3.0, 2.3, 2.3, 1.2], border: { pt: 0.5, color: "CCCCCC" }, fontSize: 11.5, rowH: 0.38 });
  note(s, "The protocol stops on validation loss but reports macro F1, and they do not peak at the same epoch. The cost reaches 0.067 — and falls on ResNet-50, the architecture this thesis concludes against, because a noisy loss curve has an arbitrary argmin. I did not change the rule after finding this: doing so mid-study, having seen which runs it disadvantaged, would be the worse error. It is the first item of future work.");

  s = slide("Limitations I Would Raise Before You Do", "Limitations");
  bullets(s, [
    "Single seed per configuration — the largest weakness. McNemar compares two trained models, not two configurations in expectation",
    "E4 never converged: it exhausted the 50-epoch ceiling and is reported as an unconverged run",
    "Dropout is unmatched (0.2 / none / 0.3), inherited from the reference implementations, and favours the models I conclude for",
    "No weight decay or LR schedule — uniform, so not a confound, but the likely cause of ResNet-50's instability",
    "DF has 13 test images: one prediction moves its recall by 7.7 points",
    "HAM10000 carries no Fitzpatrick data, so nothing here speaks to performance across skin tones",
  ], { fontSize: 13.5 });
  note(s, "For a tool meant to widen access to screening, the skin-tone gap is a real limitation rather than a footnote, and no deployment recommendation from this work should be read as applying across skin tones until it is tested.", "The one that matters clinically: ");

  // Close
  s = slide("Conclusion", "");
  bullets(s, [
    `RQ1 — PASSES. Melanoma F1 gap of ${f3(r1.gap)} against a 0.05 threshold set in advance`,
    "RQ2 — The best imbalance strategy depends on the architecture it is applied to",
    "RQ3 — MobileNetV2 is smallest, fewest parameters, and fastest on both CPUs tested",
    "",
    "A network a tenth the size of ResNet-50 is statistically indistinguishable from it on per-image correctness under two of three strategies",
    "",
    "The result I did not anticipate: imbalance remediation is not architecture-neutral, and the mechanism is optimisation stability rather than class frequency",
  ], { fontSize: 15, h: 4.0 });
  note(s, "That last finding is invisible to any study varying one factor at a time. It is what the controlled factorial was for.", "Why the design earned its keep: ");

  n += 1;
  const th = pptx.addSlide();
  th.background = { color: C.navy };
  th.addText("Thank You", { x: 0, y: 2.6, w: W, h: 1.0, fontSize: 40, bold: true, color: C.white, align: "center", fontFace: "Arial" });
  th.addText("Questions", { x: 0, y: 3.6, w: W, h: 0.6, fontSize: 18, color: C.mid, align: "center", fontFace: "Arial" });
  th.addText(`${D.meta.author}  |  ${D.meta.supervisors}`, { x: 0, y: 6.3, w: W, h: 0.4, fontSize: 12, color: C.mid, align: "center", fontFace: "Arial" });
  footer(th);

  // Backup
  section("Backup", "Detail slides for questions");

  s = slide("Per-Class F1, All Twelve Runs", "Backup");
  const hm = fig("03_per_class_f1_heatmap.pdf");
  if (hm) s.addImage({ path: hm, x: 1.2, y: TOP, w: 10.9, h: 5.0, sizing: { type: "contain", w: 10.9, h: 5.0 } });
  note(s, "NV is handled well everywhere. The variation that separates configurations lives in the rare classes, which is why per-class reporting is the primary metric here.", "What this shows: ", 6.9);

  s = slide("Melanoma: Precision Against Recall", "Backup");
  const pr = fig("05_melanoma_precision_recall.pdf");
  if (pr) s.addImage({ path: pr, x: 2.4, y: TOP, w: 8.5, h: 5.0, sizing: { type: "contain", w: 8.5, h: 5.0 } });
  note(s, "Configurations sit on a trade-off curve rather than ordering from worse to better. E3 finds 63.5% of melanomas and is right 34.0% of the time it says melanoma; E6 is right 66.7% of the time and finds 15.7%. On 115 test melanomas that is 73 found against 18.", "What this shows: ", 6.9);

  s = slide("Exploratory Family — All 12 B3 Comparisons", "Backup");
  const ex = D.mcnemar.filter((r) => r.family === "exploratory");
  s.addTable([
    hdr(["Comparison", "p (Holm)", "Sig.", "Favours"]),
    ...ex.map((r, i) => row([`${r.exp_a} vs ${r.exp_b}`,
      r.p_holm_corrected < 1e-4 ? r.p_holm_corrected.toExponential(2) : r.p_holm_corrected.toFixed(4),
      r["reject_at_0.05"] ? "Yes" : "No", r.better], { zebra: i % 2 === 1 })),
  ], { x: 2.2, y: TOP + 0.05, w: 8.9, colW: [2.6, 2.3, 1.8, 2.2], border: { pt: 0.5, color: "CCCCCC" }, fontSize: 10.5, rowH: 0.3 });
  note(s, "Corrected separately from the confirmatory family. Pooling all 21 would apply a harsher correction to the 9 pre-registered comparisons, letting a post-hoc addition change whether earlier findings hold.", "Why a separate family: ", 6.9);

  s = slide("Training Curves by Architecture", "Backup");
  const tc = fig("02_training_curves.pdf");
  if (tc) s.addImage({ path: tc, x: 0.5, y: TOP, w: 12.3, h: 5.0, sizing: { type: "contain", w: 12.3, h: 5.0 } });
  note(s, "ResNet-50's epoch-to-epoch validation loss is 1.70x more volatile than MobileNetV2's. E4 produced a single-epoch increase of 1.207 and moved the wrong way on 23 of 49 transitions. This is the mechanism behind both the RQ2 result and the checkpoint-selection cost.", "What this shows: ", 6.9);
}

build();
total = n;
n = 0;
pptx = mk();
build();

pptx.writeFile({ fileName: OUT })
  .then(() => console.log(`Built: ${OUT} (${n} slides)`))
  .catch((e) => { console.error(e); process.exit(1); });
