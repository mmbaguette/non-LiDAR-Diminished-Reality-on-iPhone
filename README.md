# non-LiDAR-Diminished-Reality-on-iPhone
A diminished reality (DR) app that reprojects camera pixels and uses image inpainting to make real 3D objects disappear on an iPhone augmented reality (AR) app.

Removing a real object from a live camera view, convincingly, completely offline,
on an iPhone 13. 

This experienced taught me to use coding agents to code and design platforms myself, that would typically require a team of experienced software developers to develop.

Documentation only — no source is published. Implementation was written with AI
assistance. What is mine, and what this document records, is the constraint
choice, the model and approach selection, the experimental design, the
measurements, and the thermal and memory engineering. Every figure here was
measured on device unless marked as offline.

## Contents

1. [Problem and constraints](#1-problem-and-constraints)
2. [Pipeline](#2-pipeline)
3. [Model and approach selection](#3-model-and-approach-selection)
4. [Thermal and memory budgeting](#4-thermal-and-memory-budgeting)
5. [Evaluation](#5-evaluation)
6. [Licensing and patents](#6-licensing-and-patents)
7. [Retrospective](#7-retrospective)
8. [Skills](#8-skills)
9. [Benchmarks](#9-benchmarks)

---

## 1. Problem and constraints

Diminished reality is the inverse of AR: remove a real object and show what is
behind it. It has to stay gone as the camera moves, the revealed surface has to
look like a surface, and the illusion has to survive something passing in front.

I picked constraints that remove the usual shortcuts:

- **No network.** Everything on device. Rules out server-side inpainting, which
  is how most convincing demos of this are made.
- **iPhone 13 — no LiDAR.** The central constraint. Every piece of geometry
  must be inferred from one moving RGB camera plus device motion. LiDAR phones
  get this free.
- **4 GB RAM device.** The process is killed near **2.1 GB** of footprint. Weights,
  full-resolution frames and cached results all compete inside that.
- **60 fps preview**, regardless of what else runs.
- **Commercially clean.** Every shipped component must be usable commercially
  without negotiation, with no budget for model training or legal counsel.
  This removed the best-performing inpainting model from contention (§3, §6).

---

## 2. Pipeline

**At removal:** resolve the tap to a mask of a segmented foreground object on the screen; find the surface it
rests on; synthesise the hidden background once; project that background onto the surface
in world space so it stays put as the camera moves.

**Every frame after:** track the object, so it is never drawn back over its own
replacement; composite anything passing in front; replace invented pixels with
real ones as motion reveals the surface; correct for the camera's changing
exposure; blend the seam.

At 60 fps there are 16.7 ms per frame, and camera
tracking already takes part of that. So the steady-state loop is kept to a
reprojection and a composite, and everything expensive — segmentation,
inpainting, surface finding — happens as a one-shot event at the tap, where a
few seconds is acceptable. Re-synthesis of the background happens rarely, on a
large viewpoint change, never continuously, because each replacement risks
visible drift.

Surfaces are detected planes. An object can span several, so the synthesised
background is split across all the planes it covers. A tall object viewed at an
angle hides more of the floor behind it than its footprint, so the replaced
region is extended by a margin derived from object height and viewing angle.


---

## 3. Model and approach selection

### Segmentation

**Rejected Apple Vision alone.** It is the obvious choice: free, fast, no
conversion. It is a detector, not a tracker — no memory, no concept of which
object the user picked. On device it returned nothing on **45% of passes**, and
often returned the wrong object. Everything built to compensate was scaffolding
around that.

**Selected EdgeTAM** (Segment Anything 2 for edge devices) for tracking, and
kept Vision for one job: resolving the initial tap to an instance. A measured
comparison found Vision better at that specific task, so both are used where
each is stronger. EdgeTAM's decisive property is that it is promptable and
stateful — told once which object was chosen, it holds that object across
frames, and it reports absence as distinct from failure. Acting on that
distinction is what stops the object being painted into its own replacement.

Two mask requirements came out of inpainting tests rather than segmentation
tests (§5): the mask has to be grown to swallow adjacent fine detail such as
printed text, and masks of neighbouring objects have to be subtracted so the
fill does not eat an object that is only partly covered.

These were the coding agent's design choices based on the testing I peformed.

### Inpainting

The quality reference and the commercial candidates are different methods, and
this is the section where that matters most.

I compared candidates offline on still photographs of real scenes before
anything ran on the phone. Criteria: quality on large holes over texture and
over structure (floor edges, material joins), and whether the method *and* its
weights could ship commercially.

| Method | Result | Status |
| --- | --- | --- |
| Classical diffusion (Telea, Navier–Stokes) | Unacceptable blur on large holes over texture | Rejected on quality |
| Criminisi exemplar-based | Worked well on the test scenes | Commercial candidate |
| OpenCV shift-map (`xphoto`) | Worked well; produced artefact lines next to fine text unless the mask was grown | Commercial candidate |
| LaMa | Best quality; handles large masks and periodic structure | Reference only — weights not commercially clean |
| MI-GAN | Light AI inpainter; poor, distracting infills in each test | Reference only — same training-data issue as LaMa |
| PatchMatch | slow, but effective | Rejected — patent live to ~2029 |
| G'MIC | unacceptable quality compared to other patch matching algorithms  | Rejected — copyleft licence blocks closed-source use |
| PixMix (Herling & Broll) | Not implemented as a fill method. | Rejected — see §6 |

Classical fills are cheap and unencumbered but reproduce texture rather than
structure. LaMa reproduces structure, which is what floors and walls mostly
are — and cannot ship (§6). So LaMa is kept as the quality ceiling: the thing
the commercial fills are measured against, not the thing tuned for.

The on-device inpainting figures in §9 are LaMa, converted to Core ML at three
resolutions.

The intended auto-selection startegy between the two commercial fill algorithms, Criminisi and OpenCV shift-map, 
is driven by the pixels around
the hole: gradient energy and local variance measured on the known ring of
pixels around the mask choose between diffusion (smooth surroundings) and an
exemplar method (repeating texture). Though this hasn't been implemented yet.

### Depth

[**Depth Anything V2 Small** is Apache 2.0, the Large variant CC-BY-NC
and therefore unusable commercially. The scope is deliberately narrow — depth
is needed *once*, at removal, to establish which surface an object stands on 
(this solves the undetected surfaces problem, see (§5). A
one-shot question needs no temporal stability and no per-frame budget, which is
a far easier problem.

These were the coding agent's design, but I'm currently rethinking my entire inpainting strategy. 

---

## 4. Thermal and memory budgeting

The hardest part of the project, and where measurement repeatedly overturned
reasoning — including mine.

### The constraint that shapes the rest

**The Neural Engine is fp16-only.** Any model kept at fp32 for numerical
reasons runs on CPU and GPU instead. That one fact determined most of the
thermal behaviour below.

### Where the memory goes

The process dies near **2.1 GB**. Inpainting weights are **196–207 MB each**;
the five-stage tracker is **76 MB at fp32**, **40 MB at fp16**. Materialised and
specialised for a compute unit they occupy considerably more than file size. One
removal takes the process from a **155–272 MB** idle baseline to **866 MB**,
settling near 640 MB.

The assumption worth discarding early was that retained image data — cached
frames, textures, debug imagery — was the bulk. It is not; those are tens of
megabytes. Weights dominate by roughly an
order of magnitude.

### Isolating the heat source

Above the lowest thermal state the system clamps clocks and everything slows at
once: rendering halves from **60 to 30 fps**, inference roughly triples. That
makes every timing suspect, and I lost a day to it before logging thermal state
alongside every measurement.

**I isolated the cause by A/B rather than inference.** The tracker's output
feeds two features; with both switched off, so the tracker does not run between
removals, the thermal state barely climbs. With either on it reaches `serious`
in roughly three removals.

The first version of that experiment was invalid, and catching why mattered
more than the result: the debug overlay — a diagnostic that *draws* the
tracker's output — was in the same condition that decided whether the tracker
*ran*. A diagnostic was causing the work it was meant to observe. Left as-is,
switching both features off would have shown no change and I would have
concluded the tracker was innocent.

### The result that inverted my expectation

Switching the tracker to fp16 made each pass **2.4× faster** and did not make
the device cooler. I was prepared to conclude fp16 did not help thermals.

Counting invocations instead of reading the temperature showed the opposite.
The tracking loop was single-flight and nothing more — it relaunched the instant
the previous pass returned. **Single-flight is not a throttle.** Making the
model faster did not reduce work; it raised the rate until something else
became the limit. Across two matched runs, fp16 ran **~780 passes where fp32
ran ~180** — four times the tracking for slightly more heat.

So fp16 is far more efficient per unit of work, and the application was spending
the entire gain on frequency. It also means **comparing two precisions without a
rate cap compares nothing**, because each saturates whatever it is capable of. A
**2 Hz ceiling** fixed both. fp32 sits below that ceiling unaided, so the cap
only binds the faster configuration — the correct asymmetry.

### Duty cycle

The geometry the tracker maintains lives in world space and the rendering that
consumes it is a function of camera pose, so a camera that has not moved cannot
change either. Gating on translation *or* rotation — turning in place moves the
camera nowhere while changing everything the occlusion depends on — cut tracker
invocations roughly **4×** in handheld use.

This was the coding agent's decision.

### Load scheduling

Neural Engine weight compilation is device-specific — it targets a particular
Neural Engine revision, so it cannot be done on a development machine and
shipped. The fp16 set costs **13.6–17.0 s** cold and **6.0 s** with the
device's compilation cache warm, against **1.5–2.2 s** for fp32. Loading it in
the background at launch moves that into time the user already spends aiming
the camera.

A measurement artefact hid the real number for several rounds: **reinstalling
the app invalidates that cache**, so every timing taken right after an install
reads cold. The warm figure only appeared once a build was relaunched without
reinstalling.

This was the coding agent's design.

### Outcome

From the first removal, both starting from the lowest thermal state:

| | fp32 | fp16 |
| --- | --- | --- |
| Time to `serious` | ~3:30 | ~6:00 |
| Passes in that window | ~180 | ~780 |
| Peak footprint | 931 MB | 677 MB |

Twice the endurance for four times the work, at lower peak memory — the latter
with *more* removals performed, which is the opposite of what plate count would
predict, leaving the larger fp32 weights as the explanation.

Two runs, different objects, different removal counts — individually solid, not
a controlled comparison. The invocation counts make the direction unambiguous;
the absolute times are weaker.

The device also sheds heat about as fast as it gains it: down a level within
~10 s, full frame rate within a minute.

---

## 5. Evaluation

Two stages. Offline, on still photographs, to choose an inpainting method
before running anything on the phone. Then manual, on device, against real
scenes. 

### Offline inpainting comparison

Scenes: a wood deck, wallpaper, an office desk with printed text, and a cluster
of plant pots — chosen to cover periodic texture, fine high-contrast detail and
objects crowding the target.

- **Diffusion broke** on every large hole over texture — a smooth smear.
- **Shift-map produced artefact lines next to text.** The fix was on the mask,
  not the fill: grow the mask until it swallows adjacent fine detail.
- **Fills ate partly covered neighbours.** In the plant-pot scene the fill
  sampled from, and overwrote, pots touching the target. Subtracting
  neighbouring objects' masks fixed it.
- **Criminisi and shift-map held up** on the test scenes; LaMa was better on
  structure, as expected.

### On device

**The test that mattered most was object variation**. Simple convex objects —
cups, bottles, boxes — worked consistently. Flicker appeared
only on a lotion bottle with a pump on top and on headphones. By varying object shape
while holding everything else fixed, I established the failure tracked
*geometry*, not the recent changes I had been blaming. 

That killed the representation: a single convex volume cannot express a band
arching over an open gap, or a wide body with a narrow nozzle. No margin makes a
convex hull concave — increasing it swallowed more scene while still leaving the
object visible through its own replacement. Replacing it with a per-cell height
model carved from successive segmentations resolved both, verified visually:
complete coverage, no artefacts.

The height ladder algorithm was my coding agent's design. Here's a nice ASCII visual of what's going on with the lotion bottle:
'''
        ##          nozzle
        ##
    ########
    ########        body
    ########
  ──────────────    support plane


       /##\
     /######\       hull bridges
   /##########\     nozzle to body
   ##############
  ──────────────
     ^^^^      ^^^^
     swallows scene on both sides

   cell:  1  2  3  4  5  6
   h:     0  0  9  9  0  0
                ^
          nozzle cell is tall,
          body cells are short,
          empty cells are zero
'''

**Held up — multiple planes**, including scenes with **17 detected planes** and
the object spanning 3. **Held up — thermal degradation and recovery.**

**Broke — undetected surfaces.** An object on a
table the device never detected produces a removal projected onto the *floor*.
Signature: **one detected plane**, zero surfaces rejected, object height
measured at **0.45–0.61 m** — its height above the floor, not the table.

I spent two iterations fixing surface *selection* before establishing that
selection was working correctly and the correct surface under the object wasn't being detected to be
selected. The gates I added were right and changed nothing. Distinguishing "the
logic chose wrongly" from "the input was absent" is the lesson, and it is what
started the depth work.

**Broke — objects near the target.** Nearby surfaces were getting their own
removal region on ray votes alone. The separating signal is whether the object
*rises above* that surface: one standing a centimetre above a plane is behind
it, not on it, and the rays that voted passed the object rather than being
blocked by it.

**Broke — varying lighting.** I traced the plates glowing to the exposure
correction, and the failure is a feedback loop: the gain is measured against the
region it then writes into, so a bright region measures bright, is written
brighter, and measures brighter still until a clamp stops it. The correct reference is the
surrounding photographed content, not the output. Disabled and documented as
broken rather than removed.

---

## 6. Licensing and patents

I constrained the work to components usable commercially without negotiation,
applied from the start. License checks were done per model/algorithm
at selection time, not deferred to release: a commercial blocker found late costs the
development built on top of it. This is a record of engineering decisions, not
legal analysis.

**Model weights are licensed separately from model code — and training data
separately from both.** A permissive repository licence says nothing about the
weights it downloads. Two cases from this project:

- One depth model is Apache 2.0 in its small variant and CC-BY-NC (Creative Commons Non-Commercial) in its large
  one — same architecture, same authors, same repository, one of them unusable.
- LaMa's code is permissively licensed, but its weights are not stored in its
  repository; they are downloaded from a third-party host. They were trained on
  Places2, whose Creative Commons licence covers the dataset authors'
  classification models, not the images — the dataset states that copyright in
  the images stays with their owners. I checked this against the dataset's own
  pages after being told, by another AI assistant, that the weights were clean
  because the dataset was Creative Commons. The claim had the licence's scope
  backwards. MI-GAN has the same training-data problem.

Check the checkpoint and what it was trained on, not the project.

**Algorithmic patents outlive the papers.** Several well-known inpainting
techniques were patented by their originating institutions. Being published,
widely implemented, and shipped inside a popular open-source library does not
settle the question, though those three facts are often cited as though they do.
I restricted selection to methods whose patents are expired, lapsed, or were
never granted, verified against patent records:

| Method | Patent | Status |
| --- | --- | --- |
| Criminisi exemplar-based | US6987520B2 (Microsoft) | Lapsed 2018-02-12 for non-payment of maintenance fees |
| Shift-map (Pritch / Peleg) | US8249394B2 (Yissum) | Lapsed 2020-09-28 for non-payment of maintenance fees |
| Shift-map as implemented in OpenCV `xphoto` (cites He & Sun 2012) | US20140369622A1 (Microsoft) | Application abandoned; never granted |
| PixMix-related (Herling & Broll) | US9412188B2 | Lapsed 2024-09-16 |
| PixMix-related (Herling & Broll) | US8660305B2 | Lapsed 2026-03-30 |
| PatchMatch (Adobe) | - | Live to ~2029 — not used |

**Lapsed is not the same as expired.** A patent that has run its full term is
finished. One that lapsed for unpaid fees can, in some circumstances, be
reinstated. I accepted lapses that are years old and treated PixMix as
off-limits because its most recent family member lapsed months before I
checked. It cost little: only three ideas from the paper were used (§3), none
of them its fill method.

**Implementation licences are a separate check from method patents.** The
shift-map patent being clear says nothing about the code. The OpenCV module
that implements it carries a 3-clause BSD notice in its header inside a
repository licensed Apache 2.0. Both permit commercial use; only Apache 2.0
includes an explicit patent grant.

---

## 7. Retrospective

**Measurement beat reasoning, repeatedly, and I learned to distrust the
reasoning first.** A performance problem was attributed to build configuration
and written up as such; the configuration had always been correct, and the real
cause was per-frame geometry work on the main thread. It was confirmed by a
symptom nobody predicted — a settings menu freezing on tap, in a part of the app
with nothing to do with rendering. Separately, a numerical drift figure (0.98
correlation at fp16 against 7e-5 at fp32) predicted visible tracking
degradation twice on EdgeTAM. It never appeared: that figure describes error appearing
along a continuous track, and every real session had removals in between, each
resetting the model's memory. The figure was right; the inference about real
usage was not.

**Under-instrumented for too long.** Conclusions had to be discarded because the
logs could not support them. Thermal state without timings are uninterpretable. Log lines without elapsed time cannot answer
"how long until it throttles," and I caught that question being answered anyway,
from a line count times an assumed interval. Both were one-line fixes that
should have existed before the first measurement.

**Testing don't always represent baseline performance.** A debug overlay defaulting to on meant every
session, including every measurement session, paid for it — and because it was on
during all of them, its cost to memory and thermal state looked like baseline performance. 

**What I would do differently:** instrument thermal state, elapsed time and
invocation counts on day one; test the non-convex object first, since it
invalidated an entire geometric representation and everything built on it;
check whether a required input exists before improving the logic that selects
among inputs; and make the shippable component the default from the first build
so the reference model can only be reached manually.

---

## 8. Skills

- **Agentic coding** to achieve things that would typically require a team of expert engineers.
- **Computer vision under hard constraints** — segmentation, stateful tracking,
  inpainting, plane estimation, occlusion, and the trade-offs between them on a
  fixed compute and memory budget.
- **Model selection on measured behaviour** rather than published benchmarks,
  including rejecting the obvious choice with evidence, and separating the
  quality reference from what can ship.
- **Mobile neural network deployment** — Core ML conversion, weights precision
  trade-offs, and Neural Engine constraints.
- **Performance engineering under thermal and memory limits** — duty cycling,
  rate limiting, load scheduling, and the instrumentation that makes any of it
  measurable.
- **AR and 3D geometry** — camera projection, homography, plane detection,
  raycasting, coordinate spaces, volume from silhouettes.
- **Licensing and IP as a design constraint** — patent searching, implementation
  licences, and weight and training-data checked separately and
  against primary sources.

---

## 9. Benchmarks

iPhone 13 (A15, 4 GB, no LiDAR), release builds.

### Platform

| Metric | Value |
| --- | --- |
| Footprint budget before termination | ~2.1 GB |
| Idle footprint | 155–272 MB |
| After one removal | 866 MB, settling ~640 MB |
| Render rate, lowest two thermal states | 60 fps |
| Render rate, `serious` | 30 fps |
| Recovery from `serious` | ~10 s down a level, ~1 min to full rate |

### Tracker, fp32 vs fp16

| Metric | fp32 | fp16 |
| --- | --- | --- |
| Image encoding | ~175 ms | ~38 ms |
| Memory attention | ~240 ms | ~90 ms |
| Mask decoding | ~50 ms | ~45 ms |
| Memory write | ~35 ms | ~35 ms |
| **Per pass** | **~500 ms** | **~208 ms** |
| Model set on disk | 76 MB | 40 MB |
| Cold load | 1.5–2.2 s | 13.6–17.0 s |
| Warm load | 2-2.5 s | 6.0 s |
| Passes in ~9.5 min, uncapped | ~180 | ~780 |
| Effective rate, uncapped | 0.31 Hz | 1.37 Hz |
| First removal to `serious` | ~3:30 | ~6:00 |
| Peak footprint | 931 MB | 677 MB |
| Correlation vs reference | 7e-5 match | 0.98 |
Note: ANE (Apple Neural Engine is designed to support FP16 weights. FP32 counterintuitively takes less time to load because it didn't load on ANE but only CPU.

Per-stage fp32 sizes: 22, 21, 19, 9, 5 MB.

### Inpainting — reference model (LaMa)

| Resolution | Weights | Cold | Thermally throttled |
| --- | --- | --- | --- |
| 512 | 196 MB | 2.97 s | 7.24 s |
| 800 | 207 MB | 8.56 s | 24.6–29.9 s |
| 1024 | 196 MB | to-do | to-do |

800 and 1024 are restricted to devices above 4 GB. 800
peaks at **1.2–1.4 GB** against a ~2.1 GB budget and triggers system memory
warnings mid-inference, causing ANE to abandon processing.

### Inpainting — commercial candidates

| Method | Time at 512 |
| --- | --- | --- |
| Criminisi | 0.7-2 s |
| Shift-map | 1.5-2 s |
Note: The thermal state (a hot iPhone) dramatically affects processing time.

### Removal stages

| Stage | Time |
| --- | --- |
| Frame conversion | 4–32 ms |
| Instance segmentation (Vision) | 127–343 ms |
| Surface raycast | 3–10 ms |
| Full removal at 512 (LaMa) | 3.5–7.6 s |
| Composite and render | 82–344 ms |

### Segmentation quality

| Metric | Value |
| --- | --- |
| Model-reported confidence, typical | 0.96–0.99 |
| Close-range failure | 0.34 confidence, 0.4% of frame |
| Vision passes returning nothing | 45% |

---
