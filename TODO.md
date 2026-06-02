# TODO

## Completed
- (Existing) check-valve refinement stage improvements scaffolded in `Backend/Backend/local_detection.py`

## Next
- [ ] Strengthen check-valve refinement in `Backend/Backend/local_detection.py`:
  - [x] Replace IOU-only promotion with distance/size-based “near evidence” scoring to template valve hits.
  - [x] Add nearby OCR valve-tag cue gate (`_VALVE_TAG_RE`).
  - [x] Ensure promoted valves get confidence bumped to survive final `active_thresh` filtering.
- [ ] Smoke test with existing local runner (`Backend/Backend/run_sample_detect.py`).

