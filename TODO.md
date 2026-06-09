# TODO

- [x] Add SCADA/HMI vs P&ID diagram classifier to `Backend/Backend/local_detection.py` using OCR keywords + visual heuristics.
- [x] Update counting behavior for SCADA/HMI: still count motor/pump/valve/tank but use conservative rules to avoid UI/legend inflation.
- [x] Expose `diagram_type` ("PID" | "SCADA_HMI") and ensure counts are returned accordingly.
- [ ] If required by UI/API, update `Backend/Backend/yolo.py` response model to include `diagram_type`.
- [x] Ensure existing pipeline remains backward compatible for P&ID.
- [ ] Sanity-check logic against provided sample image `70642920-a5c7-41df-8e90-26d479e7b4b3.png` (no terminal; via code path).
