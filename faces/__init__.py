"""faces — Unified face recognition package (Stack B).

Core modules:
    engine      SCRFD detector, ArcFace embedder, alignment, ONNX session cache
    tracker     IoU-based face tracker with majority-vote smoothing
    store       Milvus face store abstraction (+ in-memory impl for tests)
    persons     YOLOv8n person detector
    annotate    Video annotation with face overlays
    service     Library-level operations (ingest, merge, onboard, tag, search)
"""
