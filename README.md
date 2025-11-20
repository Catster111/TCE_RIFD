# TCE_RIFD
Face detection and landmark localization using TCE

The research addresses the significant drop in face detection accuracy caused by rotational variations, especially out-of-plane (ROP) rotations which distort landmarks and lead to inaccurate face center localization. Current methods often handle in-plane rotation (RIP) but fail with ROP because they don't capture the structural relationships between facial features. Additionally, existing datasets suffer from unreliable landmark annotations and imbalanced rotational data.

To solve these issues, the authors propose:

A Topology-Guided Semantic Face Center Estimation Method: Uses graph-based landmark relationships to maintain structural integrity under both RIP and ROP.
A New Rotation-Aware Face Dataset: Features accurate face center annotations and balanced data across various rotation angles, specifically designed for training under extreme poses.
A Hybrid-ViT Model: Combines CNNs (for spatial features) and Transformers (for topological reasoning), enhanced by a center-refinement module.
Robust Localization Technique: Uses the accurately predicted centers as anchors for RoI alignment and polar transformation to handle extreme rotations effectively.
A Hybrid Evaluation Metric: Assesses the quality of the predicted center based on topological consistency and feature alignment in the polar domain.
Experiments show that this comprehensive approach outperforms current state-of-the-art models in cross-dataset evaluations.


RAF Dataset Download link --> https://drive.google.com/drive/folders/1h7hjm7qrB_JJys0Qf2wzkbmRHF2phZq7?usp=sharing
