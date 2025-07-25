> This package is still under active development. We are actively adding features and the API are still not stable.

# flashscenic

SCENIC is a powerful tool for inferring gene regulatory networks (GRNs) from observational single-cell data, but today its application is often suffered from conflicting software versions and the computational cost of GRN inference, which can take hours or days on high-performance clusters. Here, we introduce `flashSCENIC`, a GPU-accelerated workflow that replaces the bottleneck step with our diffusion model-based RegDiffusion, and includes a GPU-powered AUCell calculation. This new pipeline runs in seconds instead of hours and makes the GRN analysis scalable to 20,000 genes and millions of cells. This workflow can also effectively correct for batch effects during data integration while preserving biological signals, as reported in SCENIC. You can also adjust network granularity from broad lineages to specific subtypes by tuning hyperparameter k. 
