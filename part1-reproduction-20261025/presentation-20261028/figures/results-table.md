# Teaching extension: all ranks reported

Three training-subset seeds; fixed reused 1,000-image test set; mean +/- sample SD.

| CSA rank | Core scalars | Budget PCA dim | CSA acc (%) | Equal-latent PCA acc (%) | Budget PCA acc (%) | CSA test MSE | Equal-latent PCA test MSE | Budget PCA test MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 20.53 +/- 1.68 | 23.43 +/- 1.42 | 10.00 +/- 0.00 | 0.04795 +/- 0.00097 | 0.04669 +/- 0.00023 | 0.05236 +/- 0.00004 |
| 2 | 4 | 2 | 43.83 +/- 0.91 | 50.03 +/- 1.02 | 31.13 +/- 0.90 | 0.03988 +/- 0.00009 | 0.03547 +/- 0.00007 | 0.04237 +/- 0.00020 |
| 3 | 9 | 5 | 72.60 +/- 1.31 | 77.40 +/- 0.75 | 61.13 +/- 1.65 | 0.03089 +/- 0.00004 | 0.02466 +/- 0.00016 | 0.03252 +/- 0.00015 |
| 4 | 16 | 9 | 80.17 +/- 1.44 | 83.73 +/- 0.31 | 77.40 +/- 0.75 | 0.02384 +/- 0.00003 | 0.01758 +/- 0.00012 | 0.02466 +/- 0.00016 |
| 5 | 25 | 15 | 85.17 +/- 1.43 | 85.20 +/- 1.25 | 83.27 +/- 0.55 | 0.01610 +/- 0.00003 | 0.01225 +/- 0.00004 | 0.01836 +/- 0.00018 |
| 6 | 36 | 22 | 85.77 +/- 0.87 | 85.53 +/- 0.25 | 84.83 +/- 1.15 | 0.01080 +/- 0.00000 | 0.00829 +/- 0.00011 | 0.01381 +/- 0.00012 |
| 7 | 49 | 30 | 85.20 +/- 0.50 | 84.33 +/- 0.65 | 85.17 +/- 0.81 | 0.00713 +/- 0.00001 | 0.00549 +/- 0.00003 | 0.01020 +/- 0.00009 |

For CSA r=1, the storage budget cannot fit one PCA component; d=0 is mean-only reconstruction.
Its 1-NN prediction has all-distance ties, resolved to the first training label (10% here).
Storage counts N=300 training-image cores, bases and the shared mean; test metrics do not enter this budget.
This is not a same-byte compressed file comparison or a reproduction of original Fig. 5.
