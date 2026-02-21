# Replication

These scripts replicate the results in Table 1 of the [deep recurrent optimal stopping paper][2] for the deep optimal stopping (DOS) and deep neural network optimal stopping policy gradient (DNN-OSPG) methods on the Bermudan max-call option scenario described in the [deep optimal stopping paper][1]. The method implementations are our independent replications in PyTorch based on (1) the method descriptions in the papers, and (2) the [TensorFlow implementations][3] of the methods provided by the authors of the deep recurrent optimal stopping paper.

In addition, a script runs the *classifier-based* deep optimal stopping method proposed in our paper, where the original loss (which is linear in the output probabilities) is replaced by the weighted binary cross-entropy loss (which is logarithmic in the output probabilities). The code difference is simply:
```diff
-phi = model.sigmoid(model.output_layer(values)).squeeze(1)
+logits = model.output_layer(values).squeeze(1)
+phi = model.sigmoid(logits)

-loss = -(phi * g_stop + (1.0 - phi) * g_cont)
-loss = loss.float().mean()
+delta = g_stop - g_cont
+labels = (delta > 0.0).to(dtype=torch.float32)
+weights = torch.abs(delta).to(dtype=torch.float32)
+loss = F.binary_cross_entropy_with_logits(
+    logits, labels, weight=weights, reduction="mean"
+)
```

To run the replication scripts:

   1. Install dependences: ``pip install -r requirements.txt``.
   2. Create the synthetic data: ``python generate_synthetic_data.py``.
   3. Run the DOS replication: ``python run_dos_replication.py``.
   4. Run the DNN-OSPG replication: ``python run_dnn_ospg_replication.py``.
   5. Run the classifier-based DOS: ``python run_dos_classifier_replication.py``.

You will need roughly 1GB of VRAM for each replication. On an NVIDIA H100 SXM GPU, running `run_dos_replication.py` takes about 40 minutes, and running `run_dnn_ospg_replication.py` and `run_dos_classifier_replication.py` takes about 10 minutes each.

## Replication Results

The authors' results are from Table 1 of the [deep recurrent optimal stopping paper][2]. Our results are from one run of each script. The authors' results are the means over 10 runs with random train-test splits.

| (d, p0) | Authors' DNN-OSPG | Our DNN-OSPG | Authors' DOS | Our DOS | Classifier DOS |
|---|---:|---:|---:|---:|---:|
| d=20, p0=90 | 37.20 | 37.22 | 37.08 | 37.11 | 37.09 |
| d=20, p0=100 | 51.02 | 51.15 | 50.86 | 51.04 | 51.00 |
| d=20, p0=110 | 64.91 | 64.97 | 64.73 | 64.85 | 64.56 |
| d=50, p0=90 | 53.46 | 53.32 | 53.30 | 53.17 | 53.05 |
| d=50, p0=100 | 69.06 | 69.14 | 68.87 | 68.89 | 68.83 |
| d=50, p0=110 | 84.65 | 85.03 | 84.50 | 84.77 | 84.62 |
| d=100, p0=90 | 66.02 | 65.95 | 65.83 | 65.83 | 65.79 |
| d=100, p0=100 | 82.96 | 83.15 | 82.79 | 83.12 | 83.04 |
| d=100, p0=110 | 99.93 | 99.77 | 99.73 | 99.55 | 99.35 |
| d=200, p0=90 | 78.43 | 78.55 | 78.23 | 78.36 | 78.32 |
| d=200, p0=100 | 96.79 | 96.78 | 96.57 | 96.52 | 96.67 |
| d=200, p0=110 | 115.10 | 115.40 | 114.87 | 115.15 | 115.17 |

[1]: https://jmlr.org/papers/volume20/18-232/18-232.pdf
[2]: https://openreview.net/pdf?id=XetXfkYZ6i
[3]: https://openreview.net/forum?id=XetXfkYZ6i&noteId=yvO0EuQGej
