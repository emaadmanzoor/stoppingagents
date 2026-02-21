# Replication

These scripts replicate the results in Table 1 of the [deep recurrent optimal stopping paper][2] for the deep optimal stopping (DOS) and deep neural network optimal stopping policy gradient (DNN-OSPG) methods on the Bermudan max-call option scenario described in the [deep optimal stopping paper][1]. The method implementations are our independent replications in PyTorch based on (1) the method descriptions in the papers, and (2) the [TensorFlow implementations][3] of the methods provided by the authors of the deep recurrent optimal stopping paper.

To run the replication scripts:

   1. Install dependences: ``pip install -r requirements.txt``.
   2. Create the synthetic data: ``python generate_synthetic_data.py``.
   3. Run the DOS replication: ``python run_dos_replication.py``.
   4. Run the DNN-OSPG replication: ``python run_dnn_ospg_replication.py``.

You will need roughly 1GB of VRAM for each replication. On an NVIDIA H100 SXM GPU, running `run_dos_replication.py` takes about 40 minutes, and running `run_dnn_ospg_replication.py` takes about 10 minutes.

## Replication Results

The authors' results are from Table 1 of the [deep recurrent optimal stopping paper][2]. Our results are from one run of each replication script.

| (d, p0) | Authors' DNN-OSPG | My DNN-OSPG | Authors' DOS | My DOS |
|---|---:|---:|---:|---:|
| d=20, p0=90 | 37.20 | 37.22 | 37.08 | 37.11 |
| d=20, p0=100 | 51.02 | 51.15 | 50.86 | 51.04 |
| d=20, p0=110 | 64.91 | 64.97 | 64.73 | 64.85 |
| d=50, p0=90 | 53.46 | 53.32 | 53.30 | 53.17 |
| d=50, p0=100 | 69.06 | 69.14 | 68.87 | 68.89 |
| d=50, p0=110 | 84.65 | 85.03 | 84.50 | 84.77 |
| d=100, p0=90 | 66.02 | 65.95 | 65.83 | 65.83 |
| d=100, p0=100 | 82.96 | 83.15 | 82.79 | 83.12 |
| d=100, p0=110 | 99.93 | 99.77 | 99.73 | 99.55 |
| d=200, p0=90 | 78.43 | 78.55 | 78.23 | 78.36 |
| d=200, p0=100 | 96.79 | 96.78 | 96.57 | 96.52 |
| d=200, p0=110 | 115.10 | 115.40 | 114.87 | 115.15 |

[1]: https://jmlr.org/papers/volume20/18-232/18-232.pdf
[2]: https://openreview.net/pdf?id=XetXfkYZ6i
[3]: https://openreview.net/forum?id=XetXfkYZ6i&noteId=yvO0EuQGej
