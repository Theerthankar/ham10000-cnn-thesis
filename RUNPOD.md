# Running this on RunPod

Written for a fresh pod. Following it in order should get from nothing to twelve
finished experiments without surprises.

## Pod

Pick a PyTorch template so torch and CUDA are already there. Requirements:

| | |
|---|---|
| GPU | 16 GB VRAM minimum. Verified on an RTX A4500 (20 GB) |
| Disk | 60 GB (dataset unpacks to ~2.7 GB, twelve checkpoint sets add up) |
| Template | RunPod PyTorch 2.x |

Peak GPU memory for a real training step at batch size 32, measured on the pod rather
than estimated:

| Configuration | Peak | Experiments |
|---|---|---|
| MobileNetV2 @224 | 2.6 GB | E1, E3, E5 |
| ResNet-50 @224 | 3.1 GB | E2, E4, E6 |
| EfficientNet-B3 @224 | 5.8 GB | E7, E8, E9 |
| EfficientNet-B3 @300 | 10.2 GB | E10, E11, E12 |

E10 to E12 set the ceiling. A 16 GB card clears it with room to spare, so most GPUs on
offer will do. If one ever does run out of memory, move to a bigger card rather than
lowering the batch size: batch size is part of the fixed protocol, and changing it for
three runs would make them incomparable with the other nine.

Budget 10 to 16 hours of GPU time for all twelve, as an estimate rather than a
measurement. At roughly $0.40 to $0.80 an hour that lands somewhere around $6 to $15.
The three 300px runs are the slow ones. Time E1 first to calibrate against the pod you
actually get before leaving the rest unattended.

## Getting the code and data across

```bash
# from your laptop
scp -P <port> ham10000-cnn-comparison.zip root@<pod-ip>:/workspace/
```

Then on the pod:

```bash
cd /workspace
unzip -q ham10000-cnn-comparison.zip
cd ham10000-cnn-comparison
```

The dataset is not in the zip. It is 2.7 GB and publicly available, and you already have
it locally in `dataverse_files/`, so the simplest route is to push that across:

```bash
scp -P <port> -r dataverse_files root@<pod-ip>:/workspace/ham10000-cnn-comparison/
```

Only three of the files in there are needed. Copying just those saves about 430 MB:

```bash
mkdir -p dataverse_slim
cp dataverse_files/HAM10000_metadata \
   dataverse_files/HAM10000_images_part_1.zip \
   dataverse_files/HAM10000_images_part_2.zip dataverse_slim/
scp -P <port> -r dataverse_slim root@<pod-ip>:/workspace/ham10000-cnn-comparison/dataverse_files
```

Notebook `01` looks for `dataverse_files/` at the repo root automatically, so nothing
needs editing if it lands there.

Downloading straight onto the pod is faster than uploading over a home connection, if
you would rather do that. Get the per-file download links from the dataset page:

> https://doi.org/10.7910/DVN/DBW86T

Open it, use the download button on `HAM10000_metadata`,
`HAM10000_images_part_1.zip` and `HAM10000_images_part_2.zip`, and copy the resulting
`/api/access/datafile/<id>` URLs into `curl -L -o <name> <url>` on the pod. The numeric
ids are not stable across dataset versions, which is why they are not written out here.

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` deliberately leaves torch and torchvision commented out. The pod
already has a CUDA-matched build, and `pip install torch` can silently replace it with
a CPU-only wheel. That failure is nastier than a crash, because nothing errors: training
just runs about fifty times slower and you may not notice until hours in. Notebook `00`
checks for exactly this.

## Jupyter

RunPod exposes Jupyter on port 8888 through the web UI. If you would rather tunnel:

```bash
# on the pod
jupyter lab --allow-root --no-browser --port=8888 --ip=0.0.0.0

# from your laptop
ssh -N -L 8888:localhost:8888 -p <port> root@<pod-ip>
```

## Order to run things

1. `00_environment_check` — a minute. Stop here if the GPU check fails.
2. `01_data_and_split` — a few minutes, mostly unzipping.
3. `02_eda` — a minute. Produces the data-chapter figures.
4. `03` through `14` — the twelve training runs. This is the long part.
5. `20_efficiency_benchmark` — a minute, CPU only.
6. `21_statistical_tests` — seconds. Needs all twelve finished.
7. `22_figures_and_tables` — a minute. Produces everything the write-up needs.

Start with `03` (E1). It is the fastest of the twelve and it creates the shared split
file, so if something is wrong with the setup you find out in half an hour rather than
after a 300px run.

## Shut each kernel down when its run finishes

This one will bite you around the sixth or seventh notebook if you skip it.

A finished notebook does not release GPU memory. The kernel stays alive holding its CUDA
context and the model, roughly 3 GB per run. Work through twelve notebooks without
closing anything and the card fills up. The next run then fails to allocate, writes
`config.json` and an empty `history.csv`, and stops. The kernel goes back to idle, which
looks identical to still running: no progress, no error on screen, GPU at 0%.

After each run completes, in JupyterLab open the left sidebar, go to **Running Terminals
and Kernels**, and shut down that notebook's kernel. Closing the browser tab is not
enough.

To check what is holding memory:

```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
```

Free memory in the low hundreds of MB means you have stale kernels. To see which
notebook each kernel belongs to, take the token from `jupyter server list` and:

```bash
curl -s -H "Authorization: token <token>" http://127.0.0.1:8888/api/sessions \
  | python3 -m json.tool | grep -E '"path"|execution_state'
```

## Surviving a dropped connection

Every epoch writes `checkpoint_last.pt`. If the pod disconnects or the browser tab dies,
re-run the training cell and it resumes from the last completed epoch. Nothing is lost
and nothing needs to be reset.

To run unattended instead of through the browser:

```bash
nohup .venv/bin/python scripts/train.py --experiment E7 > logs/E7.log 2>&1 &
tail -f logs/E7.log
```

The notebooks and the CLI write to the same place, so mixing them is fine.

## Before terminating the pod

Billing stops at termination, and so does the disk. Pull the results down first:

```bash
# from your laptop
scp -P <port> -r root@<pod-ip>:/workspace/ham10000-cnn-comparison/runs ./
scp -P <port> -r root@<pod-ip>:/workspace/ham10000-cnn-comparison/figures ./
scp -P <port> -r root@<pod-ip>:/workspace/ham10000-cnn-comparison/tables ./
```

`runs/` without the checkpoints is only a few MB. The checkpoints are about 40 to 90 MB
each and are not needed for any figure, table or statistic in the write-up, so leave
them unless you plan to do inference later:

```bash
# smaller pull, everything except the weights
rsync -av --exclude='*.pt' -e "ssh -p <port>" \
  root@<pod-ip>:/workspace/ham10000-cnn-comparison/runs ./
```

## If something goes wrong

**CUDA out of memory on E10 to E12.** Expected on anything under 24 GB. Use a larger
GPU rather than reducing the batch size.

**`torch.cuda.is_available()` is False.** Either the pod has no GPU attached or pip
overwrote torch with a CPU build. Reinstall the CUDA build that matches the pod.

**A run stops at epoch 50.** It hit the ceiling instead of converging. Worth noting in
the write-up, since every run in the earlier round stopped well before that.

**Split hash differs between notebooks.** Something regenerated the split. Delete
`runs/splits/` and re-run notebook `01`, then retrain anything that had the old hash.
Do not mix results across two different splits.
