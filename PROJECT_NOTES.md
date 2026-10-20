# Project notes

Problems I ran into while getting this project to run on a 4 GB laptop GPU under Windows, and how each one was resolved. They are roughly in the order I met them.

**Setup:** Windows 11, GTX 1650 Ti (4 GB, Turing, compute capability 7.5), Python 3.11.9, PyTorch 2.5.1 (cu124), Transformers 4.46.3, PEFT 0.13.2, Datasets 3.1.0, NumPy 1.26.4, pyarrow 16.1.0.

## 1. CUDA driver newer than any PyTorch build

The driver reported CUDA 13.2, and there is no `torch==2.5.1` wheel for it. Installing from the `cu124` index works, because NVIDIA drivers are backward compatible: the driver only has to be at least as new as the toolkit the build was made with. Pick the PyTorch build that exists, not the one that matches the driver number.

## 2. Python 3.14 is too new

The system Python was 3.14, and the `torch` wheels I needed were not published for it. I made a dedicated virtual environment on Python 3.11.9. On Windows it is worth pinning the interpreter version before installing anything.

## 3. The `adapters` library conflicts with `transformers`

`adapters` 1.0.1 pinned `transformers` to roughly 4.45.2, while the rest of the stack needed 4.46.3. Instead of downgrading everything, I wrote the bottleneck adapter myself: a Houlsby-style module of about 60 lines, attached with forward hooks, with a zero-initialised up-projection so that an untrained adapter is an identity map. It also gave me full control over what the adapter does.

## 4. A native crash from a pyarrow/numpy mismatch

Reading cached Arrow files from `datasets` crashed Python with a Windows access violation and no traceback. I turned on `python -X faulthandler` to get the C-level stack, which pointed into `pyarrow` (at import and in `arrow_schema` during `load_dataset`). The cause was a `pyarrow` built against NumPy 2 loaded next to NumPy 1.26.4. Pinning `pyarrow==16.1.0` and clearing the stale datasets cache fixed it. A silent native crash usually means a binary incompatibility between compiled packages.

## 5. bf16 reported as supported on Turing

The framework's bf16 support check returned true on the Turing GPU, but Turing has no real bf16 acceleration and training was unstable. I now force fp16 by checking the compute capability and rejecting bf16 below the architecture that introduced it.

## 6. A "quick" sanity run that took hours

The AG News sanity check was supposed to finish in about a minute, but at full size it ran for hours. I added per-class caps to the data config, so that a small subsample is a normal option and not a special case.

## 7. A `.gitignore` pattern that dropped source code

A `data/` entry in `.gitignore` matched `src/data/` as well as the dataset directory I meant, so part of the data pipeline was silently left out of version control. Anchoring it as `/data/` fixed it. Check with `git ls-files` and `git check-ignore -v` after editing ignore files.

## 8. GPU memory growing between runs

During the ablation sweeps, peak VRAM rose by about 0.5 GB per run, even though every run used a similar configuration. The PEFT and adapter forward hooks create reference cycles, and Python's reference counting alone did not free finished models promptly, so they piled up on the device. I now call `gc.collect()` and `torch.cuda.empty_cache()` after every run and after every per-task model in the oracle loop. Peak memory stayed flat at about 1.9 GB across a whole sweep. This only showed up when many runs ran back to back, which is why I ran a cheap end-to-end smoke test before every long job.

## Working method

The code was built in seven phases, each with a definition of done and a passing test suite before the next phase started (28 unit tests in the end). Each problem above could be traced to the phase that introduced it.
