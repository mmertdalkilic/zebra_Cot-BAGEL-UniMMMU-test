# UniCoT v0.2 × Uni-MMMU on MoLab, with GitHub checkpoints

Target GitHub repository:
`mmertdalkilic/zebra_Cot-BAGEL-UniMMMU-test`

Target branch:
`unicot-eval-molab`

This package runs the **native UniCoT v0.2 model code** against the six
official Uni-MMMU task scripts. It does not instantiate TorchUMM or a TorchUMM
model adapter.

## Scientific scope

UniCoT upstream currently does not ship an official Uni-MMMU / reasoning-based
understanding evaluator. The model backend here imports
`Fr0zenCrane/UniCoT/inference_unicot_v0.2.py` directly and preserves the
released v0.2 model loader, native transforms/cache, temperature-0.3 text
sampling, sequential breakdown, self-reflection, CFG settings, and 50
diffusion steps.

The benchmark glue executes the official Vchitect Uni-MMMU
`sample_code_example/gpt/*.py` task scripts and injects the native UniCoT
provider functions.

For multi-source image editing (especially Jigsaw), UniCoT v0.2 has no
author-released multi-source editing API. The native bridge therefore places
the source images on a lossless montage before the native self-reflection/edit
loop. Treat Jigsaw as an adapter-based native-model evaluation, not an
author-reported UniCoT protocol.

## Output / resume

Outputs are tracked under:

`runs/unicot_native_v02/`

and are checkpointed to GitHub every 900 seconds by default. Generated images
are tracked with Git LFS. `_done.ok` files are preserved, so rerunning resumes
instead of regenerating completed examples.

The smoke test runs exactly one sample from each task. The full run uses the
same output tree, so those six successful smoke samples are reused.

## Runtime

Designed for one MoLab RTX PRO 6000 Blackwell 96GB workspace. The historical
UniCoT requirements pin an older PyTorch stack, so setup installs PyTorch
2.8/cu128 and builds FlashAttention 2.8.3.post1 for SM120 while preserving the
rest of the upstream requirements.

## Important

This package starts the **generation/inference** half of Uni-MMMU. Official
Uni-MMMU scoring/judge evaluation is a separate later process.


## v2 MoLab runtime fix

MoLab exposes the RTX PRO 6000 driver/runtime but may not include `nvcc`.
v2 no longer compiles FlashAttention from source. It installs the official
FlashAttention 2.8.3 release wheel matching PyTorch 2.8, CUDA 12, CPython 3.10,
and the detected PyTorch CXX11 ABI, then launches a real BF16 attention kernel
as a setup check.

v2 also preserves UniCoT's `huggingface_hub==0.29.1` compatibility with
`transformers==4.49.0` instead of accidentally upgrading HF Hub to 1.x, and
skips UI/training-only requirements (`gradio`, `wandb`, `bitsandbytes`,
`xlsxwriter`) during the inference environment install.
