# Leonardo generations to MoLab evaluation

This package moves the completed BAGEL Zebra-CoT LoRA Uni-MMMU generations
from Leonardo to a new GitHub branch and evaluates them on one MoLab RTX PRO
6000 96 GB GPU. It does not repeat training or generation.

Repository: https://github.com/mmertdalkilic/zebra_Cot-BAGEL-UniMMMU-test

Branch: leonardo-lora-0015270-molab-eval

Model/output name: bagel-zebra-cot-lora

## Leonardo commands

Upload bagel_leonardo_to_molab_eval_v1.zip to:

    /leonardo_work/EUHPC_B38_013/mdalkili/

Then run:

    cd /leonardo_work/EUHPC_B38_013/mdalkili
    unzip -o bagel_leonardo_to_molab_eval_v1.zip
    module load python/3.11.6--gcc--8.5.0
    python3 molab_eval_transfer/export_leonardo.py \
      --project /leonardo_work/EUHPC_B38_013/mdalkili/bagel_zebra_leonardo \
      --checkout /leonardo_work/EUHPC_B38_013/mdalkili/bagel_molab_transfer_checkout_v1

Enter the GitHub token at the hidden prompt. Do not put the token in the command
or save it in a file. A fine-grained token needs Contents: read/write permission
for the target repository.

The exporter requires all 880 cases: math 140, science 157, code 200, jigsaw
150, maze 149, and sliding 84. It checks completion markers and file sizes,
rewrites only absolute paths inside copied result metadata, and uploads the
generated text/images in bounded commits. It excludes checkpoints, models,
training data, caches, and environments. Rerun the same command after an
interruption.

Wait for:

    TRANSFER COMPLETE: leonardo-lora-0015270-molab-eval

No Slurm job or GPU is required for the transfer.

## MoLab commands and cells

Create a MoLab notebook from this URL after the transfer:

https://github.com/mmertdalkilic/zebra_Cot-BAGEL-UniMMMU-test/blob/leonardo-lora-0015270-molab-eval/molab_eval_transfer/molab_notebook.py

Then:

1. Attach the NVIDIA RTX PRO 6000 96 GB GPU in notebook specifications.
2. Run the notebook cells.
3. Enter the GitHub token in the password widget and submit it.
4. Keep the 10-hour budget for a fresh session.
5. Click Start or resume evaluation.
6. Run the monitoring cell.

Complete copy-and-paste cell contents are in MOLAB_CELLS.md.

The session creates a fresh checkout and a private Python 3.11 environment.
It downloads the pinned official scorer/data and these AWQ judges:

- Qwen/Qwen2.5-VL-72B-Instruct-AWQ
- Qwen/Qwen3-32B-AWQ

The judges run sequentially. They are quantized substitutes for the benchmark's
BF16 judges, so note this when comparing scores with the paper.

## Automatic saving and resume

A separate supervisor commits and pushes evaluation files every 900 seconds,
including while a judge call is running. It also pushes at startup, after each
phase, and on a handled stop. Completed judge-scored cases are under:

    outputs/_eval/bagel-zebra-cot-lora/<task>/items/

The branch also receives:

- scores.csv with every numeric score saved so far.
- PROGRESS.md and progress.json with saved/expected case counts.
- Official task summaries, detail JSON, CSV files, and final XLSX report.
- session_status.json with running, paused, failed, or complete state.

Successful backups print:

    [sync] pushed <commit> <UTC time>

If MoLab ends the session, open a new 96 GB GPU session and run the same notebook
again. It restores committed item files and skips completed cases. The unfinished
judge call and up to the work since the last successful 15-minute push may rerun.

Use only one MoLab evaluation session on this branch. Pushes are normal pushes;
the code never force-pushes. After three consecutive backup failures, evaluation
stops instead of continuing without saving.

To pause, interrupt the monitoring cell and run:

    evaluation_process.terminate()
    evaluation_process.wait()

Wait for the final [sync] pushed line. Exit code 0 means complete, 75 means
paused, and 1 means inspect the logs.

## Logs in MoLab

    subprocess.run(["cat", str(supervisor_log)], check=True)
    subprocess.run(["cat", str(runtime / "setup.log")], check=True)
    subprocess.run(["cat", str(runtime / "evaluation.log")], check=True)

## Read the results on Leonardo

After the first MoLab push:

    cd /leonardo_work/EUHPC_B38_013/mdalkili
    git clone --single-branch \
      --branch leonardo-lora-0015270-molab-eval \
      https://github.com/mmertdalkilic/zebra_Cot-BAGEL-UniMMMU-test.git \
      bagel_molab_results_v1
    cd bagel_molab_results_v1
    cat outputs/_eval/bagel-zebra-cot-lora/PROGRESS.md
    cat outputs/_eval/bagel-zebra-cot-lora/session_status.json

For later updates:

    cd /leonardo_work/EUHPC_B38_013/mdalkili/bagel_molab_results_v1
    git pull --ff-only
    cat outputs/_eval/bagel-zebra-cot-lora/PROGRESS.md

The token is used through a temporary askpass helper. It is absent from remote
URLs, evaluation subprocesses, committed files, and logs.
