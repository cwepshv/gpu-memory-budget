# llm fine tuning cloud: how much GPU memory and budget you actually need, and how to pick the right instance for QLoRA, LoRA and full fine-tuning

Search **llm fine tuning cloud** and you'll run into three very different products mixed together: managed fine-tuning APIs, GPU marketplaces renting single cards by the hour, and dedicated monthly servers. Which one fits depends on two numbers that most articles skip: how much VRAM your chosen method needs, and how many GPU-hours your run will take. Get those right and the platform choice becomes almost obvious. Get them wrong and you either pay for an 8-GPU node to run QLoRA on a 7B model, or you OOM-loop a 70B fine-tune on a card half its size.

This guide does the memory math first, maps it onto Vultr's on-demand GPU plans, then works through what a fine-tuning run actually costs so you can budget before you deploy anything.

> **Quick pick:** For QLoRA on models up to about 32B, one 48GB GPU (L40S-class) is enough. For 70B QLoRA, or DPO on 7B–13B models, you want an 80GB A100 or H100. Vultr bills GPU instances by the hour with no commitment, so you pay only while training runs and nothing once you destroy the instance. 👉 [Check Vultr's current cloud GPU plans](https://www.vultr.com/products/cloud-gpu/?ref=7268283)

## What "fine-tuning in the cloud" actually requires

A plain GPU virtual machine is enough for the vast majority of fine-tuning jobs. You rent it, install an open-source trainer, run the job, and copy the adapter off before you stop the instance. No special service is involved. The platforms you'll encounter split into three groups:

- **Managed fine-tuning APIs** (OpenPipe, Predibase, Azure AI Foundry) handle GPUs, training jobs and sometimes serving for you. Convenient, but you trade away control over data location, GPU choice and cost structure.
- **Hourly GPU VMs** are the do-it-yourself route: you run Unsloth, Axolotl or Hugging Face TRL yourself. Maximum flexibility, and the cheapest option for bursty workloads.
- **Dedicated monthly servers** make sense only when a GPU stays busy nearly around the clock.

Whichever route you take, the training method decides the hardware. **LoRA** freezes the base model and trains a small adapter for your task. **QLoRA** loads the base model in 4-bit precision, which cuts memory dramatically and makes larger models tunable on modest GPUs. **Full fine-tuning** updates every weight, gives the most control, and needs an order of magnitude more memory. The practical rule, which comes from Unsloth's own guidance: start with QLoRA and move up only with a reason. A task that fails on QLoRA will almost certainly fail with full fine-tuning too, and a QLoRA run costs the least to find out.

It's also worth checking whether you should fine-tune at all. The standard checklist: you've already tried prompt engineering and RAG, you have thousands of high-quality samples with ground-truth data, latency matters, and you have clear evaluation metrics. Fine-tuning shines when the target behavior is stable — a classifier returning the same label set, an extractor producing the same JSON schema, an assistant following a strict format. Adding new facts at query time is retrieval's job, and it usually does that better.

## The VRAM table that decides everything

VRAM, not raw speed, decides which GPU you need. Here are the published numbers, compiled from the memory tables that Unsloth, Axolotl and LLaMA-Factory publish:

| Model size | QLoRA (4-bit base) | LoRA (16-bit base) | Full fine-tune (model states only) |
| --- | --- | --- | --- |
| 7B–8B | 5–14 GB | 16–24 GB | 112–144 GB |
| 13B–14B | 8.5–20 GB | 28–40 GB | 208–252 GB |
| 30B–32B | 24–32 GB | 64–80 GB | 480–576 GB |
| 70B | 40–48 GB | 160–164 GB | 1,120–1,260 GB |

Two caveats keep this table honest. The QLoRA and LoRA ranges are minimums or short-sequence estimates, so longer sequences and bigger batches need more. The full fine-tuning column assumes 16–18 bytes per parameter for weights, gradients and Adam states, before activations — real runs go higher. A quick cross-check if you'd rather do the math yourself: roughly 0.6 bytes per parameter for QLoRA, 2 for LoRA, 16 for full fine-tuning.

Translated into GPU choices:

- One 48GB card handles QLoRA up to 32B with room for longer sequences.
- 70B QLoRA wants 80GB. In Unsloth's benchmark, a 70B QLoRA reached 89,389 tokens of context on 80GB against 12,106 on 48GB. The smaller card technically fits the model but chokes on context.
- 16-bit LoRA at 70B needs about 160GB, which means two 80GB GPUs minimum.
- Full fine-tuning of anything 7B or larger already requires multiple GPUs or optimizer offloading.

## Vultr's GPU plans, mapped to fine-tuning jobs

Vultr sells on-demand NVIDIA GPU instances billed by the hour, from mid-range cards up to A100 and H100 with 80GB, plus bare metal and managed Kubernetes when a single VM stops being enough. No monthly commitment, and billing stops the moment you destroy the instance — which matches how fine-tuning actually happens: short, intense runs separated by evaluation and data work, not a server humming 24/7.

| Vultr GPU option | Memory | What it fits | Billing | Deploy |
| --- | --- | --- | --- | --- |
| L40S-class single GPU | 48 GB | QLoRA up to 32B; 16-bit LoRA up to 14B | Hourly, on-demand | [Deploy](https://www.vultr.com/products/cloud-gpu/?ref=7268283) |
| A100 | 80 GB | 70B QLoRA; DPO on 7B–13B; long-context QLoRA | Hourly, on-demand | [Deploy](https://www.vultr.com/products/cloud-gpu/?ref=7268283) |
| H100 | 80 GB | Same jobs as A100, faster throughput | Hourly, on-demand | [Deploy](https://www.vultr.com/products/cloud-gpu/?ref=7268283) |
| Multi-GPU instance | 160 GB+ | 16-bit LoRA at 70B; full fine-tune of 7B–14B with sharding | Hourly, on-demand | [Deploy](https://www.vultr.com/products/cloud-gpu/?ref=7268283) |

The live catalog also includes smaller and intermediate cards that work fine for 7B QLoRA experiments at lower hourly rates, and availability shifts by region — the deploy screen shows current stock and pricing per data center, so it's worth comparing before you launch.

The deployment flow itself is boring in a good way:

1. Pick a region close to you; Vultr runs data centers across North America, Europe, Asia and Australia.
2. Launch a GPU instance from a marketplace image or a plain Ubuntu box.
3. Install the baseline stack: Python 3.10+, PyTorch 2.1+, Transformers 4.40+, PEFT 0.10+, and bitsandbytes for 4-bit loading.
4. Run your trainer, checkpoint to local storage, and copy adapters off before you stop the instance.
5. Destroy the VM. Billing stops with it.

👉 [Compare Vultr's GPU instances and current hourly rates](https://www.vultr.com/products/cloud-gpu/?ref=7268283) before you commit to a region — stock on 80GB cards moves around.

## What a fine-tuning run really costs

The bill is GPU hourly rate × GPU-hours, and GPU-hours are dominated by wasted time: OOM restarts, tiny batches forced by tight memory, and slow preprocessing. A "cheaper" GPU that runs at half the speed can cost more than a pricier one that finishes sooner.

The estimation workflow is spreadsheet-simple:

- `total training tokens ≈ num_samples × avg tokens per sample × num_epochs`
- `steps ≈ total tokens ÷ (micro_batch × seq_len × grad_accum_steps)`
- Run a 200–500 step calibration, measure seconds per step, and extrapolate total time.
- `cost = rate × (N GPUs × hours)` + storage + egress + retry costs.

Here's the arithmetic in action: 20,000 samples at ~800 tokens, 3 epochs, gives roughly 48M tokens. With an effective batch of 16 × 1,024 tokens, that's about 2,900 steps. If calibration shows 2.5 seconds per step, the run takes around two hours on a single 48GB GPU. Even at a premium hourly rate, you're in single-digit dollars — the numbers only get scary when a job OOMs at step 2,700 and restarts from zero.

For market calibration, one budget GPU marketplace (QuantaCloud) published these rates on 30 September 2026: RTX A6000 48GB from $0.48/hr, L40S $1.09/hr, A100 80GB $1.48–1.50/hr, H100 PCIe $2.59/hr, H200 NVL 141GB $3.43/hr. At the other end, dedicated-server vendor GPUCloudHQ sells a flat-monthly A100 80GB at $649 versus a market median of roughly $1,285/month at hourly rates — a discount that only pays off if the card runs most of the month. Fine-tuning queues rarely do that; experiments finish, evaluation runs, then everyone waits for new data. Hourly on-demand fits that rhythm, which is exactly Vultr's model.

The usual budget killers, in order of how often they bite: model size, precision (FP16 versus 8-bit or 4-bit), context length (4k to 32k tokens inflates step time dramatically), wasteful padding instead of sequence packing, and spot-style interruptions that restart jobs instead of resuming them.

👉 [See Vultr's live GPU pricing](https://www.vultr.com/products/cloud-gpu/?ref=7268283) and plug the number into the formulas above — two minutes of math prevents most billing surprises.

## Which trainer to run: Unsloth, Axolotl or Hugging Face TRL

All three run on a plain GPU VM, and the method you choose matters more than the tool. Versions current as of 28 September 2026:

- **Unsloth (2026.9.12)** covers LoRA, QLoRA, full fine-tuning, DPO and GRPO with its own memory-saving kernels; multi-GPU via `torchrun` or `accelerate launch`.
- **Axolotl (0.19.0)** does the same training modes configured in a single YAML file, with DDP by default plus DeepSpeed ZeRO 1–3 and FSDP2.
- **Hugging Face TRL (1.14.0)** provides trainers for SFT, DPO, GRPO and reward models, called from your own Python code and scaled through Accelerate.

One memory note for preference optimization: DPO on 7B–13B models fits a single 80GB card. PPO-style RLHF, which keeps policy, reference and reward models resident at once, lands in 141GB or multi-GPU territory.

## When a managed fine-tuning platform beats renting GPUs

If you don't want to touch infrastructure at all, or your goal is converting production logs into a cheaper small model, a managed service does the plumbing for you — and renting GPUs would be overkill. A few names worth knowing, information only:

- **OpenPipe** installs as a drop-in replacement for the OpenAI SDK, records prompt/completion pairs from your live application, and trains a smaller specialized model from them. Its recent work centers on reinforcement learning for agents through the open-source ART library and GRPO.
- **Predibase** focuses on efficient LoRA/QLoRA tuning, and its LoRAX serving layer packs many tuned adapters onto shared GPUs, loading each as requests arrive. Its Ludwig and LoRAX cores are open source, so you're not fully locked in.
- **Azure** offers guided fine-tuning for GPT and Phi models through dashboards or a Python SDK — the sensible path if you're already building inside that ecosystem and want no-code or low-code labs.

Whichever path you take, one rule applies everywhere: a fine-tune only counts if it beats the base model *and* a strong prompt on held-out examples from your own data. Falling training loss proves the model fit your training set; it says nothing about real user inputs.

## FAQ

**Can you fine-tune a 70B model on a single GPU?**
Yes, with QLoRA on 80GB — the memory tables put 70B QLoRA at 40–48GB. 16-bit LoRA needs around 160GB, and full fine-tuning needs roughly 1.1TB, which means multi-GPU nodes either way.

**Do I need multiple GPUs for fine-tuning?**
Not for QLoRA up to 32B on a 48GB card. Multi-GPU starts at 16-bit LoRA for large models, full fine-tuning of anything 7B or larger, and PPO-style RLHF.

**Should I pay hourly or monthly for a fine-tuning GPU?**
Hourly for experiments and bursty queues; flat monthly dedicated servers only when a card stays busy nearly around the clock. For reference, the monthly-vs-hourly comparison above showed a $649 dedicated A100 versus a ~$1,285 hourly median — the gap closes fast on an idle GPU.

**What usually blows up a fine-tuning budget?**
OOM restarts from tight memory, oversized context windows, padding waste, and jobs that restart instead of resuming after interruptions. A short calibration run catches most of this before it costs anything.

## Pick your setup and start the run

- **7B–13B QLoRA experiments:** the cheapest smaller GPU you can find, hourly, destroyed when idle.
- **QLoRA up to 32B:** one 48GB L40S-class instance.
- **70B QLoRA, long context, or DPO:** one 80GB A100 or H100.
- **16-bit LoRA at 70B, or full fine-tuning of 7B+:** multi-GPU, with DeepSpeed or FSDP for sharding.
- **Zero-infrastructure, log-driven small models:** a managed platform instead of a rented VM.

Before the real run, do the 200–500 step calibration, keep checkpoints on local storage, and copy adapters off before you stop anything — hourly billing only rewards you if you actually destroy idle instances. 👉 [Open Vultr's cloud GPU plans and launch your first fine-tune](https://bit.ly/VulTR)
