"""MiniCPM-V 4.6 model for lmms-eval.

MiniCPM-V 4.6 is an all-around vision-language model supporting image and video inputs.
It uses the standard HuggingFace generation pipeline (processor + model.generate),
unlike earlier MiniCPM-V versions that relied on a custom .chat() API.

https://huggingface.co/openbmb/MiniCPM-V-4_6

Example Usage:
    python -m lmms_eval --model minicpmv4_6 \\
        --model_args pretrained=openbmb/MiniCPM-V-4_6 \\
        --tasks mmmu_val \\
        --batch_size 1
"""

import time
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

try:
    from transformers.models.minicpmv4_6 import MiniCPMV4_6ForConditionalGeneration
except ImportError:
    MiniCPMV4_6ForConditionalGeneration = None

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None

MAX_NUM_FRAMES = 64


def encode_video(video_path: str, max_frames: int = MAX_NUM_FRAMES) -> List[Image.Image]:
    """Extract frames from video file."""
    if VideoReader is None:
        raise ImportError("decord is required for video processing. Install with: pip install decord")

    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > max_frames:
        frame_idx = uniform_sample(frame_idx, max_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")) for v in frames]
    return frames


@register_model("minicpmv4_6")
class MiniCPMV4_6(lmms):
    """
    MiniCPM-V 4.6 model for multi-modal evaluation.

    https://huggingface.co/openbmb/MiniCPM-V-4_6
    """

    def __init__(
        self,
        pretrained: str = "openbmb/MiniCPM-V-4_6",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        max_num_frames: int = MAX_NUM_FRAMES,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        trust_remote_code: Optional[bool] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.max_num_frames = max_num_frames
        self.system_prompt = system_prompt

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map and device_map not in ("", "none"):
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # Prepare model loading arguments
        model_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": self.device_map,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # AutoModel resolves to MiniCPMV4_6Model (without lm_head/generate),
        # so we must use MiniCPMV4_6ForConditionalGeneration directly.
        if MiniCPMV4_6ForConditionalGeneration is not None:
            self._model = MiniCPMV4_6ForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        else:
            eval_logger.warning("MiniCPMV4_6ForConditionalGeneration not found in transformers, falling back to AutoModel (may lack generate())")
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained(pretrained, **model_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)

        self._config = self._model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for MiniCPMV4_6")

    def flatten(self, input_list):
        new_list = []
        for i in input_list:
            for j in i:
                new_list.append(j)
        return new_list

    def _build_message_content(self, context: str, visual) -> list:
        """Build message content list for chat template."""
        content = []

        if visual is not None:
            if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                # Video file - extract frames
                try:
                    frames = encode_video(visual, self.max_num_frames)
                    for frame in frames:
                        content.append({"type": "video", "video": frame})
                except Exception as e:
                    eval_logger.warning(f"Failed to encode video: {e}")

            elif isinstance(visual, Image.Image):
                content.append({"type": "image", "image": visual})

            elif isinstance(visual, (list, tuple)):
                for v in visual:
                    if isinstance(v, Image.Image):
                        content.append({"type": "image", "image": v})
                    elif isinstance(v, str) and v.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                        try:
                            frames = encode_video(v, self.max_num_frames)
                            for frame in frames:
                                content.append({"type": "video", "video": frame})
                        except Exception as e:
                            eval_logger.warning(f"Failed to encode video: {e}")

        content.append({"type": "text", "text": context})
        return content

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        total_elapsed_time = 0
        total_tokens = 0

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            # Build messages for each sample
            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                visual = visuals[i] if i < len(visuals) else None
                content = self._build_message_content(context, visual)
                message.append({"role": "user", "content": content})
                batched_messages.append(message)

            # Apply chat template
            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)

            # Collect images and videos for processor
            all_images = []
            all_videos = []
            for msg_list in batched_messages:
                for msg in msg_list:
                    if isinstance(msg.get("content"), list):
                        for item in msg["content"]:
                            if item.get("type") == "image" and isinstance(item.get("image"), Image.Image):
                                all_images.append(item["image"])
                            elif item.get("type") == "video" and isinstance(item.get("video"), list):
                                all_videos.append(item["video"])

            # Process inputs
            processor_kwargs = {
                "text": texts,
                "padding": True,
                "return_tensors": "pt",
            }
            if all_images:
                processor_kwargs["images"] = all_images
            if all_videos:
                processor_kwargs["videos"] = all_videos

            inputs = self.processor(**processor_kwargs)

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 1024,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            end_time = time.time()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.post_process_image_text_to_text(
                generated_ids_trimmed,
                skip_special_tokens=True,
            )

            # Apply stop sequences
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            total_elapsed_time += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
        metric_dict = {
            "total_gen_tokens": total_tokens,
            "total_elapsed_time": total_elapsed_time,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for MiniCPMV4_6")
