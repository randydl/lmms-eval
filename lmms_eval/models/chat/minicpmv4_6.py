"""MiniCPM-V 4.6 chat model for lmms-eval.

Chat model variant that uses the `doc_to_messages` protocol for structured
message formatting with the ChatMessages class.

https://huggingface.co/openbmb/MiniCPM-V-4_6

Example Usage:
    python -m lmms_eval --model minicpmv4_6 \\
        --model_args pretrained=openbmb/MiniCPM-V-4_6 \\
        --tasks mmmu_val \\
        --batch_size 1
"""

import time
from typing import List

from loguru import logger as eval_logger
from tqdm import tqdm

try:
    import decord
except ImportError:
    decord = None

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.minicpmv4_6 import MiniCPMV4_6 as MiniCPMV4_6Simple
from lmms_eval.protocol import ChatMessages


@register_model("minicpmv4_6_chat")
class MiniCPMV4_6(MiniCPMV4_6Simple):
    is_simple = False

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]

            # Extract media from messages
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)

            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            # Build HF messages for processor
            batched_messages = [chat_message.to_hf_messages() for chat_message in chat_messages]

            # Apply chat template
            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)

            # Collect images and videos for processor
            all_images = []
            all_videos = []
            for msg_list in batched_messages:
                for msg in msg_list:
                    if isinstance(msg.get("content"), list):
                        for item in msg["content"]:
                            if item.get("type") == "image" and isinstance(item.get("image"), object):
                                img = item["image"]
                                from PIL import Image as PILImage
                                if isinstance(img, PILImage.Image):
                                    all_images.append(img)
                            elif item.get("type") == "video":
                                vid = item.get("video")
                                if isinstance(vid, list):
                                    all_videos.append(vid)

            # Process inputs
            processor_kwargs = {
                "text": texts,
                "padding": True,
                "padding_side": "left" if self.batch_size > 1 else "right",
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

            # Calculate timing metrics
            total_elapsed_time += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for i, (ans, context) in enumerate(zip(answers, texts)):
                res.append(GenerationResult(text=ans, token_counts=TokenCounts(output_tokens=len(generated_ids_trimmed[i]))))
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
