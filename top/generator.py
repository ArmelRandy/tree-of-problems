from typing import List
import torch
import os

from top.models import hf_generate

STOP_WORDS = [
    "###",
    "\n" * 5,
    "\n\n---",
    "____",
    "....",
    ". . . .",
    "strong>strong>",
    "Q:",
    "\nProblem:",
    "://",
    "\nA:",
    "<|eot_id|>",
    "<|start_header_id|>",
    "\n\nFinal Answer:",
    "\n\nProblem:",
    "\n\nInput:",
    "#include",
    "[INST]",
    "\nHuman:",
    "\nNote:",
    "<end_of_turn>",
    "<EOS_TOKEN>",
]

MAX_NUMBER_OF_PROPOSITIONS = 8


class Generator:
    """
    Sampler for Compositional Translation (CoTra)
    Arguments
    ---------
        - model_name_or_path: str,
            Name or path to the model of interest e.g. google/gemma-2-2b-it on HF, gpt-3.5-turbo-0125 on OpenAI etc.
        - tokenizer_name_or_path: str,
            Name or path to the tokenizer of interest if relevant. Usually the same as model_name_or_path on HF.
    """

    def __init__(self, model_name_or_path: str, tokenizer_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        num_return_sequences: int,
        num_beams: int,
        do_sample: bool,
        request_batch_size: int,
        verbose: bool = True,
    ) -> List[List[str]]:
        raise NotImplementedError("The function `generate` is not implemented.")


from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import torch


class HFGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path
            if self.tokenizer_name_or_path
            else self.model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
        )
        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
            if (self.tokenizer.pad_token is None)
            else self.tokenizer.pad_token
        )
        self.accelerator = Accelerator()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map={"": self.accelerator.process_index},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=(
                "eager"
                if "gemma-2-" in self.model_name_or_path
                else "flash_attention_2"
            ),
        )

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        do_sample: bool = False,
        request_batch_size: int = 8,
        verbose: bool = True,
    ) -> List[List[str]]:
        if isinstance(prompts, list):
            pass
        else:
            # single prompt, i.e str
            prompts = [prompts]
        response = hf_generate(
            accelerator=self.accelerator,
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_words=[
                "\n\nQ:",
                "\n\n###",
                "\nProblem:",
                "://",
                "<|eot_id|>",
                "<|start_header_id|>",
            ],
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            forced_bos_token_id=None,
            batch_size=request_batch_size,
        )
        outputs = []
        for i, r in enumerate(response):
            output = r["answer"]
            outputs.append(output)
        if verbose:
            print("===")
            for i, output in enumerate(outputs):
                for out in output:
                    print(f"{i+1} -> {out}")
            print("===")
        return outputs


import os
import openai


class OpenAIGenerator(Generator):
    def __init__(self, api_key=None, max_retry=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retry = max_retry
        self.api_key = api_key if api_key else os.environ["OPENAI_API_KEY"]
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        do_sample: bool = False,
        request_batch_size: int = 8,
        verbose: bool = True,
    ) -> List[List[str]]:
        if isinstance(prompts, list):
            pass
        else:
            # single prompt, i.e str
            prompts = [prompts]
        if self.model_name_or_path in [
            "babbage-002",
            "davinci-002",
            "gpt-3.5-turbo-instruct",
        ]:
            attempt = 0
            while attempt < self.max_retry:
                try:
                    response = self.client.completions.create(
                        model=self.model_name_or_path,
                        prompt=prompts,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        best_of=num_beams,
                    )
                    break
                except Exception as e:
                    print(f"OpenAIError: {e}.")
                    attempt += 1
            outputs = []
            for j, _ in enumerate(prompts):
                output = response.choices[
                    j * num_return_sequences : (j + 1) * num_return_sequences
                ]
                if verbose:
                    for _, out in enumerate(output):
                        print(f"{j+1} -> {out.text}\n{out.finish_reason}")
                outputs.append([out.text for out in output])
            return outputs
        else:
            attempt = 0
            responses = []
            while attempt < self.max_retry:
                try:
                    start = len(responses)
                    for q, prompt in enumerate(prompts):
                        if q < start:
                            continue
                        response = self.client.chat.completions.create(
                            model=self.model_name_or_path,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_new_tokens,
                            n=num_return_sequences,
                            # seed=self.seed
                        )
                        responses.append(response)
                    break
                except Exception as e:
                    print(f"OpenAIError: {e}.")
                    attempt += 1
            assert len(responses) == len(prompts), "Size mismatch."
            outputs = []
            for j, response in enumerate(responses):
                if verbose:
                    for choice in response.choices:
                        print(
                            f"{j+1} -> {choice.message.content}\n{choice.finish_reason}"
                        )
                outputs.append([choice.message.content for choice in response.choices])
            return outputs


from vllm import LLM, SamplingParams


class vLLMGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            best_of=1,
            repetition_penalty=1.03,
            use_beam_search=False,
            skip_special_tokens=True,
            stop=[
                "\nQ:",
                "\n###",
                "<|eot_id|>",
                "\nHuman:",
                "\n<end_of_turn>",
                "<|start_header_id|>",
                "\n\n\n\n\n",
                ">>>>>",
                "```python",
                "=====",
                "\\\n" * 4,
                "<EOS_TOKEN>",
            ],
        )
        if "gemma-2-" in self.model_name_or_path:
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

        if (
            "awq" in self.model_name_or_path.lower()
            or "gptq" in self.model_name_or_path.lower()
        ):
            self.llm = LLM(
                model=self.model_name_or_path,
                quantization=(
                    "AWQ" if "awq" in self.model_name_or_path.lower() else "GPTQ"
                ),
                dtype="half",
                max_model_len=(
                    2048
                    if any(
                        [
                            element in self.model_name_or_path
                            for element in ["bloom", "OLMo", "opt", "xglm", "llama-2"]
                        ]
                    )
                    # else 4096
                    # else 3584
                    else 3072
                ),
                enforce_eager=True,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
            )
        else:
            self.llm = LLM(
                model=self.model_name_or_path,
                dtype=("bfloat16" if "gemma-2-" in self.model_name_or_path else "half"),
                max_model_len=(
                    2048
                    if any(
                        [
                            element in self.model_name_or_path
                            for element in ["bloom", "OLMo", "opt", "xglm"]
                        ]
                    )
                    else 4096
                ),
                enforce_eager=True,
                trust_remote_code=True,
                swap_space=8,
            )

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        do_sample: bool = False,
        request_batch_size: int = 8,
        verbose: bool = True,
    ) -> List[List[str]]:
        # Initialization
        self.sampling_params.tempearture = temperature
        self.sampling_params.top_p = top_p
        self.sampling_params.best_of = num_beams
        self.sampling_params.repetition_penalty = repetition_penalty
        self.sampling_params.use_beam_search = not do_sample and num_beams > 1
        self.sampling_params.max_tokens = max_new_tokens
        self.sampling_params.n = num_return_sequences
        self.sampling_params.skip_special_tokens = (True,)
        self.sampling_params.ignore_eos = True

        if isinstance(prompts, list):
            pass
        else:
            # single prompt, i.e str
            prompts = [prompts]
        response = self.llm.generate(prompts, self.sampling_params)
        if verbose:
            print("===")
            for i, r in enumerate(response):
                for element in r.outputs:
                    print(f"{i+1} -> {element.text}\n{element.finish_reason}")
            print("===")
        return [[element.text for element in r.outputs] for r in response]


if __name__ == "__main__":
    generator = OpenAIGenerator(
        api_key=os.environ.get(
            "OPENAI_API_KEY", "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"
        ),
        model_name_or_path="gpt-3.5-turbo-0125",
        tokenizer_name_or_path=None,
    )
    sentence = "What is the capital city of Cameroon?"
    t = generator.generate(
        sentences=[sentence], temperature=0.7, num_return_sequences=4, do_sample=True
    )
    print(f"Answer: {t[0]}")
