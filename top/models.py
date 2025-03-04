import tqdm
import torch
import warnings
from torch.utils.data.dataloader import DataLoader
from transformers import StoppingCriteriaList
from top.utils import TokenizedDataset, EndOfFunctionCriteria


def hf_generate(
    accelerator,
    model,
    tokenizer,
    prompts,
    max_new_tokens,
    temperature,
    top_p,
    stop_words,
    num_beams,
    repetition_penalty,
    num_return_sequences,
    do_sample,
    forced_bos_token_id=None,
):
    accelerator.free_memory()
    results = []
    tokenized_dataset = TokenizedDataset(tokenizer=tokenizer, dataset=prompts)
    dataloader = DataLoader(tokenized_dataset, batch_size=1)
    dataloader = accelerator.prepare(dataloader)
    pad_first = tokenizer.padding_side == "left"
    for _, batch in tqdm.tqdm(enumerate(dataloader)):
        with torch.no_grad():
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            index_prompt = batch["index_prompt"]
            stopping_criteria = StoppingCriteriaList(
                [EndOfFunctionCriteria(attention_mask.sum(), stop_words, tokenizer)]
            )
            response = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria,
                do_sample=do_sample,
                forced_bos_token_id=forced_bos_token_id,
            )
            padded_responses = accelerator.pad_across_processes(
                response, dim=1, pad_index=tokenizer.pad_token_id, pad_first=pad_first
            )
            padded_attention_mask = accelerator.pad_across_processes(
                attention_mask, dim=1, pad_index=0, pad_first=pad_first
            )
            indices = accelerator.gather(index_prompt)
            answers = accelerator.gather(padded_responses)
            padded_attention_mask = accelerator.gather(padded_attention_mask)
            for i in range(accelerator.num_processes):
                results.append(
                    {
                        "prompt": prompts[indices[i]],
                        "answer": tokenizer.batch_decode(
                            answers[
                                i
                                * num_return_sequences : (i + 1)
                                * num_return_sequences
                            ],
                            skip_special_tokens=True,
                        ),
                    }
                )
    accelerator.free_memory()
    return results


def apply_template(key):
    """
    Take as input a model's name (as on Hugging Face) and return the template to apply to it during inference.
    Particularly helpful for instruction fine-tuned models.
    """
    if key in [
        "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
    ]:
        return lambda prompt: f"<s>[INST]{prompt}[/INST]"
    elif key in ["TheBloke/Llama-2-13B-Chat-AWQ", "TheBloke/Llama-2-70B-Chat-AWQ"]:
        return (
            lambda prompt: f"[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n{prompt}[/INST]"
        )
    elif key in [
        "casperhansen/llama-3-70b-instruct-awq",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ]:
        return (
            lambda prompt: f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
    elif key in ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf"]:
        return (
            lambda prompt: f"<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n{prompt} [/INST]",
        )
    elif key in [
        "Qwen/Qwen2-72B-Instruct-AWQ",
    ]:
        return (
            lambda prompt: f"<|im_start|>system\nYou are a helpful assitant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
    elif key in [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    ]:
        return (
            lambda prompt: f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif key in [
         "Qwen/Qwen2.5-0.5B-Instruct",
         "Qwen/Qwen2.5-1.5B-Instruct",
         "Qwen/Qwen2.5-7B-Instruct",
         "Qwen/Qwen2.5-32-Instruct",
         "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
         "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
         "Qwen/Qwen2.5-7B-Instruct-AWQ",
         "Qwen/Qwen2.5-32-Instruct-AWQ"
    ]:
        return lambda prompt: f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    elif key in ["gpt-4o-mini-2024-07-18", "gpt-3.5-turbo-instruct"]:
        return lambda prompt: prompt
    else:
        warnings.warn(
            f"If '{key}' is an instruction fine-tuned model, you should incorporate its template in `models.py`, otherwise feel free to ignore this warning."
        )
        return lambda prompt: prompt
