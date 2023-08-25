import os, sys
import logging
import json, pickle
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from trie import Trie
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from tqdm import tqdm
from utils import DTYPE_MAP


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    # Get dataset and save it to data_dir
    dataset = config["task"]
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset
    )
    out_dir = config["data_dir"]
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    # print dataset statistics
    logging.info(f"Dataset: {dataset}")
    logging.info(f"Corpus size: {len(corpus)}")
    logging.info(f"Queries size: {len(queries)}")
    logging.info(f"Qrels size: {len(qrels)}")

    # Load model and tokenizer
    model_config = config["model"]
    if model_config["type"] == "CausalLM":
        ModelClass = AutoModelForCausalLM
    elif model_config["type"] == "Seq2SeqLM":
        ModelClass = AutoModelForSeq2SeqLM
    else:
        raise ValueError("Invalid model type")
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = ModelClass.from_pretrained(
        model_config["model_name_or_path"],
        low_cpu_mem_usage=model_config["low_cpu_mem_usage"],
        torch_dtype=DTYPE_MAP[model_config["dtype"]],
        device_map=model_config["device_map"],
        max_memory=model_config["max_memory"],
    ).eval()

    # Create or load trie
    trie_config = config["trie"]
    # If num_beam > trie_depth then assert
    assert config["num_beams"] <= trie_config["trie_depth"]
    trie_path = f'{config["trie_dir"]}/{model_config["name"]}/{dataset}'
    # Create save dir if not exists
    Path(trie_path).mkdir(parents=True, exist_ok=True)
    trie_file = f'{trie_path}/{trie_config["type"]}_{trie_config["trie_depth"]}.pkl'
    # Manually add eos token to the end of the sentence
    # Also append index of corpus after eos token
    if config["create_trie"] or not os.path.exists(trie_file):
        sents = []
        for k, v in tqdm(corpus.items()):
            if trie_config["type"] == "single_index":
                sentence = tokenizer.encode(
                    v["text"], truncation=True, max_length=trie_config["trie_depth"]
                )
                if sentence[-1] != tokenizer.eos_token_id:
                    sentence.append(tokenizer.eos_token_id)
                sentence.append(k)
                sents.append(sentence)

            elif trie_config["type"] == "multiple_index_sentence":
                sentences = v["text"].split(". ")
                for sent in sentences:
                    sentence = tokenizer.encode(
                        sent, truncation=True, max_length=trie_config["trie_depth"]
                    )
                    if sentence[-1] != tokenizer.eos_token_id:
                        sentence.append(tokenizer.eos_token_id)
                    sentence.append(k)
                    sents.append(sentence)
            elif trie_config["type"] == "multiple_index_length":
                tokenized_input = tokenizer.encode(
                    v["text"], max_length=2048, truncation=True
                )
                num_chunks = len(tokenized_input) // trie_config["trie_depth"]
                for i in range(num_chunks):
                    sentence = tokenized_input[
                        i
                        * trie_config["trie_depth"] : (i + 1)
                        * trie_config["trie_depth"]
                    ]
                    if sentence[-1] != tokenizer.eos_token_id:
                        sentence.append(tokenizer.eos_token_id)
                    sentence.append(k)
                    sents.append(sentence)
            else:
                raise ValueError("Invalid trie type")
        trie = Trie(sents)
        logging.info(f"Saving trie to {trie_file}")
        trie.save(trie_file)
    else:
        logging.info(f"Loading trie from {trie_file}")
        trie = Trie.load(trie_file)
    logging.info(f"Lenght of trie: {len(trie)}")

    def prefix_allowed_fn(batch_id, sent):
        sent = sent.tolist()
        trie_out = trie.get(sent[start_pos:])
        return trie_out

    template = config["templates"]["template"]

    results = {}
    for i, (q_id, c) in enumerate(tqdm(qrels.items())):
        results[q_id] = {}
        if i >= config["max_gen"]:
            break
        input_str = template.replace("[QUERY]", queries[q_id])
        with torch.inference_mode():
            input_ids = tokenizer(
                input_str, return_tensors="pt", max_length=2048, truncation=True
            ).input_ids.to("cuda")
            input_len = input_ids.shape[1]
            if model_config["type"] == "CausalLM":
                start_pos = input_len
            elif model_config["type"] == "Seq2SeqLM":
                start_pos = 0
            outputs = model.generate(
                input_ids,
                max_new_tokens=2048,
                prefix_allowed_tokens_fn=prefix_allowed_fn,
                num_beams=config["num_beams"],
                num_return_sequences=config["num_beams"],
                remove_invalid_values=True,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        for output, score in zip(
            outputs.sequences, torch.exp(outputs.sequences_scores)
        ):
            out_list = output.tolist()
            retrieved = trie.get(out_list[start_pos:])
            for cid in retrieved:
                if cid not in results[q_id]:
                    converted_score = score.item()
                    results[q_id][cid] = converted_score

    retriever = EvaluateRetrieval()
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, results, [1, 3, 5, 10], ignore_identical_ids=False
    )
    # Create save dir if not exists
    save_path = f'{config["save_path"]}/{model_config["name"]}/{dataset}'

    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_file = f'{save_path}/{config["templates"]["template_name"]}_{trie_config["type"]}_{trie_config["trie_depth"]}.jsonl'
    with open(save_file, "w") as f_out:
        for metric in [ndcg, _map, recall, precision]:
            f_out.write(json.dumps(metric, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.setrecursionlimit(5000)
    main()
