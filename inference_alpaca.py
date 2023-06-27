# import os,sys
# path = os.path.join(os.path.dirname(__file__), os.pardir)
# print(path)
# sys.path.append(path)
import os.path
from dataclasses import dataclass, field

import numpy as np
import torch
import transformers
from transformers import GenerationConfig
import random
random.seed(42)
from train_alpaca import ModelArguments, smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, \
    DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, PROMPT_DICT, TASK_PROMPTS
import re
import json
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix



def evaluate_confusion_matrix(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for i in range(num_classes):
        true_positives = confusion_matrix[i, i]
        false_positives = np.sum(confusion_matrix[:, i]) - true_positives
        false_negatives = np.sum(confusion_matrix[i, :]) - true_positives

        precision[i] = true_positives / (true_positives + false_positives)
        recall[i] = true_positives / (true_positives + false_negatives)
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    micro_true_positives = np.sum(np.diag(confusion_matrix))
    micro_false_positives = np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)

    micro_f1 = micro_true_positives / (micro_true_positives + np.sum(micro_false_positives))
    macro_f1 = np.mean(f1)

    return precision, recall, f1, micro_f1, macro_f1


@dataclass
class InferenceArguments:
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load the model in 8-bit mode."},
    )
    inference_dtype: torch.dtype = field(
        default=torch.bfloat16,
        metadata={"help": "The dtype to use for inference."},
    )
    test_data_path: str = field(
        default="",
        metadata={"help": "test data path"},
    )
    few_shot: bool = field(
        default=False,
        metadata={"help": "few-shot/in-context learning"},
    )
    subset_name: str = field(default=None, metadata={"help": "subset name if loading from huggingface datasets"})
    prompt_type: str = field(default="attribution-no-definition", metadata={"help": "prompt engineering: which prompt to use"})
    input_has_query: bool = field(default=True, metadata={"help": "whether to include query in the input for evaluating attribution"})


def generate_prompt(example,prompt_type,input_has_query=False,demo=None):
    prompt_template, input_template, _ = TASK_PROMPTS[prompt_type]

    if input_has_query:
        query = example['query'] if example['query'] != "nan" else ""
        input = input_template.format(query + " " + example['answer'], example['reference'])
    else:
        input = input_template.format(example['answer'], example['reference'])


    if not demo:
        prompt_input = PROMPT_DICT["prompt_input"]
        res = prompt_input.format(prompt_template, input)
    else:
        prompt_input = PROMPT_DICT["prompt_input_with_demo"]
        res = prompt_input.format(prompt_template, demo, input)

    return res

def inference():
    parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
    model_args, inference_args = parser.parse_args_into_dataclasses()
    try:
        print("loading dataset from huggingface...")
        test_data = [row for row in load_dataset(inference_args.test_data_path, inference_args.subset_name)['test']]
    except:
        print("loading dataset from local file...")
        test_data = [row for row in load_dataset(inference_args.test_data_path.split(".")[-1], data_files=inference_args.test_data_path)['train']]


    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=inference_args.load_in_8bit,
        torch_dtype=inference_args.inference_dtype,
        device_map="auto",
    )
    model.cuda()
    model.eval()

    generation_config = GenerationConfig(
        temperature=0,
        top_p=0.9,
        num_beams=1,
        do_sample=False,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        model_max_length=inference_args.model_max_length,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

    true_labels = []
    pred_labels = []

    predictions = []

    label_map = TASK_PROMPTS[inference_args.prompt_type][-1]
    label_regex = r"|".join(list(label_map.keys()))

    for ii, datum in enumerate(tqdm(test_data)):
        if not inference_args.few_shot:
            prompt = generate_prompt(datum, prompt_type=inference_args.prompt_type, input_has_query=inference_args.input_has_query)

        else:
            demo_file_name = {"attribution-no-definition": "demo_attr.txt",
                              "attribution-with-definition": "demo_attr.txt",
                              "fact-checking": "demo_fact-checking.txt",
                              "nli": "demo_NLI.txt",
                              "summarization": "demo_sum.txt"}
            with open(f"prompt_and_demo/{demo_file_name[inference_args.prompt_type]}") as rf:
                demo_str = rf.read()
                rf.close()
            prompt = generate_prompt(datum, prompt_type=inference_args.prompt_type, input_has_query=inference_args.input_has_query, demo=demo_str)


        inputs = tokenizer(prompt, return_tensors="pt", max_length=inference_args.model_max_length, truncation=True)
        outputs = model.generate(input_ids=inputs["input_ids"].cuda(),
                                 generation_config=generation_config,
                                 max_new_tokens=512,
                                 return_dict_in_generate=True,
                                 output_scores=True)
        input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        prediction = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        pred_label = re.search(label_regex, prediction, re.IGNORECASE).group() if re.search(
            label_regex,
            prediction, re.IGNORECASE) is not None else 'None'

        pred_label = label_map[pred_label.capitalize()] if pred_label.capitalize() in label_map else "None"

        output_dict = {
            "Input": prompt,
            "Prediction": prediction,
            "Pred_Label": pred_label,
            "Label": datum["label"]
        }

        # print(output_dict)

        predictions.append(output_dict)

        true_labels.append(datum["label"])
        pred_labels.append(pred_label)

    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=["Attributable", "Contradictory", "Extrapolatory"])

    precision, recall, f1, micro_f1, macro_f1 = evaluate_confusion_matrix(conf_matrix)
    print(conf_matrix)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("micro_f1:", micro_f1)
    print("macro_f1:", macro_f1)

    json.dump(predictions, open(os.path.join(model_args.model_name_or_path, "predictions.json"), 'w'))


if __name__ == "__main__":
    inference()
