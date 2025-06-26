import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel, DatasetEntry
#from train_embedding_pix2pix import MLPGenerator, MLPDiscriminator, train_embedding_pix2pix

# Set GPU devices explicitly (e.g., 0, 1, 2)
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4"

def load_tokenizer_and_model(model_name: str, hf_token: str, method):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token_id = 0  # silence warnings

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, token=hf_token
    )
    device = (
        "cuda:0" if torch.cuda.is_available()
        else "mps:0" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)
    model = ControlModel(model, list(range(-5, -18, -1)), method)

    return tokenizer, model


def template(user_tag: str, asst_tag: str, persona: str, suffix: str) -> str:
    return f"{user_tag} Act as if you're extremely {persona}. {asst_tag} {suffix}"


def build_dataset(tokenizer, suffixes, user_tag="[INST]", asst_tag="[/INST]"):
    positive_personas = ["happy", "ecstatic", "delighted"]
    negative_personas = ["sad", "depressed", "dismayed"]

    dataset = []
    for suffix in suffixes:
        tokens = tokenizer.tokenize(suffix)
        for i in range(1, len(tokens)):
            truncated = tokenizer.convert_tokens_to_string(tokens[:i])
            for pos, neg in zip(positive_personas, negative_personas):
                dataset.append(DatasetEntry(
                    positive=template(user_tag, asst_tag, pos, truncated),
                    negative=template(user_tag, asst_tag, neg, truncated),
                ))
    return dataset


def evaluate_model(model, tokenizer, prompt, control_vector=None, strength=None):
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_settings = {
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": False,
        "max_new_tokens": 128,
        "repetition_penalty": 1.1,
    }

    if control_vector is not None and strength is not None:
        model.set_control(control_vector, strength)

    output = model.generate(**input_ids, **gen_settings).squeeze()
    return tokenizer.decode(output)


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    hf_token = "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"  # replace or export token as ENV

    user_tag, asst_tag = "[INST]", "[/INST]"

    # Load dataset
    with open("data/all_truncated_outputs.json") as f:
        suffixes = json.load(f)


    # Prompt for generation
    prompt = f"{user_tag} What are human beings like? {asst_tag}"

    methods = [
        "pca_contrastvector",
        "reading_contrastvector",
        "pca_readingvector",
        "pca_center",
        "readingvector",
        "umap"
    ]


    for method in methods:
        # Load tokenizer and model
        tokenizer, model = load_tokenizer_and_model(model_name, hf_token, method)
        dataset = build_dataset(tokenizer, suffixes, user_tag, asst_tag)
        print("=== BASELINE ===")
        model.reset()
        print(evaluate_model(model, tokenizer, prompt))

        print(f"\n++ METHOD: {method} (Positive direction)")
        model.reset()
        control_vector = ControlVector.train(
            model, tokenizer, dataset, method=method
        )
        print(evaluate_model(model, tokenizer, prompt, control_vector, strength=1.5))

        print(f"\n-- METHOD: {method} (Negative direction)")
        model.reset()
        print(evaluate_model(model, tokenizer, prompt, control_vector, strength=-2.0))

        model.reset()


if __name__ == "__main__":
    main()

