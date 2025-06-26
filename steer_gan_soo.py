import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3,4'

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel, DatasetEntry
import wandb
from repeng.train_embedding_pix2pix import MLPGenerator, MLPDiscriminator, train_embedding_pix2pix
from repeng.extract import batched_get_hiddens
from repeng.control import ControlModel, model_layer_list
import numpy as np



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
    positive_personas = [
    "happy", "ecstatic", "joyful", "elated", "cheerful", "content", "grateful", "hopeful", "inspired", "playful",
    "enthusiastic", "optimistic", "satisfied", "relieved", "thrilled", "peaceful", "radiant", "amused", "kind", "loving",
    "generous", "tender", "affectionate", "caring", "excited", "buoyant", "uplifted", "motivated", "blessed", "bubbly",
    "blissful", "serene", "vibrant", "exuberant", "adventurous", "energized", "empowered", "fulfilled", "warm", "free",
    "curious", "confident", "encouraged", "resilient", "balanced", "harmonious", "cheery", "merry", "delighted", "glad",
    "gleeful", "charmed", "lighthearted", "sanguine", "spirited", "joyous", "upbeat", "sunny", "overjoyed", "sparkling",
    "grinning", "rejoicing", "euphoric", "celebratory", "appreciative", "calm", "secure", "playful", "smiling", "spirited",
    "thankful", "supportive", "compassionate", "joy-spreading", "sincere", "faithful", "admiring", "cheer-spreading", "friendly",
    "devoted", "glowing", "radiating", "bright", "giddy", "zestful", "breezy", "mirthful", "light", "carefree",
    "soulful", "gentle", "glorious", "honored", "helpful", "wholehearted", "enthused", "bubbly", "witty", "perky",
    "brave", "mindful", "introspective", "refreshed", "rejuvenated", "respectful", "romantic", "sentimental", "chipper", "gleaming",
    # repeat variations or synonyms to reach 300
] * 4 
    negative_personas = [
    "sad", "depressed", "dismayed", "gloomy", "melancholy", "downcast", "unhappy", "miserable", "hopeless", "disheartened",
    "discouraged", "grieving", "sorrowful", "anguished", "tearful", "regretful", "hurt", "betrayed", "broken", "lonely",
    "isolated", "resentful", "frustrated", "tired", "exhausted", "anxious", "panicked", "ashamed", "disgusted", "angry",
    "furious", "outraged", "irritated", "annoyed", "restless", "bored", "apathetic", "insecure", "embarrassed", "humiliated",
    "fearful", "terrified", "scared", "nervous", "tense", "uneasy", "paranoid", "distrustful", "jealous", "envious",
    "resentful", "moody", "weary", "doubtful", "worthless", "invisible", "ignored", "neglected", "shamed", "offended",
    "disturbed", "distraught", "empty", "pained", "defeated", "vulnerable", "devastated", "lost", "dreadful", "confused",
    "critical", "pessimistic", "cynical", "brokenhearted", "hollow", "suffering", "tormented", "self-loathing", "paralyzed", "guilty",
    "regretful", "mean", "withdrawn", "timid", "awkward", "afraid", "numb", "detached", "cold", "bitter",
    "worried", "choked", "dejected", "neglectful", "bleak", "resenting", "stressed", "pressured", "blaming", "distancing",
    # repeat variations or synonyms to reach 300
] * 4

    dataset = []
    for iteration in range(1):
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



def evaluate_model_pix2pix(model, tokenizer, prompt, generators=None):
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_settings = {
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": False,
        "max_new_tokens": 128,
        "repetition_penalty": 1.1,
    }
    control_vector: dict[int, typing.Any] = {}
    if generators is None:
        # Load all generators from the "pix2pix_generators" directory
        save_dir = "pix2pix_generators"
        generators = sorted(
            os.listdir(save_dir),
            key=lambda x: int(x.split("_")[1].split(".")[0]),
            reverse=True
        )
    #TODO
    batch_size = 2
    inputs = [prompt for i in range(batch_size)]
    n_layers = len(model_layer_list(model))
    hidden_layers = range(-1, -model.config.num_hidden_layers, -1)
    hidden_layers = [i if i >= 0 else n_layers + i for i in hidden_layers]
    train_strs = inputs+inputs # you may need to change this later
    layer_hiddens = batched_get_hiddens(model, tokenizer, train_strs, hidden_layers, batch_size)

    for filename in generators:
        layer = int((filename.split(".")[0]).split("_")[1])+1
        if not filename.endswith(".pt"):
            print("check")
            continue
        # Infer input_dim from model config or load dummy layer if needed
        input_dim = model.config.hidden_size  # adjust if needed
        noise_dim = 16  # must match the value used in training
        generator = MLPGenerator(input_dim=input_dim, noise_dim=noise_dim)
        generator.load_state_dict(torch.load(os.path.join(save_dir, filename)))
        generator.eval()
        # Inference device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator.to(device)
        #prepare inputs for pix2pix model inference
        h = layer_hiddens[layer]
        assert h.shape[0] == len(inputs) * 2
        x = torch.tensor(h[::2], dtype=torch.float32).to(device)
        z = torch.randn(batch_size, 16).to(device)
        out = generator(x, z)
        out = out.cpu().detach().numpy()
        print(out.shape, h.shape) #(1, 4096) (2, 4096)
        
        control_vector[layer] = h[::2]
    control_vector = ControlVector(model_type=model.config.model_type, directions=control_vector)
    #print(control_vector) #TODO
    strength = 1
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
        "pix2pix", 
    ]


    for method in methods:
        # Load tokenizer and model
        tokenizer, model = load_tokenizer_and_model(model_name, hf_token, method)
        # Path to cached dataset
        data_path = "./data/sentiment_data.txt"

        if os.path.exists(data_path):
            print(f"Loading dataset from {data_path}...")
            with open(data_path, "r") as f:
                lines = f.readlines()
    
            # Assume each line is a tab-separated pair: positive \t negative
            dataset = []
            for line in lines:
                if "\t" not in line:
                    continue  # skip malformed lines
                pos, neg = line.strip().split("\t")
                dataset.append(DatasetEntry(positive=pos, negative=neg))

        else:
            print("Generating dataset...")
            dataset = build_dataset(tokenizer, suffixes, user_tag, asst_tag)

            # Save it for reuse
            with open(data_path, "w") as f:
                for entry in dataset:
                    f.write(f"{entry.positive}\t{entry.negative}\n")
         
        print("=== BASELINE ===")
        model.reset()
        print(evaluate_model(model, tokenizer, prompt))

        print(f"\n++ METHOD:PCA_ContrastVector (Positive direction)")
        model.reset()
        control_vector = ControlVector.train(
            model, tokenizer, dataset, method="pca_contrastvector"
        )
        print(evaluate_model(model, tokenizer, prompt, control_vector, strength=1.5))
        

        print(f"\n++ METHOD: {method} (Positive direction with pca_contrastvector )")
        model.reset()
        control_vector = ControlVector.train_wt_pix2pix(
            model, tokenizer, dataset, method=method
        )    
        print(evaluate_model_pix2pix(model, tokenizer, prompt, None))





if __name__ == "__main__":
    main()

