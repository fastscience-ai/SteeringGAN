import dataclasses
import os
import typing
import warnings
from sklearn.linear_model import LogisticRegression
import gguf
import numpy as np
from sklearn.decomposition import PCA
import torch
import os 
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .control import ControlModel, model_layer_list
from .saes import Sae
from .train_embedding_pix2pix import *


@dataclasses.dataclass
class DatasetEntry:
    positive: str
    negative: str


@dataclasses.dataclass
class ControlVector:
    model_type: str
    directions: dict[int, np.ndarray]

    @classmethod
    def train(
        cls,
        model: "PreTrainedModel | ControlModel",
        tokenizer: PreTrainedTokenizerBase,
        dataset: list[DatasetEntry],
        **kwargs,
    ) -> "ControlVector":
        """
        Train a ControlVector for a given model and tokenizer using the provided dataset.

        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            dataset (list[DatasetEntry]): The dataset used for training.
            **kwargs: Additional keyword arguments.
                max_batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                method (str, optional): The training method to use. Can be either
                    "pca_diff" or "pca_center". Defaults to "pca_diff".

        Returns:
            ControlVector: The trained vector.
        """
        method = kwargs.get("method", "pca_diff")
        if method not in ["scav", "pix2pix"]:
            with torch.inference_mode():
                dirs = read_representations(
                     model,
                     tokenizer,
                     dataset,
                     **kwargs,
                 )
            return cls(model_type=model.config.model_type, directions=dirs)
        elif method =="scav" :
            print("train funciton is not competible with the SCAV method")
        elif method == "pix2pix":
            print("train function is not competible with the Pix2pix method")


    @classmethod
    def train_wt_pix2pix(
        cls,
        model: "PreTrainedModel | ControlModel",
        tokenizer: PreTrainedTokenizerBase,
        dataset: list[DatasetEntry],
        **kwargs,
    ):
        """
        Train a ControlVector for a given model and tokenizer using the provided dataset.

        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            dataset (list[DatasetEntry]): The dataset used for training.
            **kwargs: Additional keyword arguments.
                max_batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                method (str, optional): The training method to use. Can be either
                    "pix2pix".

        Returns:
            ControlVector: The trained generator models.
        """
        method = kwargs.get("method", "pca_diff")
        if method == "pix2pix":
            results = read_representations(
                     model,
                     tokenizer,
                     dataset,
                     **kwargs,
            )
            
            # Create directory if it doesn't exist
            save_dir = "pix2pix_generators"
            os.makedirs(save_dir, exist_ok=True)
            sorted_layers = sorted(results.keys(), reverse=True)
            for i, layer in enumerate(sorted_layers):
                generator = results[layer]
                filename = f"g_{len(results) - 1 - i}.pt"
                torch.save(generator.state_dict(), os.path.join(save_dir, filename))
            return results
        else:
            print("train function is not competible with the method")





    @classmethod
    def train_with_sae(
        cls,
        model: "PreTrainedModel | ControlModel",
        tokenizer: PreTrainedTokenizerBase,
        sae: Sae,
        dataset: list[DatasetEntry],
        *,
        decode: bool = True,
        method: typing.Literal["pix2pix", "scav", "umap", "readingvector", "reading_contrastvector", "pca_contrastvector","pca_readingvector", "pca_center"] = "pca_center",
        **kwargs,
    ) -> "ControlVector":
        """
        Like ControlVector.train, but using an SAE. It's better! WIP.


        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            sae (saes.Sae): See the `saes` module for how to load this.
            dataset (list[DatasetEntry]): The dataset used for training.
            **kwargs: Additional keyword arguments.
                decode (bool, optional): Whether to decode the vector to make it immediately usable.
                    If not, keeps it as monosemantic SAE features for introspection, but you will need to decode it manually
                    to use it. Defaults to True.
                max_batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                method (str, optional): The training method to use. Can be either
                    "pca_diff" or "pca_center". Defaults to "pca_center"! This is different
                    than ControlVector.train, which defaults to "pca_diff".

        Returns:
            ControlVector: The trained vector.
        """

        def transform_hiddens(hiddens: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
            sae_hiddens = {}
            for k, v in tqdm.tqdm(hiddens.items(), desc="sae encoding"):
                sae_hiddens[k] = sae.layers[k].encode(v)
            return sae_hiddens

        with torch.inference_mode():
            dirs = read_representations(
                model,
                tokenizer,
                dataset,
                transform_hiddens=transform_hiddens,
                method=method,
                **kwargs,
            )

            final_dirs = {}
            if decode:
                for k, v in tqdm.tqdm(dirs.items(), desc="sae decoding"):
                    final_dirs[k] = sae.layers[k].decode(v)
            else:
                final_dirs = dirs

        return cls(model_type=model.config.model_type, directions=final_dirs)

    def export_gguf(self, path: os.PathLike[str] | str):
        """
        Export a trained ControlVector to a llama.cpp .gguf file.
        Note: This file can't be used with llama.cpp yet. WIP!

        ```python
        vector = ControlVector.train(...)
        vector.export_gguf("path/to/write/vector.gguf")
        ```
        ```
        """

        arch = "controlvector"
        writer = gguf.GGUFWriter(path, arch)
        writer.add_string(f"{arch}.model_hint", self.model_type)
        writer.add_uint32(f"{arch}.layer_count", len(self.directions))
        for layer in self.directions.keys():
            writer.add_tensor(f"direction.{layer}", self.directions[layer])
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @classmethod
    def import_gguf(cls, path: os.PathLike[str] | str) -> "ControlVector":
        reader = gguf.GGUFReader(path)

        archf = reader.get_field("general.architecture")
        if not archf or not len(archf.parts):
            warnings.warn(".gguf file missing architecture field")
        else:
            arch = str(bytes(archf.parts[-1]), encoding="utf-8", errors="replace")
            if arch != "controlvector":
                warnings.warn(
                    f".gguf file with architecture {arch!r} does not appear to be a control vector!"
                )

        modelf = reader.get_field("controlvector.model_hint")
        if not modelf or not len(modelf.parts):
            raise ValueError(".gguf file missing controlvector.model_hint field")
        model_hint = str(bytes(modelf.parts[-1]), encoding="utf-8")

        directions = {}
        for tensor in reader.tensors:
            if not tensor.name.startswith("direction."):
                continue
            try:
                layer = int(tensor.name.split(".")[1])
            except (IndexError, ValueError):
                raise ValueError(
                    f".gguf file has invalid direction field name: {tensor.name}"
                )
            directions[layer] = tensor.data

        return cls(model_type=model_hint, directions=directions)

    def _helper_combine(
        self, other: "ControlVector", other_coeff: float
    ) -> "ControlVector":
        if self.model_type != other.model_type:
            warnings.warn(
                "Trying to add vectors with mismatched model_types together, this may produce unexpected results."
            )

        model_type = self.model_type
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = self.directions[layer]
        for layer in other.directions:
            other_layer = other_coeff * other.directions[layer]
            if layer in directions:
                directions[layer] = directions[layer] + other_layer
            else:
                directions[layer] = other_layer
        return ControlVector(model_type=model_type, directions=directions)

    def __eq__(self, other: "ControlVector") -> bool:
        if self is other:
            return True

        if self.model_type != other.model_type:
            return False
        if self.directions.keys() != other.directions.keys():
            return False
        for k in self.directions.keys():
            if (self.directions[k] != other.directions[k]).any():
                return False
        return True

    def __add__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, 1)

    def __sub__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for -: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, -1)

    def __neg__(self) -> "ControlVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = -self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __mul__(self, other: int | float | np.number) -> "ControlVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = other * self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __rmul__(self, other: int | float | np.number) -> "ControlVector":
        return self.__mul__(other)

    def __truediv__(self, other: int | float | np.number) -> "ControlVector":
        return self.__mul__(1 / other)







def read_representations(
    model: "PreTrainedModel | ControlModel",
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[DatasetEntry],
    hidden_layers: typing.Iterable[int] | None = None,
    batch_size: int = 32,
    method: typing.Literal[
        "umap", "readingvector", "reading_contrastvector",
        "pca_contrastvector", "pca_readingvector", "pca_center", "scav", "pix2pix",
    ] = "pca_contrastvector",
    transform_hiddens: (
        typing.Callable[[dict[int, np.ndarray]], dict[int, np.ndarray]] | None
    ) = None,
) -> dict[int, typing.Any]:  # Returns either directions or classifiers
    """
    Extract either steering directions or classifiers, depending on the method.
    """
    if not hidden_layers:
        hidden_layers = range(-1, -model.config.num_hidden_layers, -1)

    n_layers = len(model_layer_list(model))
    hidden_layers = [i if i >= 0 else n_layers + i for i in hidden_layers]

    train_strs = [s for ex in inputs for s in (ex.positive, ex.negative)]

    layer_hiddens = batched_get_hiddens(
        model, tokenizer, train_strs, hidden_layers, batch_size
    )

    if transform_hiddens is not None:
        layer_hiddens = transform_hiddens(layer_hiddens)

    results: dict[int, typing.Any] = {}
    for layer in tqdm.tqdm(hidden_layers):
        h = layer_hiddens[layer]
        assert h.shape[0] == len(inputs) * 2

        if method == "scav": #SOO scav method-> return list of classifier models
            pos_train_embds, neg_train_embds = h[::2], h[1::2]
            x = np.concatenate([pos_train_embds, neg_train_embds], axis=0)
            y = np.concatenate([np.ones(len(pos_train_embds)), np.zeros(len(neg_train_embds))], axis=0)
            clf = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000)
            clf.fit(x, y)
            results[layer] = clf
            continue  # skip rest of the loop

        elif method == "pix2pix":
            pos_train_embds, neg_train_embds = h[::2], h[1::2]
            print(len(h[::2]), len(h[1::2]))
            # Train GAN to learn mapping: pos → neg
            generator = train_embedding_pix2pix(pos_train_embds, neg_train_embds, layer=layer)
    
            results[layer] = generator
            continue  # skip rest

        # other methods → compute direction vectors
        if method == "pca_contrastvector":
            train = h[::2] - h[1::2]
        elif method == "pca_readingvector":
            train = h[::2]
        elif method == "reading_contrastvector":
            train = h[::2] - h[1::2]
        elif method == "readingvector":
            train = h[::2]
        elif method == "pca_center":
            center = (h[::2] + h[1::2]) / 2
            train = h
            train[::2] -= center
            train[1::2] -= center
        elif method == "umap":
            train = h
        else:
            raise ValueError("unknown method " + method)

        # compute direction from embeddings
        if method not in ["umap", "readingvector", "reading_contrastvector"]:
            pca_model = PCA(n_components=1, whiten=False).fit(train)
            direction = pca_model.components_.astype(np.float32).squeeze(axis=0)
        elif method == "umap":
            import umap  # type: ignore
            umap_model = umap.UMAP(n_components=1)
            embedding = umap_model.fit_transform(train).astype(np.float32)
            direction = np.sum(train * embedding, axis=0) / np.sum(embedding)
        else:
            direction = np.mean(train, axis=0)

        # align direction so positive > negative
        projected_hiddens = project_onto_direction(h, direction)
        positive_smaller_mean = np.mean([
            projected_hiddens[i] < projected_hiddens[i + 1]
            for i in range(0, len(inputs) * 2, 2)
        ])
        positive_larger_mean = np.mean([
            projected_hiddens[i] > projected_hiddens[i + 1]
            for i in range(0, len(inputs) * 2, 2)
        ])

        if positive_smaller_mean > positive_larger_mean:
            direction *= -1

        results[layer] = direction

    return results



def batched_get_hiddens(
    model,
    tokenizer,
    inputs: list[str],
    hidden_layers: list[int],
    batch_size: int,
) -> dict[int, np.ndarray]:
    """
    Using the given model and tokenizer, pass the inputs through the model and get the hidden
    states for each layer in `hidden_layers` for the last token.

    Returns a dictionary from `hidden_layers` layer id to an numpy array of shape `(n_inputs, hidden_dim)`
    """
    batched_inputs = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]
    hidden_states = {layer: [] for layer in hidden_layers}
    with torch.no_grad():
        for batch in tqdm.tqdm(batched_inputs):
            # get the last token, handling right padding if present
            encoded_batch = tokenizer(batch, padding=True, return_tensors="pt")
            encoded_batch = encoded_batch.to(model.device)
            out = model(**encoded_batch, output_hidden_states=True)
            ##
            #pred_ids = torch.argmax(out.logits, dim=-1)
            #decoded = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            #print(decoded)
            #exit()
            attention_mask = encoded_batch["attention_mask"]
            for i in range(len(batch)):
                last_non_padding_index = (
                    attention_mask[i].nonzero(as_tuple=True)[0][-1].item()
                )
                for layer in hidden_layers:
                    hidden_idx = layer + 1 if layer >= 0 else layer
                    hidden_state = (
                        out.hidden_states[hidden_idx][i][last_non_padding_index]
                        .cpu()
                        .float()
                        .numpy()
                    )
                    hidden_states[layer].append(hidden_state)
            del out

    return {k: np.vstack(v) for k, v in hidden_states.items()}


def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = np.linalg.norm(direction)
    assert not np.isinf(mag)
    return (H @ direction) / mag
