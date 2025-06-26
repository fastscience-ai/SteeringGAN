import dataclasses
import typing
import warnings

import torch
from transformers import PretrainedConfig, PreTrainedModel

if typing.TYPE_CHECKING:
    from .extract import ControlVector


class ControlModel(torch.nn.Module):
    """
    **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

    A wrapped language model that can have controls set on its layers with `self.set_control`.
    """

    def __init__(self, model: PreTrainedModel, layer_ids: typing.Iterable[int], method):
        """
        **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

        Build a new ControlModel around a model instance, initializing control on
        the layers specified in `layer_ids`.
        """

        super().__init__()
        self.model = model
        self.method = method
        layers = model_layer_list(model)
        self.layer_ids = [i if i >= 0 else len(layers) + i for i in layer_ids]
        for layer_id in layer_ids:
            layer = layers[layer_id]
            if not isinstance(layer, ControlModule):
                layers[layer_id] = ControlModule(layer)
            else:
                warnings.warn(
                    "Trying to rewrap a wrapped model! Probably not what you want! Try calling .unwrap first."
                )

    @property
    def config(self) -> PretrainedConfig:
        return self.model.config

    @property
    def device(self) -> torch.device:
        return self.model.device

    def unwrap(self) -> PreTrainedModel:
        """
        Removes the mutations done to the wrapped model and returns it.
        After using this method, `set_control` and `reset` will not work.
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layers[layer_id] = layers[layer_id].block
        return self.model

#    def set_control(
#        self, control: "ControlVector", coeff: float = 1.0, **kwargs
#    ) -> None:
#        """
#        Set a `ControlVector` for the layers this ControlModel handles, with a strength given
#        by `coeff`. (Negative `coeff` values invert the control vector, e.g. happiness→sadness.)
#        `coeff` defaults to `1.0`.
#
#        Additional kwargs:
#        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale the
#          activation to that magnitude after control (default: `False`)
#        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and control
#          (default: +)
#        """
#
#        raw_control = {}
#        for layer_id in self.layer_ids:
#            raw_control[layer_id] = torch.tensor(
#                coeff * control.directions[layer_id]
#            ).to(self.model.device, dtype=self.model.dtype)
#        if self.method != "pix2pix":
#            self.set_raw_control(raw_control, **kwargs)
#        else:
#            self.set_raw_control_gan(raw_control, **kwargs)
    ####TODO SOO
    def set_control(
        self,
        control: "ControlVector | dict[int, typing.Any]",  # Either directions or generators
        coeff: float = 1.0,
        **kwargs
    ) -> None:
        """
        Unified control setter: either vector-based (SCAV/PCA) or GAN-based (pix2pix).
        Requires:
            - method = "pix2pix" → control = {layer_id: generator}
            - method ≠ "pix2pix" → control = ControlVector with .directions[layer_id]
    
        kwargs:
            - prompt: str (for pix2pix)
            - tokenizer: tokenizer object
            - noise_dim: int (default = 16)
        """
        raw_control = {}

        if self.method != "pix2pix":
        # === Vector-based steering (SCAV/PCA) ===
            for layer_id in self.layer_ids:
                raw_control[layer_id] = torch.tensor(
                    coeff * control.directions[layer_id]
                ).to(self.model.device, dtype=self.model.dtype)

            self.set_raw_control(raw_control, **kwargs)

        else:
            for layer_id in self.layer_ids:
                raw_control[layer_id] = torch.tensor(
                    control.directions[layer_id]
                ).to(self.model.device, dtype=self.model.dtype)

            self.set_raw_control_pix2pix(raw_control, **kwargs)

            # === pix2pix GAN-based control ===
        #    assert isinstance(control, dict), "pix2pix expects a dict[layer_id] = generator"

        #    prompt = kwargs["prompt"]
        #    tokenizer = kwargs["tokenizer"]
        #    noise_dim = kwargs.get("noise_dim", 16)

        #    input_ids = tokenizer(prompt, return_tensors="pt").to(self.model.device)
        #    with torch.no_grad():
        #        outputs = self.model.model(**input_ids, output_hidden_states=True)

        #       for layer_id in self.layer_ids:
        #            generator = control[layer_id]
        #            hidden = outputs.hidden_states[layer_id]  # (1, seq_len, dim)
        #            rep = hidden[:, -1]  # use last token's embedding
        #            z = torch.randn(1, noise_dim).to(self.model.device)
        #            vec = generator(rep, z).squeeze(0) * coeff
        #            raw_control[layer_id] = vec.to(self.model.device, dtype=self.model.dtype)

         #   self.set_raw_control_gan(raw_control, **kwargs)

    def reset(self) -> None:
        """
        Resets the control for all layer_ids, returning the model to base behavior.
        """
        self.set_raw_control(None)

    def set_raw_control(
        self, control: dict[int, torch.Tensor] | None, **kwargs
    ) -> None:
        """
        Set or remove control parameters to the layers this ControlModel handles.
        The keys of `control` should be equal to or a superset of the `layer_ids` passed to __init__.
        Only those layers will be controlled, any others in `control` will be ignored.

        Passing `control=None` will reset the control tensor for all layer_ids, making the model act
        like a non-control model.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale the
          activation to that magnitude after control (default: `False`)
        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and control
          (default: +)
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layer: ControlModule = layers[layer_id]  # type: ignore
            if control is None:
                layer.reset()
            else:
                layer.set_control(BlockControlParams(control[layer_id], **kwargs))

    def set_raw_control_pix2pix(
        self, control: dict[int, torch.Tensor] | None, **kwargs
    ) -> None:
        """
        Set or remove control parameters to the layers this ControlModel handles.
        The keys of `control` should be equal to or a superset of the `layer_ids` passed to __init__.
        Only those layers will be controlled, any others in `control` will be ignored.

        Passing `control=None` will reset the control tensor for all layer_ids, making the model act
        like a non-control model.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale the
          activation to that magnitude after control (default: `False`)
        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and control
          (default: +)
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layer: ControlModule = layers[layer_id]  # type: ignore
            if control is None:
                layer.reset()
            else:
                layer.set_control(BlockControlParams_GAN(control[layer_id], **kwargs))


    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@dataclasses.dataclass
class BlockControlParams:
    control: torch.Tensor | None = None
    normalize: bool = False
    operator: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        lambda current, control: current + control
    )

    @classmethod
    def default(cls) -> "BlockControlParams":
        return cls()

#SOO
@dataclasses.dataclass
class BlockControlParams_GAN:
    control: torch.Tensor | None = None
    normalize: bool = False
    operator: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        lambda current, control: control
    )

    @classmethod
    def default(cls) -> "BlockControlParams_GAN":
        return cls()


class ControlModule(torch.nn.Module):
    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self.block: torch.nn.Module = block
        
        self.params: BlockControlParams = BlockControlParams.default()

    def set_control(self, params: BlockControlParams) -> None:
        self.params = params

    def set_control_gan(self, params: BlockControlParams_GAN) -> None:
        self.params = params

    def reset(self) -> None:
        self.set_control(BlockControlParams.default())

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)

        control = self.params.control

        if control is None:
            return output
        elif len(control.shape) == 1:
            control = control.reshape(1, 1, -1)
        elif len(control.shape)==2:
            control = control.reshape(1,1,-1)

        if isinstance(output, tuple):
            modified = output[0]
        else:
            modified = output
        #print(control.shape, modified.shape, len(control.shape), len(modified.shape)) 
        #torch.Size([1, 1, 4096]) torch.Size([1, 1, 4096]) 3 3
        #torch.Size([1, 4096]) torch.Size([1, 14, 4096])
        assert len(control.shape) == len(modified.shape)
        control = control.to(modified.device)

        norm_pre = torch.norm(modified, dim=-1, keepdim=True)

        # we should ignore the padding tokens when doing the activation addition
        # mask has ones for non padding tokens and zeros at padding tokens.
        # only tested this on left padding
        if "position_ids" in kwargs:
            pos = kwargs["position_ids"]
            zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
            col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
            target_shape = modified.shape
            mask = (
                (col_indices >= zero_indices)
                .float()
                .reshape(target_shape[0], target_shape[1], 1)
            )
            mask = mask.to(modified.dtype).to(modified.device)
        else:
            mask = 1.0

        modified = self.params.operator(modified, control * mask)

        if self.params.normalize:
            norm_post = torch.norm(modified, dim=-1, keepdim=True)
            modified = modified / norm_post * norm_pre

        if isinstance(output, tuple):
            output = (modified,) + output[1:]
        else:
            output = modified

        return output


def model_layer_list(model: ControlModel | PreTrainedModel) -> torch.nn.ModuleList:
    if isinstance(model, ControlModel):
        model = model.model

    if hasattr(model, "model"):  # mistral-like
        return model.model.layers
    elif hasattr(model, "transformer"):  # gpt-2-like
        return model.transformer.h
    else:
        raise ValueError(f"don't know how to get layer list for {type(model)}")
