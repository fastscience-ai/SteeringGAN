from classifier_manager import ClassifierManager
import torch

class Perturbation:
    def __init__(self, classifier_manager: ClassifierManager, target_probability: float=0.001, accuracy_threshold: float=0.9, perturbed_layers: list[int]=None):
        self.classifier_manager = classifier_manager
        self.target_probability = target_probability
        self.accuracy_threshold = accuracy_threshold
        self.perturbed_layers = perturbed_layers

    def get_perturbation(self, output_hook: torch.Tensor, layer: int) -> torch.Tensor:
        if self.perturbed_layers is None or layer in self.perturbed_layers:
            if self.classifier_manager.testacc[layer] > self.accuracy_threshold and \
                self.classifier_manager.classifiers[layer].predict_proba(output_hook[0][:, -1, :]) > self.target_probability:
                perturbed_embds = self.classifier_manager.cal_perturbation(
                    embds_tensor=output_hook[0][:, -1, :],
                    layer=layer,
                    target_prob=self.target_probability,
                )
                output_hook[0][:, -1, :] = perturbed_embds
        return output_hook