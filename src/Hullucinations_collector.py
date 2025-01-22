import torch
from functools import partial
from captum.attr import IntegratedGradients
from transformers import LlamaTokenizer, LlamaForCausalLM
from typing import Any, Dict
import numpy as np
import os

class IntegratedGradientsAnalyzer:
    def __init__(self, model_dir: str, gpu: str = "0", model_name: str = "open_llama_7b"):
        self.model_name = model_name
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir

        # Load model and tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(model_dir)
        self.model = LlamaForCausalLM.from_pretrained(model_dir,
                                                     cache_dir=model_dir,
                                                     device_map=self.device,
                                                     torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)
        self.forward_func = partial(self.model_forward, model=self.model, extra_forward_args={})
        self.embedder = self.get_embedder(self.model)

        # Integrated Grads Parameters
        self.ig_steps = 64
        self.internal_batch_size = 4

    def get_next_token(self, x, model):
        with torch.no_grad():
            return model(x).logits

    def normalize_attributes(self, attributes: torch.Tensor) -> torch.Tensor:
        attributes = attributes.squeeze(0)
        norm = torch.norm(attributes, dim=1)
        attributes = norm / torch.sum(norm)
        return attributes

    def model_forward(self, input_: torch.Tensor, model, extra_forward_args: Dict[str, Any]) -> torch.Tensor:
        output = model(inputs_embeds=input_, **extra_forward_args)
        return torch.nn.functional.softmax(output.logits[:, -1, :], dim=-1)

    def get_embedder(self, model):
        if "falcon" in self.model_name:
            return model.transformer.word_embeddings
        elif "opt" in self.model_name:
            return model.model.decoder.embed_tokens
        elif "llama" in self.model_name:
            return model.model.embed_tokens
        else:
            raise ValueError(f"Unknown model {self.model_name}")

    def get_ig(self, prompt: str) -> np.ndarray:
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
        prediction_id = self.get_next_token(input_ids, self.model).squeeze()[-1].argmax()
        encoder_input_embeds = self.embedder(input_ids).detach()
        ig = IntegratedGradients(forward_func=self.forward_func)
        attributes = self.normalize_attributes(
            ig.attribute(
                encoder_input_embeds,
                target=prediction_id,
                n_steps=self.ig_steps,
                internal_batch_size=self.internal_batch_size
            )
        ).detach().cpu().numpy()
        return attributes

    def analyze_text(self, text: str) -> np.ndarray:
        return self.get_ig(text)


if __name__ == '__main__':
    model_dir = "/home/disk2/ghh/llama"
    analyzer = IntegratedGradientsAnalyzer(model_dir=model_dir)

    input_text = "Where was Albert Einstein born?"
    attributes_first = analyzer.analyze_text(input_text)

    print("Integrated Gradients Attributes for the input text:", attributes_first)
