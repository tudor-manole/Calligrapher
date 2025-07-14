import torch
from PIL import Image

from models.calligrapher import Calligrapher
from models.transformer_flux_inpainting import FluxTransformer2DModel
from pipeline_calligrapher import CalligrapherPipeline

class CalligrapherLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {"multiline": False}),
                "image_encoder_path": ("STRING", {"multiline": False}),
                "calligrapher_path": ("STRING", {"multiline": False}),
                "device": ("STRING", {"default": "cuda"}),
                "num_tokens": ("INT", {"default": 128}),
            }
        }

    RETURN_TYPES = ("CALLIGRAPHER_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "calligrapher"

    def load_model(self, base_model_path, image_encoder_path, calligrapher_path, device="cuda", num_tokens=128):
        transformer = FluxTransformer2DModel.from_pretrained(
            base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        pipe = CalligrapherPipeline.from_pretrained(
            base_model_path, transformer=transformer, torch_dtype=torch.bfloat16
        ).to(device)
        model = Calligrapher(pipe, image_encoder_path, calligrapher_path, device=device, num_tokens=num_tokens)
        return (model,)


class CalligrapherGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CALLIGRAPHER_MODEL",),
                "prompt": ("STRING", {"multiline": True}),
                "reference_image": ("IMAGE",),
                "source_image": ("IMAGE",),
                "mask": ("IMAGE",),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "seed": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "calligrapher"

    def generate(self, model, prompt, reference_image, source_image, mask, steps=50, scale=1.0, seed=0):
        ref_pil = Image.fromarray(reference_image)
        src_pil = Image.fromarray(source_image)
        mask_pil = Image.fromarray(mask)
        model.set_scale(scale)
        images = model.generate(
            image=src_pil,
            mask_image=mask_pil,
            ref_image=ref_pil,
            prompt=prompt,
            num_inference_steps=steps,
            seed=seed,
            width=src_pil.width,
            height=src_pil.height,
        )
        return (images[0],)


NODE_CLASS_MAPPINGS = {
    "CalligrapherLoader": CalligrapherLoader,
    "CalligrapherGenerate": CalligrapherGenerate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CalligrapherLoader": "Calligrapher Loader",
    "CalligrapherGenerate": "Calligrapher Generate",
}
