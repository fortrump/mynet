from cog import BasePredictor, Input, Path
import torch
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel
from diffusers.utils import load_image
import os
import subprocess
import time

MODEL_CACHE = "FLUX.1-dev"
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
CONTROLNET_CACHE = "FLUX.1-dev-ControlNet-Union-Pro"
CONTROLNET_MODEL_UNION = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"
MODEL_URL = (
    "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
)
CONTROLNET_URL = "https://weights.replicate.delivery/default/shakker-labs/FLUX.1-dev-ControlNet-Union-Pro/model.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        print("Starting setup...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        try:
            # Download models if they don't exist
            if not os.path.exists(CONTROLNET_CACHE):
                print(f"Downloading controlnet model...")
                download_weights(CONTROLNET_URL, CONTROLNET_CACHE)

            if not os.path.exists(MODEL_CACHE):
                print(f"Downloading base model...")
                download_weights(MODEL_URL, ".")

            print("Loading controlnet model...")
            controlnet_union = FluxControlNetModel.from_pretrained(
                CONTROLNET_CACHE, torch_dtype=torch.bfloat16
            )
            self.controlnet = FluxMultiControlNetModel([controlnet_union])
            print("ControlNet model loaded successfully!")

            print("Loading base model...")
            self.pipe = FluxControlNetPipeline.from_pretrained(
                MODEL_CACHE, controlnet=self.controlnet, torch_dtype=torch.bfloat16
            ).to(self.device)
            print("Setup completed successfully!")
        except Exception as e:
            print(f"Error during setup: {str(e)}")
            raise

    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        control_image: Path = Input(description="Control image path"),
        num_inference_steps: int = Input(
            description="Number of denoising steps", default=42, ge=1, le=100
        ),
        guidance_scale: float = Input(
            description="Guidance scale for classifier-free guidance",
            default=4.4,
            ge=1.0,
            le=20.0,
        ),
        seed: int = Input(description="Random seed for generation", default=42),
    ) -> Path:
        """Run a single prediction on the model"""
        try:
            # Load and process control image
            control_image = load_image(str(control_image))
            control_mode_depth = 4

            # Set fixed dimensions
            width, height = 1080, 1080

            # Generate image
            images = self.pipe(
                prompt,
                control_image=[control_image],
                control_mode=[control_mode_depth],
                width=width,
                height=height,
                controlnet_conditioning_scale=[0.6],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.manual_seed(seed),
            ).images

            # Save the first generated image
            output_path = Path("/tmp/output.png")
            images[0].save(str(output_path))
            return output_path

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
