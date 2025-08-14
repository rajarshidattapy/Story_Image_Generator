import os
from huggingface_hub import InferenceClient
import dotenv
dotenv.load_dotenv()

client = InferenceClient()

# output is a PIL.Image object
image = client.text_to_image(
    prompt="Astronaut riding a horse",
    model="black-forest-labs/FLUX.1-dev"
)

# Save result
image.save("astronaut.png")
