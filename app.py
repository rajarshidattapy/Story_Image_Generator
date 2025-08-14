import os
import logging
import traceback
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

# Logging setup
LOG_FORMAT = "%Y-%m-%d %H:%M:%S | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("story_image_generator")

try:
    hf_client = InferenceClient(token=HF_TOKEN)
except Exception as e:
    logger.critical(f"Failed to initialize Hugging Face client: {e}")
    raise

app = FastAPI(title="Story Generator with LangChain + Ollama + FLUX", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

try:
    llm = OllamaLLM(model="llama3.2:1b", base_url="http://localhost:11434")
except Exception as e:
    logger.critical(f"Failed to initialize Ollama LLM: {e}")
    raise

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate", response_class=HTMLResponse)
async def generate_story(request: Request, prompt: str = Form(...)):
    try:
        # --- Generate story and descriptions ---
        story_prompt = PromptTemplate.from_template(
            "Write a short story (200-300 words) based on this prompt: {prompt}. Make it engaging and creative."
        )
        try:
            story = llm.invoke(story_prompt.format(prompt=prompt)).strip()
            logger.info("Story generated successfully.")
        except Exception as e:
            logger.error(f"Error generating story: {e}\n{traceback.format_exc()}")
            raise

        char_prompt = PromptTemplate.from_template(
            "Based on this story: {story}, create a detailed character description "
            "(100-150 words) including appearance, personality, and distinctive features."
        )
        try:
            character_desc = llm.invoke(char_prompt.format(story=story)).strip()
            logger.info("Character description generated successfully.")
        except Exception as e:
            logger.error(f"Error generating character description: {e}\n{traceback.format_exc()}")
            raise

        bg_prompt = PromptTemplate.from_template(
            "Based on this story: {story}, create a detailed background/scene description "
            "(100-150 words) including environment, atmosphere, time period, and visual details."
        )
        try:
            background_desc = llm.invoke(bg_prompt.format(story=story)).strip()
            logger.info("Background description generated successfully.")
        except Exception as e:
            logger.error(f"Error generating background description: {e}\n{traceback.format_exc()}")
            raise

        # --- Build image prompts ---
        character_image_prompt = f"Portrait of a character: {character_desc}. High quality, detailed, artistic style."
        background_image_prompt = f"Background scene: {background_desc}. High quality, detailed, artistic style, matches character."

        # --- Generate images using FLUX ---
        try:
            char_image = hf_client.text_to_image(character_image_prompt, model="black-forest-labs/FLUX.1-dev")
            logger.info("Character image generated successfully.")
        except Exception as e:
            logger.error(f"Error generating character image: {e}\n{traceback.format_exc()}")
            raise
        try:
            bg_image = hf_client.text_to_image(background_image_prompt, model="black-forest-labs/FLUX.1-dev")
            logger.info("Background image generated successfully.")
        except Exception as e:
            logger.error(f"Error generating background image: {e}\n{traceback.format_exc()}")
            raise

        # --- Save images ---
        try:
            os.makedirs("static", exist_ok=True)
            char_path = "static/character.png"
            bg_path = "static/background.png"
            # Ensure images are PIL.Image.Image instances
            from PIL import Image as PILImage
            if not isinstance(char_image, PILImage.Image):
                char_image = PILImage.open(char_image)
            if not isinstance(bg_image, PILImage.Image):
                bg_image = PILImage.open(bg_image)
            char_image.save(char_path)
            bg_image.save(bg_path)
            logger.info("Images saved successfully.")
        except Exception as e:
            logger.error(f"Error saving images: {e}\n{traceback.format_exc()}")
            raise

        # --- Merge images ---
        try:
            from PIL import Image as PILImage
            if not isinstance(bg_image, PILImage.Image):
                bg_image = PILImage.open(bg_image)
            if not isinstance(char_image, PILImage.Image):
                char_image = PILImage.open(char_image)
            bg_image = bg_image.convert("RGBA")
            char_image = char_image.convert("RGBA")

            merged_image = bg_image.copy()
            char_resized = char_image.resize((int(bg_image.width / 2), int(bg_image.height / 2)))

            merged_image.paste(char_resized, (bg_image.width // 4, bg_image.height // 4), mask=char_resized)
            merged_path = "static/merged.png"
            merged_image.save(merged_path)
            logger.info("Images merged and saved successfully.")
        except Exception as e:
            logger.error(f"Error merging images: {e}\n{traceback.format_exc()}")
            raise

        # --- Render results in index.html ---
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prompt": prompt,
                "story": story,
                "character_desc": character_desc,
                "background_desc": background_desc,
                "merged_image_url": f"/{merged_path}"
            },
        )

    except Exception as e:
        logger.error(f"Unhandled error in /generate: {e}\n{traceback.format_exc()}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(e)}
        )

@app.get("/health")
async def health_check():
    try:
        return {"status": "healthy", "message": "Story Generator is running"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.critical(f"Failed to start server: {e}\n{traceback.format_exc()}")
