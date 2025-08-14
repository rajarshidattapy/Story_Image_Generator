import os
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from PIL import Image
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

# Hugging Face client
hf_client = InferenceClient(token=HF_TOKEN)

# FastAPI setup
app = FastAPI(title="Story Generator with LangChain + Ollama + FLUX", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Ollama LLM
llm = OllamaLLM(model="llama3.2:1b", base_url="http://localhost:11434")

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
        story = llm.invoke(story_prompt.format(prompt=prompt)).strip()

        char_prompt = PromptTemplate.from_template(
            "Based on this story: {story}, create a detailed character description "
            "(100-150 words) including appearance, personality, and distinctive features."
        )
        character_desc = llm.invoke(char_prompt.format(story=story)).strip()

        bg_prompt = PromptTemplate.from_template(
            "Based on this story: {story}, create a detailed background/scene description "
            "(100-150 words) including environment, atmosphere, time period, and visual details."
        )
        background_desc = llm.invoke(bg_prompt.format(story=story)).strip()

        # --- Build image prompts ---
        character_image_prompt = f"Portrait of a character: {character_desc}. High quality, detailed, artistic style."
        background_image_prompt = f"Background scene: {background_desc}. High quality, detailed, artistic style, matches character."

        # --- Generate images using FLUX ---
        char_image = hf_client.text_to_image(character_image_prompt, model="black-forest-labs/FLUX.1-dev")
        bg_image = hf_client.text_to_image(background_image_prompt, model="black-forest-labs/FLUX.1-dev")

        # --- Save images ---
        os.makedirs("static", exist_ok=True)
        char_path = "static/character.png"
        bg_path = "static/background.png"
        char_image.save(char_path)
        bg_image.save(bg_path)

        # --- Merge images ---
        bg_image = bg_image.convert("RGBA")
        char_image = char_image.convert("RGBA")

        merged_image = bg_image.copy()
        char_resized = char_image.resize((int(bg_image.width / 2), int(bg_image.height / 2)))

        merged_image.paste(char_resized, (bg_image.width // 4, bg_image.height // 4), mask=char_resized)
        merged_path = "static/merged.png"
        merged_image.save(merged_path)


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
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(e)}
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Story Generator is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
