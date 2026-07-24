from openrouter import OpenRouter
import os
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv("OPENROUTER_API_KEY")


console = Console()

with OpenRouter(api_key=api_key) as client:
    response = client.chat.send(
        model="xiaomi/mimo-v2.5",
        messages=[
            {"role": "user", "content": "what is purpose of life"}
        ],
    )

    content = response.choices[0].message.content
    console.print(Markdown(content))


