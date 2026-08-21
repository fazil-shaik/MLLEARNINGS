import { initChatModel } from "langchain";
import dotenv from "dotenv";

dotenv.config();
const apiKey = process.env.GOOGLE_API_KEY 

const model = await initChatModel("google-genai:gemini-3.7-flash");

const response = await model.invoke("Why do parrots talk give me one sentence?");

console.log(response.content)