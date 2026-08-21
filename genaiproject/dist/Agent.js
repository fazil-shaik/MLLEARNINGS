import { createAgent, tool } from "langchain";
import * as z from "zod";
const googleSearch = tool(async ({ query }) => {
    const params = new URLSearchParams({
        engine: "google",
        q: query,
        api_key: process.env.SERPAPI_API_KEY,
    });
    const response = await fetch(`https://serpapi.com/search?${params.toString()}`);
    if (!response.ok) {
        throw new Error(`SerpApi error: ${response.status}`);
    }
    const data = await response.json();
    console.log(data);
    return (data.organic_results ?? [])
        .slice(0, 5)
        .map((result) => `${result.title}\n${result.snippet}\n${result.link}`)
        .join("\n\n");
}, {
    name: "google_search",
    description: "Search Google for current information, news, and up-to-date facts.",
    schema: z.object({
        query: z.string().describe("The search query"),
    }),
});
const Answer = z.object({
    summary: z.string(),
    confidence: z.number(),
});
const agent = createAgent({
    model: "google-genai:gemini-3.6-flash",
    tools: [googleSearch],
    responseFormat: Answer,
});
const result = await agent.invoke({
    messages: [
        {
            role: "user",
            content: "Summarize the latest AI trends",
        },
    ],
});
console.log(result.structuredResponse);
//# sourceMappingURL=Agent.js.map