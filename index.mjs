import OpenAI from 'openai';
import { pipeline } from '@xenova/transformers';
import * as readline from "node:readline/promises";
import fs from "fs/promises";
import "dotenv/config";

async function loadData(filePath = "./data.json") {
  try {
    const fileContent = await fs.readFile(filePath, "utf8");
    return JSON.parse(fileContent);
  } catch (error) {
    console.error("Помилка під час завантаження файлу:", error);
    return [];
  }
}

function formatDocuments(rawData) {
  return rawData.map((item) => ({
    pageContent: `Title: ${item.title}\nContent: ${item.content}`,
    metadata: {
      id: item.id,
      title: item.title,
    },
  }));
}

function initializeOpenRouter() {
  return new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: "https://openrouter.ai/api/v1",
    defaultHeaders: {
      "HTTP-Referer": "http://localhost:3000",
      "X-Title": "RAG Search App"
    }
  });
}

async function initializeEmbeddings() {
  console.log("Завантаження моделі для embeddings...");
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  return extractor;
}

async function getEmbeddings(extractor, text) {
  const output = await extractor(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

class CustomVectorStore {
  constructor() {
    this.vectors = [];
    this.documents = [];
  }

  async addVectors(vectors, documents) {
    this.vectors = vectors;
    this.documents = documents;
  }

  async similaritySearch(query, k = 3) {
    const queryEmbedding = await getEmbeddings(this.extractor, query);
    
    const similarities = this.vectors.map((vector, index) => {
      const similarity = cosineSimilarity(queryEmbedding, vector);
      return { similarity, document: this.documents[index] };
    });

    similarities.sort((a, b) => b.similarity - a.similarity);

    return similarities.slice(0, k).map(item => item.document);
  }

  setExtractor(extractor) {
    this.extractor = extractor;
  }
}

function cosineSimilarity(a, b) {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

async function createVectorStore(docs, extractor) {
  console.log("Створення векторного сховища...");
  const embeddings = await Promise.all(
    docs.map(doc => getEmbeddings(extractor, doc.pageContent))
  );
  
  const vectorStore = new CustomVectorStore();
  vectorStore.setExtractor(extractor);
  await vectorStore.addVectors(embeddings, docs);
  
  return vectorStore;
}

async function searchDocuments(vectorStore, query) {
  return await vectorStore.similaritySearch(query, 3);
}

function formatPrompt(query, documents) {
  const context = documents.map(doc => doc.pageContent).join("\n\n");
  return `На основі наступного контексту, відповісти на запит. Якщо відповідь не знайдена в контексті, скажи про це.

Контекст:
${context}

Запит: ${query}

Відповідь:`;
}

async function getAnswer(client, prompt) {
  const response = await client.chat.completions.create({
    model: "openai/gpt-3.5-turbo",
    messages: [{ role: "user", content: prompt }],
    temperature: 0
  });
  return response.choices[0].message.content;
}

async function startChat(vectorStore, client) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  while (true) {
    const query = await rl.question("Введіть ваш запит (або 'exit' для виходу): ");
    if (query.toLowerCase() === "exit") {
      rl.close();
      break;
    }

    try {
      const relevantDocs = await searchDocuments(vectorStore, query);
      const prompt = formatPrompt(query, relevantDocs);
      const answer = await getAnswer(client, prompt);
      
      console.log("\n🔹 Відповідь:", answer);
      console.log("\n📄 Джерела:");
      const seen = new Set();
      relevantDocs.forEach((doc) => {
        if (!seen.has(doc.metadata.title)) {
          console.log(`- ${doc.metadata.title}`);
          seen.add(doc.metadata.title);
        }
      });
    } catch (error) {
      console.error("Помилка під час обробки запиту:", error);
    }

    console.log("\n-------------------\n");
  }
}
async function runRAG() {
  const rawData = await loadData();
  if (!rawData.length) {
    console.log("Дані не знайдені або порожні.");
    return;
  }

  const docs = formatDocuments(rawData);
  const client = initializeOpenRouter();
  const extractor = await initializeEmbeddings();
  const vectorStore = await createVectorStore(docs, extractor);

  console.log("Система RAG готова до використання.");
  await startChat(vectorStore, client);
}

await runRAG();