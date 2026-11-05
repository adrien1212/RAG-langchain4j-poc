package fr.adriencaubel.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import fr.adriencaubel.rag.assistant.Assistant;

import java.util.List;
import java.util.Scanner;

public class RagApplication {

	private static final String OLLAMA_URL = "http://localhost:11434";

	public static void main(String[] args) {
		// 1. Modèles : le même modèle d'embedding sert à l'ingestion ET à la recherche
		EmbeddingModel embeddingModel = OllamaEmbeddingModel.builder()
				.baseUrl(OLLAMA_URL)
				.modelName("bge-m3")
				.build();

		ChatModel chatModel = OllamaChatModel.builder()
				.baseUrl(OLLAMA_URL)
				.modelName("llama3.2")
				.build();

		// 2. Store : InMemory ici, une BDD vectorielle (pgvector...) plus tard
		EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

		// 3. Ingestion : load (classpath = src/main/resources) + parse (Tika) -> split -> embed -> store
		List<Document> documents = ClassPathDocumentLoader.loadDocuments("docs", new ApacheTikaDocumentParser());
		System.out.println(documents.size() + " documents chargés");

		DocumentSplitter splitter = DocumentSplitters.recursive(
				500,  // max segment size (in characters)
				50);  // overlap between two consecutive segments (in characters)

		EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
				.documentSplitter(splitter)
				.embeddingModel(embeddingModel)
				.embeddingStore(embeddingStore)
				.build();

		ingestor.ingest(documents);

		// 4. Retrieval : question -> embedding -> segments les plus proches
		ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
				.embeddingStore(embeddingStore)
				.embeddingModel(embeddingModel)
				.maxResults(3)
				.minScore(0.6)
				.build();

		RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
				 .contentRetriever(contentRetriever)
				 .build();

		// 5. AI Service : proxy qui assemble system message + segments + question, puis appelle le LLM
		Assistant assistant = AiServices.builder(Assistant.class)
				.chatModel(chatModel)
				.retrievalAugmentor(retrievalAugmentor)
				.build();

		// 6. Boucle console
		try (Scanner scanner = new Scanner(System.in)) {
			while (true) {
				System.out.print("\nQuestion (exit pour quitter) > ");
				if (!scanner.hasNextLine()) {
					break;
				}
				String question = scanner.nextLine().trim();
				if (question.equalsIgnoreCase("exit")) {
					break;
				}
				if (!question.isEmpty()) {
					System.out.println(assistant.chat(question));
				}
			}
		}
	}

}
