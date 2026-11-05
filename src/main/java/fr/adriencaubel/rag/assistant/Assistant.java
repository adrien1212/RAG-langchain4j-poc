package fr.adriencaubel.rag.assistant;

import dev.langchain4j.service.SystemMessage;

// Interface simple : AiServices.builder(Assistant.class) génère l'implémentation (proxy) dans RagApplication
public interface Assistant {

    @SystemMessage("Answer ONLY from the provided documents. If the answer is not there, say you don't know.")
    String chat(String userMessage);
}
