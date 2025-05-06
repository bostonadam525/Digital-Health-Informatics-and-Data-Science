# ASR - Automatic Speech Recognition
* A repo devoted to all things ASR in Clinical Data Science and Generative AI. 


---
# Conversational AI



## Resources for Conversational AI
1. [MTS Dialog](https://github.com/abachaa/MTS-Dialog)
---
# Common Metrics for ASR Evaluation
1. **Word Error Rate (WER)**
  * WER is a widely used metric that calculates the percentage of errors in the transcript compared to a reference transcript.
  * It counts substitutions, insertions, and deletions at the word level.
  * **Resources for calculating WER**
    1. [asr_evaluation](https://pypi.org/project/asr_evaluation/)
    2. [jiwer](https://jitsi.github.io/jiwer/usage/)

2. **Character Error Rate (CER)**
  * CER is similar to WER but calculates errors at the **character level**, providing a more granular assessment of the model's accuracy.


3. **Part-of-speech Error Rate (POSER)**
   * There are 2 version of this:
     * **dPOSER** -- Instead of computing WER on input words, we extract (preferably all) the parts-of-speech of the input sentences. The WER is then computed over the sequence of labels.
     * **uPOSER** -- The cited paper proposes a variant (uPOSER) with broad POS categories, in case that the used POS model has very specific categories. This can simply be implemented by using a synonym dictionary that groups up equivalent labels easily.
    

4. **Lemma Error Rate**
   * Instead of computing the WER over words, we compute the WER over lemmatized words.
  

5. **Embedding Error Rate (EmbER)**
  * Typical WER calculation, except that we weight the penalty of each word substitution if the words are deemed similar enough. This allows you to reduce the impact of e.g. minor spelling errors that do not alter the meaning much.


6. **Levenshtein Distance**
  * This metric measures the minimum number of edits (insertions, deletions, and substitutions) needed to change one string into another, often used in **comparing transcripts.**

7. **Jaro-Winkler Distance**
   * The Jaro–Winkler distance is a string metric measuring an edit distance between two sequences.
   * **The lower the Jaro–Winkler distance for two strings is, the more similar the strings are.**
   * The score is normalized such that 0 means an exact match and 1 means there is no similarity.
   * The Jaro–Winkler similarity is the inversion, (1 − Jaro–Winkler distance).
  
     * **Jaro-Winkler vs. Levenshtein**
       * Jaro-Winkler takes into account only matching characters and any required transpositions (swapping of characters). Also it gives more priority to prefix similarity.
       * Levenshtein counts the number of edits to convert one string to another.

8. **H EVAL**
  * This hybrid metric considers **both semantic correctness and error rate**, performing well in scenarios where WER and semantic distance (SD) are not effective. 

9. **SeMaScore**
  * This metric is designed to better correlate with human evaluation of ASR systems by considering **semantic similarity and accuracy.** 

10. **Semantic Distance**
  * This metric quantifies the semantic distance between the predicted and reference texts, providing insights into the semantic correctness of the ASR output. 

11. **BERTScore**
  * This metric uses BERT (Bidirectional Encoder Representations from Transformers) to assess the semantic similarity between the predicted and reference texts. 

12. **Perplexity**
    * Measure the perplexity of the speech recognition output using a language model. Lower perplexity indicates that the output is more coherent and likely.
    * Perplexity seeks to quantify the “uncertainty” a model experiences when when predicting the next token in a sequence. High uncertainty occurs when the model is unsure about the next word or token in a sequence.
    * **Resources for Perplexity**
      * [Perplexity - CometML](https://www.comet.com/site/blog/perplexity-for-llm-evaluation/)

9. **Proper Noun Evaluation**
10. **Normalization**
    * Common Normalizations:
        * **Lowercasing:** Converting all words to lowercase.
        * **Removing Punctuation:** Removing punctuation marks to focus on the core words.
        * **Expanding Abbreviations:** Replacing abbreviations with their full forms (e.g., "Dr." to "doctor").
        * **Remove stop words**
        * **Lemmatization**: A method that uses dictionaries and morphological analysis to find the base form (lemma) of a word. 
        * **Stemming**:  A heuristic process that removes common word endings (suffixes) to produce a "stem". 
        * **Removing Filler Words**: Removing words like "um" or "uh" that don't add to the content.


11. **Textual Entailment**
    * MedNLI is the most commonly used.
   
12. **Hallucination Detection**
    * Detailed in this paper [Analyzing LLM Behavior in Dialogue Summarization: Uneviling Circumstanial Hallucination Trends](https://aclanthology.org/2024.acl-long.677/)
    * Research from [ABRIDGE AI](https://www.abridge.com/ai/publications)
    * Techniques Mentioned
      1. Span Detection
      2. Binary Classification
     

13. **BioLORD**
    * BioLORD introduces **semantic context to enrich biomedical representations.**
    * The authors pair concepts with texts, which can be either their definitions or descriptions.
    * The definitions are sourced directly from the **UMLS ontology, but these are available for only ~5% of the concepts.**
    * To extend the semantic coverage for the other concepts, the authors generate simple descriptions based on the concept-to-concept relations present in the UMLS knowledge graph.
    * Example: `Template: “[more-generic-concept] which [has-relationship-with] [related-concept]”​`
    * The concepts and their respective contexts are then used as positive pairs in a contrastive learning setting, where positive pairs are attracted in the embedding space, and negative ones are repelled.
    * A detail: one concept may have different names. Thus, there may be more than one pair referring to the same concept. Ideally, these will all end up close in the trained embedding space.

    * Main contributions
      1. The expansion of our training corpus by supplementing its existing knowledge with new LLMs-generated definitions for 400 000 concepts, fusing knowledge graph and LLM insights inside the LORD pretraining (e.g. UMLS, SNOMED-CT)
      2. The introduction of a novel self-distillation technique to speed up biomedical knowledge acquisition while preserving the language understanding capabilities of the BioLORD-type models.
      3. The delivery of a state-of-the-art multilingual model for the biomedical domain, using a proven cross-lingual distillation technique.
   * Use Cases
      1. **Clinical Semantic Textual Similarity**
      2. **Biomedical Concept Representation**
      3. **Biomedical Named Entity Linking**
    
 * BioLORD Resources
   1. (Remy et al. 2022 - BioLORD: Learning Ontological Representations from Definitions (for Biomedical Concepts and their Textual Descriptions)[https://arxiv.org/abs/2210.11892]
   2. (Remy et al. 2023 - BioLORD-2023: Semantic Textual Representations Fusing LLM and Clinical Knowledge Graph Insights)[https://arxiv.org/abs/2311.16075]
(BioLORD: Contrastive learning for biomedical concepts)[https://hossboll.medium.com/biolord-contrastive-learning-for-biomedical-concepts-32c453f7a830]
   3. Hugging Face - BioLORD Dataset: https://huggingface.co/datasets/FremyCompany/BioLORD-Dataset
   4. Hugging Face - BioLORD Sentence Transformers model: https://huggingface.co/FremyCompany/BioLORD-2023-S




14. **Relation Extraction**
    * Relik is a recent open source model that is pre-trained for relation extraction and entity linking.
    * [Relik github](https://github.com/SapienzaNLP/relik?tab=readme-ov-file)
   

15. **Relational Triples Extraction**
    * Another open source model for triples/relation extraction is REBEL.
    * [REBEL github](https://github.com/Babelscape/rebel)

## Resources for Evaluation Metrics for ASR
1. [HuggingFace Audio Course](https://huggingface.co/learn/audio-course/en/chapter5/evaluation)
2. [Carnegie Mellon University 2025 Paper - On the Robust Approximation of ASR Metrics](https://arxiv.org/html/2502.12408v1)
3. [Jaro-Winkler vs. Levenshtein Distances](https://srinivas-kulkarni.medium.com/jaro-winkler-vs-levenshtein-distance-2eab21832fd6#:~:text=Jaro%2DWinkler%20takes%20into%20account,convert%20one%20string%20to%20another.)
4. [Jaro-Winkler Python library](https://github.com/rapidfuzz/JaroWinkler)
5. [Is Word Error Rate Useful?](https://www.assemblyai.com/blog/word-error-rate)
6. [How to evaluate Speech Recognition models](https://www.assemblyai.com/blog/how-to-evaluate-speech-recognition-models#:~:text=The%20most%20common%20evaluation%20metric,created%20by%20a%20human%20transcriber.)
7. [Speech Brain - Metrics for Speech Recognition Evaluation](https://speechbrain.readthedocs.io/en/v1.0.2/tutorials/tasks/asr-metrics.html#)
