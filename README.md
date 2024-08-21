# Semantic-Enhanced Differentiable Search Index Inspired by Learning Strategies

## Abstract

Recently, a new paradigm called Differentiable Search Index (DSI) has been proposed for document retrieval, wherein a sequenceto-sequence model is learned to directly map queries to relevant document identifiers. The key idea behind DSI is to fully parameterize traditional “index-retrieve” pipelines within a single neural model, by encoding all documents in the corpus into the model parameters. In essence, DSI needs to resolve two major questions: (1) how to assign an identifier to each document, and (2) how to learn the associations between a document and its identifier. In this work, we propose a Semantic-Enhanced DSI model (SE-DSI) motivated by Learning Strategies in the area of Cognitive Psychology. Our approach advances original DSI in two ways: (1) For the document identifier, we take inspiration from Elaboration Strategies in human learning. Specifically, we assign each document an Elaborative Description based on the query generation technique, which is more meaningful than a string of integers in the original DSI; and (2) For the associations between a document and its identifier, we take inspiration from Rehearsal Strategies in human learning. Specifically, we select fine-grained semantic features from a document as Rehearsal Contents to improve document memorization.

## Approach overview
DSI shares a similar way to human recall or retrieval the information that was previously encoded and remembered in the brain. Therefore, we introduce a novel Semantic-Enhanced DSI model (SE-DSI) to advance original DSI, inspired by problem-solving strategies labeled by some psychologists, i.e., Learning Strategies. Basically, the SE-DSI first constructs Elaborative Description (ED) from documents as docids to represent them with explicit semantics. Then, multiple coarse-fined contents from each document at different granularity are selected as Rehearsal Contents (RCs). In this way, we learn to build associations between original documents augmented with RCs and their corresponding EDs.

![An overview of our SE-DSI model. (a) We employ a query generation module to obtain ED from a document as its docid. (b) In the indexing phase, we propose to pair the original document and Rehearsal Contents (i.e., passage-level and sentence-level information) with the corresponding docid, respectively. In the retrieval phase, the docids are generated from the query, and a rank list of potentially-relevant documents is returned via beam search.](resources/overview.png)


## Resources

[Paper](resources/KDD23-Semantic-Enhanced_Differentiable_Search_Index_Inspired_by_Learning_Strategies.pdf)

[Slides](resources/20min.pptx)
