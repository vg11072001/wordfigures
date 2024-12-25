
# WordFigures


# WordFigures: From Basics to Advanced NLP Techniques

This repository demonstrates a structured journey through Natural Language Processing (NLP) and Large Language Models (LLMs), starting with fundamental text preprocessing techniques and advancing to modern deep learning models. Each file represents a specific aspect of the project, focusing on a variety of methods, concepts, and applications.

---

## Project Structure and File Details

### Part 1: Text Processing and Word Representations

#### **[wordFigures_part_1a_TextProcessing.ipynb](#)**
- **Description**: Demonstrates basic NLP techniques to preprocess and clean text data.
- **Techniques Covered**:
  1. Tokenization
  2. Stemming
  3. Lemmatization
  4. Removing Stopwords
  5. Bag of Words
  6. TF-IDF
- **Purpose**: Lays the foundation for text data preprocessing, transforming raw text into structured input for further analysis.
- **Reference Paper**: [Attention-All-You-Need.pdf](#)

---

#### **[wordFigures_part_1b_Word_embedding.ipynb](#)**
- **Description**: Explores word embedding techniques to convert text into numerical vectors, preserving semantic relationships.
- **Topics Covered**:
  - Overview of word embedding methods
  - Implementation of embeddings using pre-trained models (e.g., GloVe, FastText).
- **Purpose**: Introduces the concept of distributed word representations for improving NLP tasks.
- **Reference Paper**: [neural-probabilistic-lang-model-bengi003a.pdf](#)

---

#### **[wordFigures_part_1c_word2vec.ipynb](#)**
- **Description**: Demonstrates the Word2Vec model for learning word embeddings.
- **Key Features**:
  - Skip-gram and CBOW architectures
  - Training a Word2Vec model on sample text
  - Visualizing word relationships using t-SNE
- **Purpose**: Explains how Word2Vec captures contextual relationships in text data.
- **Reference Paper**: [neural-probabilistic-lang-model-bengi003a.pdf](#)

---

### Part 2: NLP Applications

#### **[wordFigures_part_2a_Sentiment_Analysis.ipynb](#)**
- **Description**: Implements sentiment analysis using Transformer-based models.
- **Techniques Covered**:
  - `TFDistilBertForSequenceClassification`
  - Fine-tuning with `TFTrainer` and `TFTrainingArguments`
- **Purpose**: Demonstrates how to classify text data into sentiment categories using state-of-the-art Transformer architectures.
- **Reference Paper**: [Attention-All-You-Need.pdf](#)

---

#### **[wordFigures_part_2b_FakeNewsClassifierusingLSTM.ipynb](#)**
- **Description**: Builds a Fake News Classifier using an LSTM-based deep learning model.
- **Key Features**:
  - Text preprocessing for fake news detection
  - LSTM architecture for sequence modeling
- **Purpose**: Highlights the application of recurrent neural networks for sequence classification tasks.
- **Reference Paper**: [LSTM-generating-sequences-RNN.pdf](#)

---

### Part 3: Transformer Models and Fine-Tuning

#### **[wordFigures_part_3a_BERT.ipynb](#)**
- **Description**: Introduces the BERT model and explains its architecture for solving NLP tasks.
- **Topics Covered**:
  - Tokenization and input preparation for BERT
  - Use of pre-trained BERT for text classification
- **Purpose**: Familiarizes users with BERT and its capabilities in understanding context-rich text.
- **Reference Paper**: [Attention-All-You-Need.pdf](#)

---

#### **[wordFigures_part_3b_BERT_Fine_Tuning.ipynb](#)**
- **Description**: Demonstrates fine-tuning BERT for specific NLP applications.
- **Techniques Covered**:
  - Customizing BERT for downstream tasks
  - Training and evaluation of the fine-tuned model
- **Purpose**: Showcases the versatility of BERT for domain-specific applications through transfer learning.
- **Reference Paper**: [LLMs Cheatsheet.pdf](#)

---

## References
1. **[Attention Is All You Need](Attention-All-You-Need.pdf)**  
   Vaswani et al. (2017) - Introduces the Transformer architecture.

2. **[GRU: On the Properties of Neural Machine Translation Encoder-Decoder](GRU-onthe-propetties-Nueral-Machine-Translation-Encoder-Decoder.pdf)**  
   Cho et al. (2014) - Discusses GRU properties in sequence modeling.

3. **[LLMs Cheatsheet](LLMs-Cheatsheet.pdf)**  
   A quick reference for modern Large Language Models.

4. **[LSTM: Generating Sequences with Recurrent Neural Networks](LSTM-generating-sequences-RNN.pdf)**  
   Hochreiter and Schmidhuber (1997) - Pioneering work on LSTM networks.

5. **[A Neural Probabilistic Language Model](neural-probabilistic-lang-model-bengi003a.pdf)**  
   Bengio et al. (2003) - Foundational work on neural-based language modeling.

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/WordFigures.git



# WordFigures: From Basics to Advanced NLP Techniques

This repository demonstrates a structured journey through Natural Language Processing (NLP) and Large Language Models (LLMs), starting with fundamental text preprocessing techniques and advancing to modern deep learning models. The project is divided into various notebooks, each covering specific topics and techniques.

---

## Project Structure

### Part 1: Text Processing and Word Representations
1. **[wordFigures_part_1a_TextProcessing.ipynb](#)**  
   Covers basic NLP techniques:  
   - Tokenization  
   - Stemming  
   - Lemmatization  
   - Removing Stopwords  
   - Bag of Words  
   - TF-IDF  

   **Reference Paper**: [Attention-All-You-Need.pdf](#)

2. **[wordFigures_part_1b_Word_embedding.ipynb](#)**  
   Introduces word embeddings and explores their applications in NLP.  

   **Reference Paper**: [neural-probabilistic-lang-model-bengi003a.pdf](#)

3. **[wordFigures_part_1c_word2vec.ipynb](#)**  
   Demonstrates the Word2Vec model for generating word embeddings and contextualizing text data.  

   **Reference Paper**: [neural-probabilistic-lang-model-bengi003a.pdf](#)

---

### Part 2: NLP Applications
1. **[wordFigures_part_2a_Sentiment_Analysis.ipynb](#)**  
   Implements sentiment analysis using Transformer models:  
   - `TFDistilBertForSequenceClassification`  
   - `TFTrainer` and `TFTrainingArguments`  

   **Reference Paper**: [Attention-All-You-Need.pdf](#)

2. **[wordFigures_part_2b_FakeNewsClassifierusingLSTM.ipynb](#)**  
   Builds a Fake News Classifier using an LSTM-based approach.  

   **Reference Paper**: [LSTM-generating-sequences-RNN.pdf](#)

---

### Part 3: Transformer Models and Fine-Tuning
1. **[wordFigures_part_3a_BERT.ipynb](#)**  
   Introduces the BERT model and its architecture for NLP tasks.  

   **Reference Paper**: [Attention-All-You-Need.pdf](#)

2. **[wordFigures_part_3b_BERT_Fine_Tuning.ipynb](#)**  
   Demonstrates fine-tuning BERT for specific downstream NLP tasks.  

   **Reference Paper**: [LLMs Cheatsheet.pdf](#)

---

## References
1. **[Attention Is All You Need](Attention-All-You-Need.pdf)**  
   Vaswani et al. (2017) - Introduces the Transformer architecture.  

2. **[GRU: On the Properties of Neural Machine Translation Encoder-Decoder](GRU-onthe-propetties-Nueral-Machine-Translation-Encoder-Decoder.pdf)**  
   Cho et al. (2014) - Discusses GRU properties in sequence modeling.  

3. **[LLMs Cheatsheet](LLMs-Cheatsheet.pdf)**  
   A quick reference for modern Large Language Models.  

4. **[LSTM: Generating Sequences with Recurrent Neural Networks](LSTM-generating-sequences-RNN.pdf)**  
   Hochreiter and Schmidhuber (1997) - Pioneering work on LSTM networks.  

5. **[A Neural Probabilistic Language Model](neural-probabilistic-lang-model-bengi003a.pdf)**  
   Bengio et al. (2003) - Foundational work on neural-based language modeling.

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/WordFigures.git



makemore takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. Under the hood, it is an autoregressive character-level language model, with a wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT). For example, we can feed it a database of names, and makemore will generate cool baby name ideas that all sound name-like, but are not already existing names. Or if we feed it a database of company names then we can generate new ideas for a name of a company. Or we can just feed it valid scrabble words and generate english-like babble.

This is not meant to be too heavyweight library with a billion switches and knobs. It is one hackable file, and is mostly intended for educational purposes. [PyTorch](https://pytorch.org) is the only requirement.

Current implementation follows a few key papers:

- Bigram (one character predicts the next one with a lookup table of counts)
- MLP, following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- CNN, following [DeepMind WaveNet 2016](https://arxiv.org/abs/1609.03499) (in progress...)
- RNN, following [Mikolov et al. 2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- LSTM, following [Graves et al. 2014](https://arxiv.org/abs/1308.0850)
- GRU, following [Kyunghyun Cho et al. 2014](https://arxiv.org/abs/1409.1259)
- Transformer, following [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

### Usage

The included `names.txt` dataset, as an example, has the most common 32K names takes from [ssa.gov](https://www.ssa.gov/oact/babynames/) for the year 2018. It looks like:

```
emma
olivia
ava
isabella
sophia
charlotte
...
```

Let's point the script at it:

```bash
$ python makemore.py -i names.txt -o names
```

Training progress and logs and model will all be saved to the working directory `names`. The default model is a super tiny 200K param transformer; Many more training configurations are available - see the argparse and read the code. Training does not require any special hardware, it runs on my Macbook Air and will run on anything else, but if you have a GPU then training will fly faster. As training progresses the script will print some samples throughout. However, if you'd like to sample manually, you can use the `--sample-only` flag, e.g. in a separate terminal do:

```bash
$ python makemore.py -i names.txt -o names --sample-only
```

This will load the best model so far and print more samples on demand. Here are some unique baby names that get eventually generated from current default settings (test logprob of ~1.92, though much lower logprobs are achievable with some hyperparameter tuning):

```
dontell
khylum
camatena
aeriline
najlah
sherrith
ryel
irmi
taislee
mortaz
akarli
maxfelynn
biolett
zendy
laisa
halliliana
goralynn
brodynn
romima
chiyomin
loghlyn
melichae
mahmed
irot
helicha
besdy
ebokun
lucianno
```

Have fun!

### License

MIT




## [Transformer](https://jalammar.github.io/illustrated-transformer/) :huggingface.co/docs/transformers/
![image](https://user-images.githubusercontent.com/67424390/210357353-2b203cab-73a3-4df6-9410-80291dbfa9c2.png)
#### Step 1: Install Transformer.
#### Step 2: Call the pretrained model.
#### Step 3: Call the tokenizer of that particular pretrained model and encode the text in ex. seq2seq manner.
#### Step 4: Convert these encoding into Dataset objects. (Different objects of dataset for tensorflow - tensors and pytorch)
#### Step 5:  Translate and decode the elements in batch

## [BERT](https://jalammar.github.io/illustrated-bert/): 
![image](https://user-images.githubusercontent.com/67424390/210527955-cf5f1405-1585-4a11-9f51-c2097da438bf.png)


### First

* [2_Fine Tuning Pretrained Model On Custom Dataset Using 🤗 Transformer: Custom_Sentiment_Analysis ](https://github.com/krishnaik06/Huggingfacetransformer/blob/main/Custom_Sentiment_Analysis.ipynb)

### Second
* [NLP: Implementing BERT and Transformers from Scratch](https://www.youtube.com/watch?v=EPa98fyxZ-s)
  * [Bert](https://github.com/msaroufim/RLnotes/blob/master/bert.md)
  * [Transformer](https://github.com/msaroufim/RLnotes/blob/master/transformer.md)


### Third - Official Hugging Face site
* [🤗 Transformers Notebooks](https://huggingface.co/docs/transformers/main/en/notebooks)
* [Tutorials by Hugging face for fine tune, processing, etc](https://huggingface.co/docs/transformers/training)
* [Summary of the tasks](https://huggingface.co/docs/transformers/task_summary)


## LSTM 
* [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
 ![image](https://user-images.githubusercontent.com/67424390/209438337-f56a9ba0-5b6f-4074-98ad-9edc5f9e569b.png)
 ![image](https://user-images.githubusercontent.com/67424390/209438362-6f59cc31-3de0-42db-b934-44bab94121b2.png)

## Word Embedding Techniques 
* https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews


## Table


| Model                   | Use Cases                                                   | Documentation                                              | Python Code                                                  |

|-------------------------|-------------------------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------|

| Bag-of-Words (BoW)      | Text classification, sentiment analysis, basic feature extraction         | [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)         | `from sklearn.feature_extraction.text import CountVectorizer`\n`vectorizer = CountVectorizer()`\n`X = vectorizer.fit_transform(corpus)` |

| TF-IDF                  | Information retrieval, document similarity, text classification             | [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)             | `from sklearn.feature_extraction.text import TfidfVectorizer`\n`vectorizer = TfidfVectorizer()`\n`X = vectorizer.fit_transform(corpus)` |

| Word2Vec                | Word embeddings, word similarity, document clustering                     | [Documentation](https://radimrehurek.com/gensim/models/word2vec.html)                    | `from gensim.models import Word2Vec`\n`model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)` |

| GloVe                   | Word embeddings, document classification, sentiment analysis, word analogy   | [Documentation](https://nlp.stanford.edu/projects/glove/)   | - |

| FastText                | Text classification, language identification, entity recognition              | [Documentation](https://fasttext.cc/docs/en/unsupervised-tutorial.html)               | - |

| LSTM (Long Short-Term Memory)       | Sequence labeling, sentiment analysis, named entity recognition     | [Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)     | - |

| GRU (Gated Recurrent Unit)         | Sequence modeling, language modeling, speech recognition         | [Documentation](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)         | - |

| Transformer             | Machine translation, language modeling, question answering         | [Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)         | - |

| BERT (Bidirectional Encoder Representations from Transformers)       | Text classification, named entity recognition, question answering     | [Documentation](https://huggingface.co/transformers/model_doc/bert.html)     | - |

| GPT (Generative Pre-trained Transformer)          | Text generation, story completion, language translation      | [Documentation](https://huggingface.co/transformers/model_doc/gpt.html)      | - |

| RoBERTa                  | Text classification, named entity recognition, sentiment analysis      | [Documentation](https://huggingface.co/transformers/model_doc/roberta.html)      | - |

| XLNet                    | Sequence classification, question answering, natural language inference  | [Documentation](https://huggingface.co/transformers/model_doc/xlnet.html)  | - |

| ALBERT                   | Text classification, named entity recognition, sentiment analysis  | [Documentation](https://huggingface.co/transformers/model_doc/albert.html)  | - |

| T5 (Text-To-Text Transfer Transformer)         | Text summarization, question answering, language translation     | [Documentation](https://huggingface.co/transformers/model_doc/t5.html)     | - |

| SBERT (Sentence-BERT)    | Semantic search, sentence similarity, text clustering          | [Documentation](https://www.sbert.net/docs/)          | - |

| XLM (Cross-lingual Language Model)       | Cross-lingual document classification, machine translation        | [Documentation](https://huggingface.co/transformers/model_doc/xlm.html)        | - |

| ELMO (Embeddings from Language Models)       | Contextual word embeddings, sentiment analysis, text classification     | [Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ELMoEmbedding)     | - |

| Flair                   | Named entity recognition, part-of-speech tagging, text classification    | [Documentation](https://flair.ai/docs/)    | - |

| ULMFiT (Universal Language Model Fine-tuning)      | Transfer learning, text classification, sentiment analysis     | [Documentation](https://docs.fast.ai/text.html)     | - |

| BioBERT                 | Biomedical text mining, named entity recognition, relation extraction   | [Documentation](https://github.com/dmis-lab/biobert)   | - |

| ELECTRA                 | Text classification, sentiment analysis, natural language understanding   | [Documentation](https://github.com/google-research/electra)   | - |

| DistilBERT              | Text classification, named entity recognition, question answering    | [Documentation](https://huggingface.co/transformers/model_doc/distilbert.html)    | - |




## Reading Materials
5 landmark papers in NLP that significantly moved the needle

These are 5 of my favorite papers that brought a step function change in natural language understanding. Anyone looking to learn NLP should read these, or at least some articles about them:

A unified architecture for natural language processing: deep neural networks with multitask learning (ronan.collobert.com/pub… ) : This paper was the first to show how a single neural network architecture could learn task-specific word embeddings and apply them to multiple NLP tasks, such as part-of-speech tagging, chunking, and named entity recognition. Before this, word embeddings existed mostly as a concept but weren’t widely used across multiple tasks.

Efficient Estimation of Word Representations in Vector Space (ronan.collobert.com/pub…) : This paper introduced Word2Vec model, which efficiently learns word embeddings by utilizing continuous bag-of-words (CBOW) and skip-gram architectures to capture semantic relationships between words. Word2Vec was the first major model to be scalable and widely used in industrial applications. It sparked a wave of X2Vec models (e.g., Doc2Vec, Tweet2Vec, Graph2vec), where various types of data were transformed into vector representations, further expanding the application of embedding techniques.

Attention is all you need (ronan.collobert.com/pub…) : Who could forget this paper? :)  It introduced the Transformer model, which eliminated the need for recurrence by relying entirely on self-attention mechanisms to model relationships between tokens in a sequence. This dramatically improved performance and efficiency in NLP tasks and laid the foundation for LLM models.

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (ronan.collobert.com/pub…) : This introduced BERT, the first LLM model based on deep bidirectional Transformers, pre-trained on vast amounts of text and fine-tuned for various NLP tasks setting a new standard for transfer learning.

Language Models are Few-Shot Learners (ronan.collobert.com/pub…) : This paper introduced GPT-3, the largest language model (at the time) capable of performing tasks with minimal fine-tuning or task-specific data. It demonstrated the power of few-shot learning and essentially began the shift away from the need for task-specific trained models.

Of course, the list is much longer, and there are many other milestone papers in NLP. Which ones are your favorites? I'd love to hear your thoughts!
## Categories of NLP:

* **Text Preprocessing**: Cleaning, tokenization, stemming, lemmatization, stop word removal, spell checking, etc.
* **Text Representation**: Bag-of-words, TF-IDF, word embeddings (Word2Vec, GloVe), contextual embeddings (BERT, ELMO), etc.
* **Syntax and Grammar**: Parsing, part-of-speech tagging, syntactic analysis, dependency parsing, constituency parsing, etc.
* **Semantics**: Named entity recognition, semantic role labeling, word sense disambiguation, semantic similarity, etc.
* **Text Classification**: Sentiment analysis, topic modeling, document categorization, spam detection, intent recognition, etc.
* **Information Extraction**: Named entity extraction, relation extraction, event extraction, entity linking, etc.
* **Machine Translation**: Neural machine translation, statistical machine translation, alignment models, sequence-to-sequence models, etc.
* **Question Answering**: Document-based QA, knowledge-based QA, open-domain QA, reading comprehension, etc.
* **Text Generation**: Language modeling, text summarization, dialogue systems, chatbots, text completion, etc.
* **Text-to-Speech and Speech-to-Text**: Automatic speech recognition (ASR), text-to-speech synthesis (TTS), voice assistants, etc.
* **Text Mining and Analytics**: Topic extraction, sentiment analysis, trend detection, text clustering, opinion mining, etc.
* **NLP Evaluation Metrics**: Precision, recall, F1-score, accuracy, BLEU score, ROUGE score, perplexity, etc.

## NLP application involves several steps

* **Define the Problem**: Clearly define the objective of your NLP application. Determine the specific task you want to perform, such as sentiment analysis, text classification, or named entity recognition.

* **Data Collection:** Gather a dataset that is relevant to your problem. This dataset should be labeled or annotated with the target labels or annotations you want to predict.

* **Data Preprocessing**: Clean and preprocess the dataset to prepare it for model training. This step may include removing unnecessary characters or symbols, handling missing data, tokenizing text, removing stop words, and performing other text normalization techniques.

* **Data Exploration and Analysis:** Perform exploratory data analysis (EDA) to gain insights into the dataset. Visualize the data, analyze its statistical properties, and understand the distribution of different classes or labels.

* **Feature Engineering:** Extract relevant features from the text data to represent it in a format suitable for machine learning algorithms. This may involve techniques such as bag-of-words, TF-IDF, word embeddings, or contextual embeddings.

* **Model Selection:** Choose an appropriate machine learning or deep learning model for your task. Consider the nature of your problem, the size of your dataset, and the available computational resources. Common models used in NLP include logistic regression, support vector machines (SVM), recurrent neural networks (RNN), and transformer models.

* **Model Training:** Split your dataset into training and validation sets. Train your chosen model on the training set using appropriate algorithms and optimization techniques. Tune hyperparameters to improve model performance. Monitor the training process to avoid overfitting.

* **Model Evaluation:** Evaluate the trained model using appropriate evaluation metrics such as accuracy, precision, recall, F1-score, or area under the ROC curve. Assess its performance on the validation set to understand its generalization capabilities.

* **Model Fine-tuning:** If the performance of the model is not satisfactory, consider refining the model architecture, adjusting hyperparameters, or applying techniques such as regularization to improve performance.

* **Model Testing:** Once you are satisfied with the model's performance, evaluate it on a separate, unseen test dataset to assess its real-world performance. Ensure that the test dataset is representative of the data your model will encounter in production.

* **Deployment:** Deploy the trained model into a production environment. This may involve integrating the model into an application or creating an API to serve predictions. Ensure the necessary infrastructure and resources are in place to support the deployment.

* **Monitoring and Maintenance:** Continuously monitor the performance of the deployed model and collect feedback from users. Regularly retrain or update the model as new data becomes available or the requirements change.

## LLM 
1. NLP
2. LLM
3. Framework - LangChain, Llamaindex
4. Fine Tune LLM models with larger dataset

![Screenshot 2023-09-03 205232](https://github.com/user-attachments/assets/9a4519a7-a58a-43c5-bd7f-5d04c0ba5fd8)


![Screenshot 2023-05-30 024318](https://github.com/user-attachments/assets/8c31718f-959d-44a3-8c17-64312ebf74a8)


## Refrences 
* [Link](https://substack.com/@kartiksinghal/note/c-70239045)
* https://www.youtube.com/playlist?list=PLZoTAELRMXVOTsz2jZl2Oq3ntWPoKRKwv
* https://neptune.ai/blog/natural-language-processing-with-hugging-face-and-transformers
* https://neptune.ai/blog/hugging-face-pre-trained-models-find-the-best
* https://www.youtube.com/watch?v=xI0HHN5XKDo