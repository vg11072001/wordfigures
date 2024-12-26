
# WordFigures


This repository demonstrates a structured journey through Natural Language Processing (NLP) and Large Language Models (LLMs), starting with fundamental text preprocessing techniques and advancing to modern deep learning models. Each file represents a specific aspect of the project, focusing on a variety of methods, concepts, and applications.

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


#### **[wordFigures_part_1b_Word_embedding.ipynb](#)**
- **Description**: Explores word embedding techniques to convert text into numerical vectors, preserving semantic relationships.
- **Topics Covered**:
  - Overview of word embedding methods
  - Implementation of embeddings using pre-trained models (e.g., GloVe, FastText).
- **Purpose**: Introduces the concept of distributed word representations for improving NLP tasks.
- **Reference Paper**: [neural-probabilistic-lang-model-bengi003a.pdf](#)


#### **[wordFigures_part_1c_word2vec.ipynb](#)**
- **Description**: Demonstrates the Word2Vec model for learning word embeddings.
- **Key Features**:
  - Skip-gram and CBOW architectures
  - Training a Word2Vec model on sample text
  - Visualizing word relationships using t-SNE
- **Purpose**: Explains how Word2Vec captures contextual relationships in text data.
- **Reference Paper**: [neural-probabilistic-lang-model-bengi003a.pdf](#)


### Part 2: NLP Applications

#### **[wordFigures_part_2a_Sentiment_Analysis.ipynb](#)**
- **Description**: Implements sentiment analysis using Transformer-based models.
- **Techniques Covered**:
  - `TFDistilBertForSequenceClassification`
  - Fine-tuning with `TFTrainer` and `TFTrainingArguments`
- **Purpose**: Demonstrates how to classify text data into sentiment categories using state-of-the-art Transformer architectures.


#### **[wordFigures_part_2b_FakeNewsClassifierusingLSTM.ipynb](#)**
- **Description**: Builds a Fake News Classifier using an LSTM-based deep learning model.
- **Key Features**:
  - Text preprocessing for fake news detection
  - LSTM architecture for sequence modeling
- **Purpose**: Highlights the application of recurrent neural networks for sequence classification tasks.
- **Reference Paper**: 
   * [LSTM-generating-sequences-RNN.pdf](#)
   * [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
   - ![image](https://user-images.githubusercontent.com/67424390/209438337-f56a9ba0-5b6f-4074-98ad-9edc5f9e569b.png)
   - ![image](https://user-images.githubusercontent.com/67424390/209438362-6f59cc31-3de0-42db-b934-44bab94121b2.png)

### Part 3: Transformer Models and Fine-Tuning

#### **[wordFigures_part_3a_BERT.ipynb](#)**
- **Description**: Introduces the BERT model and explains its architecture for solving NLP tasks.
- **Topics Covered**:
  - Tokenization and input preparation for BERT
  - Use of pre-trained BERT for text classification
- **Purpose**: Familiarizes users with BERT and its capabilities in understanding context-rich text.
- **Reference Paper**: [BERT.pdf](#)


#### **[wordFigures_part_3b_BERT_Fine_Tuning.ipynb](#)**
- **Description**: Demonstrates fine-tuning BERT for specific NLP applications.
- **Techniques Covered**:
  - Customizing BERT for downstream tasks
  - Training and evaluation of the fine-tuned model
- **Purpose**: Showcases the versatility of BERT for domain-specific applications through transfer learning.
- **Reference Paper**: [BERT.pdf](#)



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



## Word Embedding Techniques 
* https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews



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


## Inspirations 
* https://substack.com/@kartiksinghal/note/c-70239045
* https://www.youtube.com/playlist?list=PLZoTAELRMXVOTsz2jZl2Oq3ntWPoKRKwv
* https://neptune.ai/blog/natural-language-processing-with-hugging-face-and-transformers
* https://neptune.ai/blog/hugging-face-pre-trained-models-find-the-best
* https://www.youtube.com/watch?v=xI0HHN5XKDo