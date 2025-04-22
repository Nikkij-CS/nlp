# NLP and Attention Mechanisms

# Question 1: NLP Preprocessing Pipeline

## Objective
To perform basic Natural Language Processing (NLP) steps on a sample sentence using tokenization, stopword removal, and stemming.

## Input Sentence
"**NLP techniques are used in virtual assistants like Alexa and Siri.**"

## Steps Performed

### 1. Tokenization
- The sentence is split into individual words and punctuation marks using a tokenizer.
- This helps break down the sentence into manageable pieces for further processing.

### 2. Stopword Removal
- Common English words like "the", "are", "in" are removed.
- These words generally don’t add much meaning and are often filtered out in NLP tasks.
- Non-alphabetical characters (e.g., punctuation) are also excluded.

### 3. Stemming
- Each remaining word is reduced to its base/root form.
- For example, "running" becomes "run", "technologies" becomes "technolog".
- This helps normalize similar words to a common representation.

## Output (Expected)
- **Original Tokens** – All words and punctuation in the sentence.
- **Tokens Without Stopwords** – Only the meaningful words remain.
- **Stemmed Words** – Each word reduced to its base form for consistency.

# Question 2: Named Entity Recognition with SpaCy

## Objective
To extract named entities from a sentence using the SpaCy library and display key information about each entity.

## Input Sentence
"**Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009.**"

## Steps Performed

### 1. Load the Language Model
- The pre-trained English model `en_core_web_sm` is loaded from spaCy.
- This model is capable of detecting entities like people, places, organizations, dates, etc.

### 2. Process the Text
- The sentence is passed through the NLP pipeline.
- spaCy tokenizes the sentence and analyzes it for linguistic features including entity recognition.

### 3. Extract Named Entities
- Each named entity identified by the model is extracted.
- For every entity, the following details are displayed:
  - **Text**: The exact word or phrase detected (e.g., "Barack Obama").
  - **Label**: The type of entity (e.g., PERSON, DATE, ORG).
  - **Start & End Character Positions**: Index range where the entity appears in the original sentence.

## Output (Expected)
- A list of named entities with:
  - Entity Text
  - Entity Label
  - Character Start and End Positions

# Question 3: Scaled Dot-Product Attention

## Objective
To implement the scaled dot-product attention mechanism — a core component of Transformer models — using NumPy.

## Steps Performed

### 1. Input Setup
- Three matrices are defined:
  - **Q (Query)**
  - **K (Key)**
  - **V (Value)**
- These matrices represent the input sequences to be used in attention calculation.

### 2. Compute Dot Product (Q · Kᵀ)
- The query matrix is multiplied with the transpose of the key matrix.
- This gives a raw attention score that indicates how much focus one element should have on another.

### 3. Scale the Scores
- The resulting scores are divided by the square root of the dimension of the key vector (√d).
- This step prevents overly large dot products that could destabilize softmax.

### 4. Apply Softmax
- A softmax function is applied row-wise to the scaled scores.
- This converts the raw scores into normalized attention weights (probabilities).

### 5. Weighted Sum with V
- The attention weights are then multiplied with the value matrix.
- This gives the final attention output — a weighted combination of values based on relevance.

## Output (Expected)
- **Attention Weights** – A matrix showing how much attention each token pays to others.
- **Final Output** – A matrix representing the attention-modified values.

# Question 4: Sentiment Analysis using HuggingFace Transformers

## Objective
To perform sentiment classification on a given sentence using a pre-trained model from HuggingFace's Transformers library.

## Input Sentence
"**Despite the high price, the performance of the new MacBook is outstanding.**"

## Steps Performed

### 1. Load Pre-trained Pipeline
- A sentiment analysis pipeline is loaded using HuggingFace’s `transformers` library.
- This pipeline is backed by models like `distilbert-base-uncased-finetuned-sst-2-english` by default.

### 2. Provide Text Input
- The sentence is passed into the pipeline for evaluation.

### 3. Analyze Sentiment
- The pipeline processes the input and returns:
  - **Sentiment Label**: POSITIVE or NEGATIVE.
  - **Confidence Score**: A float between 0 and 1 indicating the model's certainty.

### 4. Display Output
- The result is printed with the sentiment and the corresponding confidence score (rounded for clarity).

## Output (Expected)
- **Sentiment:** POSITIVE or NEGATIVE  
- **Confidence Score:** e.g., 0.9974






