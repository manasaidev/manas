import os
import random
import numpy as np
import spacy
import gensim
from gensim.corpora import Dictionary

from models.transformer_model import TransformerModel
from models.sentiment_analyzer import SentimentAnalyzer
from utils.text_dataset import TextDataset
from utils.data_loader import load_text_data
from utils.text_processing import preprocess_text

import config

class CustomCryptoResponseGenerator:
    def __init__(self):
        self.sentiment_analyzer = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.sentiment_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sentiment_analyzer.eval()

        self.nlp = spacy.load('en_core_web_sm')

        self.stop_words = set(stopwords.words('english'))

        self.transformer_model = None
        self.vocab = None
        self.idx2word = None

        self.feminine_phrases = []
        self.crypto_keywords = []
        self.behaviour_keywords = []
        self.sample_texts = []

    def load_external_data(self, data_dir):
        sample_texts_file = os.path.join(data_dir, 'sample_texts.txt')
        with open(sample_texts_file, 'r', encoding='utf-8') as f:
            self.sample_texts = [line.strip() for line in f if line.strip()]

        feminine_phrases_file = os.path.join(data_dir, 'feminine_phrases.txt')
        with open(feminine_phrases_file, 'r', encoding='utf-8') as f:
            self.feminine_phrases = [line.strip() for line in f if line.strip()]

        crypto_keywords_file = os.path.join(data_dir, 'crypto_keywords.txt')
        with open(crypto_keywords_file, 'r', encoding='utf-8') as f:
            self.crypto_keywords = [line.strip() for line in f if line.strip()]

        behaviour_keywords_file = os.path.join(data_dir, 'behaviour_keywords.txt')
        with open(behaviour_keywords_file, 'r', encoding='utf-8') as f:
            self.behaviour_keywords = [line.strip() for line in f if line.strip()]

    def prepare_data(self):
        all_tokens = []
        for text in self.sample_texts:
            tokens = word_tokenize(text.lower())
            all_tokens.extend(tokens)
        unique_tokens = set(all_tokens)
        unique_tokens.update(['<pad>', '<unk>', '<eos>'])
        self.vocab = {token: idx for idx, token in enumerate(unique_tokens)}
        self.idx2word = {idx: word for word, idx in self.vocab.items()}

        self.dataset = TextDataset(self.sample_texts, self.vocab)

    def train_transformer_model(self, epochs=10, batch_size=32, learning_rate=0.0001):
        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.transformer_model = TransformerModel(vocab_size=len(self.vocab))
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.get('<pad>'))
        optimizer = optim.Adam(self.transformer_model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.transformer_model.train()
            epoch_loss = 0
            for batch in data_loader:
                optimizer.zero_grad()
                src = batch.transpose(0, 1)
                src_mask = self.transformer_model.generate_square_subsequent_mask(src.size(0))
                output = self.transformer_model(src, src_mask)
                loss = criterion(output.view(-1, len(self.vocab)), src.reshape(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(data_loader)}')

    def generate_response(self, input_text, max_length=50):
        sentiment_score = self.sentiment_analysis(input_text)

        main_topics = self.topic_modeling(input_text)

        entities = self.named_entity_recognition(input_text)

        base_prompt = self.create_base_prompt(sentiment_score, main_topics)

        tokens = word_tokenize(base_prompt.lower())
        token_ids = [self.vocab.get(token, self.vocab.get('<unk>')) for token in tokens]
        src = torch.tensor(token_ids, dtype=torch.long).unsqueeze(1)

        self.transformer_model.eval()
        generated_ids = self.generate_text(src, max_length)

        response_tokens = [self.idx2word.get(idx, '<unk>') for idx in generated_ids]
        response = ' '.join(response_tokens)

        response = self.post_process_response(response)

        return response

    def generate_text(self, src, max_length):
        generated_ids = src.squeeze(1).tolist()
        for _ in range(max_length):
            src_input = torch.tensor([generated_ids], dtype=torch.long).transpose(0, 1)
            src_mask = self.transformer_model.generate_square_subsequent_mask(len(generated_ids))
            with torch.no_grad():
                output = self.transformer_model(src_input, src_mask)
            next_token_logits = output[-1, 0, :]
            next_token_id = torch.argmax(next_token_logits).item()
            generated_ids.append(next_token_id)
            if next_token_id == self.vocab.get('<eos>'):
                break
        return generated_ids

    def sentiment_analysis(self, text):
        inputs = self.sentiment_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.sentiment_analyzer(**inputs)
        logits = outputs.logits
        sentiment = torch.argmax(logits).item()
        score = torch.softmax(logits, dim=1).max().item()
        if sentiment == 1:
            return score
        else:
            return -score

    def topic_modeling(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        dictionary = Dictionary([tokens])
        corpus = [dictionary.doc2bow(tokens)]
        lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=1, random_state=42)
        topics = lda_model.print_topics()
        main_topics = []
        for topic in topics:
            topic_words = topic[1].split('+')
            for word_weight in topic_words:
                word = word_weight.split('*')[1].strip().strip('"')
                main_topics.append(word)
        return main_topics

    def named_entity_recognition(self, text):
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'GPE', 'NORP', 'PRODUCT']]
        return entities

    def create_base_prompt(self, sentiment_score, main_topics):
        feminine_phrase = np.random.choice(self.feminine_phrases)
        behaviour_keyword = np.random.choice(self.behaviour_keywords)

        if sentiment_score > 0:
            sentiment_context = 'I’m thrilled about'
        elif sentiment_score < 0:
            sentiment_context = 'I’m worried about'
        else:
            sentiment_context = 'I’m contemplating'

        if main_topics:
            topic = np.random.choice(main_topics)
        else:
            topic = np.random.choice(self.crypto_keywords)

        base_prompt = f"{feminine_phrase} {sentiment_context} {topic}. Being {behaviour_keyword}, here's my take:"
        return base_prompt

    def post_process_response(self, response):
        response = response.replace('<pad>', '').replace('<eos>', '').strip()
        response = ' '.join(response.split())
        if len(response) > 280:
            response = response[:277] + '...'
        response = response.capitalize()
        if not response.endswith('.'):
            response += '.'
        return response