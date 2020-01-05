import numpy as np 
import tensorflow_hub as hub
import tensorflow as tf
from bert.tokenization import FullTokenizer
from tensorflow.keras.models import Model

class Bert_Encoding:
    def __init__(self,BERT_PATH):
        self.tokenizer = FullTokenizer(BERT_PATH + '/assets/vocab.txt', True)
        self.MAX_SEQUENCE_LENGTH = 512
        input_word_ids = tf.keras.layers.Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype=tf.int32,name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype=tf.int32,name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype=tf.int32,name="segment_ids")
        bert_layer = hub.KerasLayer(BERT_PATH,trainable=True)
        pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
        
        self.model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])

    def get_masks(self,tokens, max_seq_length):
        """Mask for padding"""
        tokens = tokens[:max_seq_length]
        if len(tokens)>max_seq_length:
            raise IndexError("Token length more than max seq length!")
        return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

    def get_segments(self,tokens, max_seq_length):
        tokens = tokens[:max_seq_length]
        """Segments: 0 for the first sequence, 1 for the second"""
        if len(tokens)>max_seq_length:
            raise IndexError("Token length more than max seq length!")
        segments = []
        first_sep = True
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                if first_sep:
                    first_sep = False 
                else:
                    current_segment_id = 1
        return segments + [0] * (max_seq_length - len(tokens))
    
    def get_ids(self,tokens, tokenizer, max_seq_length):
        tokens = tokens[:max_seq_length]
        """Token ids from Tokenizer vocab"""
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
        return input_ids
    
    def get_bert_vector_whole(self,title,question,answer):
        stokens_title = self.tokenizer.tokenize(title)
        stokens_question = self.tokenizer.tokenize(question)
        stokens_answer = self.tokenizer.tokenize(answer)

        stokens = ["[CLS]"] + stokens_title + ["[SEP]"] + stokens_question + ["[SEP]"] + stokens_answer + ["[SEP]"]
        input_ids = self.get_ids(stokens, self.tokenizer, self.MAX_SEQUENCE_LENGTH)
        input_masks = self.get_masks(stokens, self.MAX_SEQUENCE_LENGTH)
        input_segments = self.get_segments(stokens, self.MAX_SEQUENCE_LENGTH)
        pool_embs, _ = self.model.predict([[input_ids],[input_masks],[input_segments]])
        return pool_embs
    
    def get_bert_vector_signal(self,text):
        stokens = self.tokenizer.tokenize(text)
        stokens = ["[CLS]"] + stokens + ["[SEP]"]
        input_ids = self.get_ids(stokens, self.tokenizer, self.MAX_SEQUENCE_LENGTH)
        input_masks = self.get_masks(stokens, self.MAX_SEQUENCE_LENGTH)
        input_segments = self.get_segments(stokens, self.MAX_SEQUENCE_LENGTH)
        pool_embs, _ = self.model.predict([[input_ids],[input_masks],[input_segments]])
        return pool_embs

'''
c = Bert_Encoding('/Users/gengjia/Desktop/google_kaggle/uncased_L-12_H-768_A-12')
a = c.get_bert_vector("hello","my name","cc")
print(a.shape)            
'''
    
    