import torch
from torch import nn
import json
from typing import List
from tqdm import tqdm
import string
import spacy
import re
# from punctuators.models import SBDModelONNX

class AugmentData(nn.Module):
    def __init__(self, augmentModel="sbd_multi_lang"):
        super(AugmentData, self).__init__()
        # self.augmentModel = SBDModelONNX.from_pretrained(augmentModel)
        self.nlp = spacy.load("en_core_web_sm")
        self.build()
    def build(self):
        config = self.read_json("config.json")
        self.replace_tokens = config['replace tokens']
        self.coordinating_conjunction = config['coordinating conjunction']
        self.punctuation_error = config['punctuation error']
        self.unclear_eos_words = config['unclear eos words']
        self.unclear_bos_words = config['unclear bos words']
        self.candidate_num = config['candidate num']
    def forward(self, input_dataset_path, output_dataset_path):
        sentence_pool, span_sentence_pool = self.augment(input_dataset_path)
        
    def augment(self, input_dataset_path):
        input_dataset = self.read_json(input_dataset_path)
        sentence_pool = []
        span_sentence_pool = []
        for i in tqdm(range(len(input_dataset))):
            pool = []
            span_pool = []
            for j in range(len(input_dataset[i]['conversation'])):
                augment_sentence_pool, augment_sentence_span_pool = self.augment_sentence(input_dataset[i]['conversation'][j]['text'])
                pool.append(augment_sentence_pool)
                span_pool.append(augment_sentence_span_pool)
            
            sentence_pool.append(pool)
            span_sentence_pool.append(span_pool)
        assert len(sentence_pool) == len(span_sentence_pool)
        
        return sentence_pool, span_sentence_pool
    def remove_short_sentence(self, pool, span_pool):
        # print(pool)
        # print(span_pool)
        sentence_pool = []
        span_sentence_pool = []
        for i in range(len(pool)):
            words = pool[i].split()
            if len(words) > 2:
                sentence_pool.append(pool[i])
                span_sentence_pool.append(span_pool[i])
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool

    def augment_sentence(self, text, test=False):
        doc = self.nlp(text)

        if test:
            lll = [(doc[i].pos_, doc[i].lemma_) for i in range(len(doc))]
            print(lll)
        sentence_pool = []
        span_sentence_pool = []

        text = text.replace('\u2019', "'")

        ss, se, subtext = self.span(text, text)
        # unique_list_of_lists, span_list_of_lists = self.split_sentence(text)
        # sentence_pool.extend(unique_list_of_lists)
        # span_sentence_pool.extend(span_list_of_lists)
        spacy_pool, spacy_span_pool = self.spacy(text)
        sentence_pool.extend(spacy_pool)
        span_sentence_pool.extend(spacy_span_pool)
        if test:
            print("spacy:"+str(sentence_pool))
        # print("sentence_pool: "+str(sentence_pool))
        # print("span_sentence_pool: "+str(span_sentence_pool))
        # print("len entence_pool: "+str(len(sentence_pool)))
        # print("len span_sentence_pool: "+str(len(span_sentence_pool)))
        # sentence_pool, span_sentence_pool = self.remove_duplicate(sentence_pool, span_sentence_pool)
        sentence_pool, span_sentence_pool = self.remove_short_sentence(sentence_pool, span_sentence_pool)
        if test:
            print("remove_short_sentence:"+str(sentence_pool))

        sentence_pool.append(subtext)
        span_sentence_pool.append((ss, se))
        sentence_pool, span_sentence_pool = self.list_remove_punctuation(sentence_pool, span_sentence_pool)
        if test:
            print("list_remove_punctuation:"+str(sentence_pool))
        sentence_pool, span_sentence_pool = self.remove_list_start_by_coordinating_conjunction(sentence_pool, span_sentence_pool)
        if test:
            print("remove_list_start_by_coordinating_conjunction:"+str(sentence_pool))
        sentence_pool, span_sentence_pool = self.remove_duplicate(sentence_pool, span_sentence_pool)
        if test:
            print("remove_duplicate:"+str(sentence_pool))
        sentence_pool, span_sentence_pool = self.remove_unclear_content(sentence_pool, span_sentence_pool)
        if test:
            print("remove_unclear_content:"+str(sentence_pool))
        sentence_pool, span_sentence_pool = self.remove_end_by_unclear_sentence(sentence_pool, span_sentence_pool)
        if test:
            print("remove_end_by_unclear_sentence:"+str(sentence_pool))
        sentence_pool, span_sentence_pool = self.remove_dup_words(sentence_pool, span_sentence_pool)
        if test:
            print("remove_dup_words:"+str(sentence_pool))
        sentence_pool, span_sentence_pool = self.remove_dup_sentences(sentence_pool, span_sentence_pool)
        if test:
            print("remove_dup_sentences:"+str(sentence_pool))
        sentence_pool, span_sentence_pool = self.remove_unclear_last_sentence(sentence_pool, span_sentence_pool)
        if test:
            print("remove_unclear_last_sentence:"+str(sentence_pool))
        sentence_pool, span_sentence_pool = self.remove_unclear_first_sentence(sentence_pool, span_sentence_pool)
        if test:
            print("remove_unclear_first_sentence:"+str(sentence_pool))
        sentence_pool, span_sentence_pool = self.remove_interjections_in_the_end(sentence_pool, span_sentence_pool)
        if test:
            print("remove_interjections_in_the_end:"+str(sentence_pool))
        sentence_pool, span_sentence_pool = self.remove_more_adj(sentence_pool, span_sentence_pool)
        if test:
            print("remove_more_adj:"+str(sentence_pool))
        sentence_pool, span_sentence_pool = self.remove_more_intj(sentence_pool, span_sentence_pool)
        if test:
            print("remove_more_intj:"+str(sentence_pool))
        # sentence_pool, span_sentence_pool = self.remove_transition_sentence(sentence_pool, span_sentence_pool)

        # sentence_pool, span_sentence_pool = self.list_span(sentence_pool, text)

        sentence_pool, span_sentence_pool = self.fit_candidate_num(sentence_pool, span_sentence_pool)

        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool
    def fit_candidate_num(self, pool, span_pool):
        m = min(len(pool), self.candidate_num)

        return pool[len(pool) - m:], span_pool[len(pool) - m:]
    def remove_list_start_by_coordinating_conjunction(self, pool, span_pool):
        sentence_pool = []
        span_sentence_pool = []
        for i in range(len(pool)):
            span = self.remove_start_by_coordinating_conjunction(pool[i], span_pool[i])
            if span == None:
                continue
            sentence, span_sentence = span
            sentence_pool.append(sentence)
            span_sentence_pool.append(span_sentence)
        # if len(pool) != len(sentence_pool):
        #     print(str(len(pool))+"    "+str(len(sentence_pool)))
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool

    def remove_start_by_coordinating_conjunction(self, sentence, span_sentence):
        words = sentence.split()
        if words[0] in self.coordinating_conjunction:
            return None
        return sentence, span_sentence
    def split_sentence(self, text): 
        input_texts: List[str] = [text]
        # print("input_texts =: "+str(input_texts))
        results: List[List[str]] = self.augmentModel.infer(input_texts)
        results = results[0]
        results=self.replace_token(results)
        
        # results = [sentence for sentence in results if self.replace_token not in sentence]
        temp = []
        temp.extend(results)
        temp.extend(input_texts)
        # print("results =: "+str(results))
        # print("input_texts =: "+str(input_texts))
        unique_list_of_lists = [self.remove_punctuation(x) for x in temp]
        # print("unique_list_of_lists =: "+str(unique_list_of_lists))
        unique_list_of_lists = [item for item in unique_list_of_lists if len(item)>1]
        # print("unique_list_of_lists ===: "+str(unique_list_of_lists))
        # print("span_list_of_lists === : "+str(span_list_of_lists))
        unique_set = set()
        unique_list_of_lists = [x for x in unique_list_of_lists if tuple(x) not in unique_set and not unique_set.add(tuple(x))]

        span_list_of_lists = []
        for item in unique_list_of_lists:
            ss, se, _ = self.span(input_texts[0], item)
            span_list_of_lists.append((ss, se))

        return unique_list_of_lists, span_list_of_lists
    def replace_token(self, text_list):
        new_text_list = []
        # print("text list replace: "+str(text_list))
        exist_list = [1] * len(text_list)
        for i in range(len(text_list)):
            for token in self.replace_tokens:
                if token in text_list[i]:
                    exist_list[i] = 0
                    break
        
        new_text_list = [text_list[i] for i in range(len(text_list)) if exist_list[i]]    
        # print("new text list replace: "+str(new_text_list))
        return new_text_list
        
    def split_coordinating_conjunction(self, doc, text):
        # print("text: "+str(text))
        sentence_pool = []
        span_sentence_pool = []

        ss = 0
        words = []
        text_list = text.split()
        # print("text_list: "+str(text_list))
        # for i in range(len(text_list)):
        #     if doc[i].pos_ in ["CCONJ", ]:
        #         se = i+1
        #         sentence_pool.append(" ".join(words))
        #         span_sentence_pool.append((ss, se))
        #         words = []
        #         ss = i+1
        #     else:
        #         words.append(text_list[i])
        for i in range(len(text_list)):
            # print("text_list: "+str(text_list[i]))
            if text_list[i] in self.coordinating_conjunction:
                # print(text_list[i])
                se = i
                sentence_pool.append(" ".join(words))
                span_sentence_pool.append((ss, se))
                words = []
                ss = i+1
            else:
                words.append(text_list[i])
        # print("words: "+str(words))
        span = " ".join(words)
        # print("span: "+str(span))
        new_span = self.remove_punctuation(span)
        # print("new_span: "+str(new_span))
        ss, se, new_span = self.span(text, new_span)
        sentence_pool.append(new_span)
        span_sentence_pool.append((ss, se))
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool
    def split_comma(self, text):
        sentence_pool = []
        span_sentence_pool = []

        ss = 0
        words = []
        text_list = text.split()
        for i in range(len(text_list)):
            if text_list[i] == ',':
                se = i+1
                span = " ".join(words)
                # print("span: "+str(span))
                new_span = self.remove_punctuation(span)
                # print("new_span: "+str(new_span))
                ss, se, new_span = self.span(text, new_span)
                sentence_pool.append(new_span)
                span_sentence_pool.append((ss, se))
                words = []
                ss = i+1
            else:
                words.append(text_list[i])
        
        span = " ".join(words)
        new_span = self.remove_punctuation(span)
        ss, se, new_span = self.span(text, new_span)
        sentence_pool.append(span)
        span_sentence_pool.append((ss, se))
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool
    def concat_comma(self, sentence_pool, span_sentence_pool):
        # print(sentence_pool)
        # print("=============================concat_sentennce=======================")
        new_sentence_pool = []
        new_span_sentence_pool = []
        for i in range(len(sentence_pool)):
            # print("============================ sentence pool ========================")
            for j in range(i, len(sentence_pool)):
                # print("============================ span sentence pool ========================")
                ss = span_sentence_pool[i][0]
                se = span_sentence_pool[j][1]
                span = ' , '.join(sentence_pool[i:j+1])
                new_span = self.remove_punctuation(span)
                ss, se, new_span = self.span(span, new_span, ss)
                # print("span:"+span)
                # print("ss:"+str(ss))
                # print("se:"+str(se))
                # print("new_span:"+str(new_span))
                new_sentence_pool.append(span)
                new_span_sentence_pool.append((ss, se))
        # print("=============================concat_sentennce=======================")
        assert len(sentence_pool) == len(span_sentence_pool)
        return new_sentence_pool, new_span_sentence_pool
        
    def read_json(self, path):
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        return data
    def write_json(self, path, data):
        with open('output_json_file.json', 'w') as file:
            # Write the Python data structure as JSON to the file
            json.dump(data, file, indent=2)
    def remove_punctuation(self, text):
        # Create a string of all punctuation characters
        punctuation_chars = string.punctuation

        # Remove punctuation at the beginning and end of the string
        cleaned_text = text.strip(punctuation_chars).strip().strip(punctuation_chars).strip().strip(punctuation_chars).strip()
        return cleaned_text
    def list_remove_punctuation(self, pool, span_pool):
        sentence_pool = []
        span_sentence_pool = []
        for i in range(len(pool)):   
            # print("pool[i]: "+str(pool[i]))
            new_span = self.remove_punctuation(pool[i])
            # print("new_span: "+str(new_span))
            if len(new_span) == 0:
                continue
            # span = 
            # print("new_span: "+str(new_span))

            ss, se, new_new_span = self.span(pool[i], new_span, span_pool[i][0])
            sentence_pool.append(pool[i])
            span_sentence_pool.append((ss, se))
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool
    def span(self, text, sub_text, s_ind=0):
        new_text = text.split()
        if text == sub_text:
            ss = s_ind
            se = s_ind+len(new_text)
            return ss, se, text
        new_sub_text = sub_text
        new_sub_text = new_sub_text.split()
        check = True
        for i in range(len(new_text)):
            temp = i
            for j in range(len(new_sub_text)):
                if new_sub_text[j] != new_text[i + j]:
                    if j == len(new_sub_text)-1:
                        # print("new_sub_text[j]: "+str(new_sub_text[j] ))
                        # print("new_text[i + j]: "+str(new_text[i + j] ))
                        for punctuation in self.punctuation_error:

                            if new_sub_text[j]+punctuation  == new_text[i + j]:
                                # print(new_sub_text)
                                new_sub_text[j] = new_sub_text[j]+punctuation 
                                ss = i
                                se = i+j
                                # print(new_sub_text)
                                return s_ind+ss, s_ind+se+1, ' '.join(new_sub_text)
                    check = False
                    break
                else:
                    temp = i + j
                    check = True
            if check == True:
                ss = i
                se = temp
                return s_ind+ss, s_ind+se+1, sub_text
        # return 
    def list_span(self, pool, text):
        sentence_pool = []
        span_sentence_pool = []
        for i in range(len(pool)):
            # print(text + "   " + pool[i])
            span = self.span(text, pool[i])
            if span == None:
                continue
            ss, se, subtext = span
            sentence_pool.append(pool[i])
            span_sentence_pool.append((ss, se))
        
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool

    def spacy(self, text):
        sentence_pool = []
        span_sentence_pool = []
        doc = self.nlp(text)

        comma_pool, comma_span_pool = self.split_comma(text)
        comma_pool, comma_span_pool = self.concat_comma(comma_pool, comma_span_pool)
        comma_pool, comma_span_pool = self.remove_duplicate(comma_pool, comma_span_pool)
        # pron_chunks_pool, pron_chunks_span_pool = self.pron_chunks(text)
        # pron_chunks_pool, pron_chunks_span_pool = self.concat_sentennce(pron_chunks_pool, pron_chunks_span_pool)
        # pron_chunks_pool, pron_chunks_span_pool = self.remove_duplicate(pron_chunks_pool, pron_chunks_span_pool)
        # print("comma")
        # print("comma_pool : "+str(len(comma_pool)))
        # print("comma_span_pool : "+str(len(comma_span_pool)))
        coordinating_conjunction_pool, coordinating_conjunction_span_pool = self.split_coordinating_conjunction(doc, text)
        # print("split_coordinating_conjunction")
        # print("coordinating_conjunction_pool : "+str(coordinating_conjunction_pool))
        # print("coordinating_conjunction_span_pool : "+str(coordinating_conjunction_span_pool))
        # print("comma_pool : "+str(len(coordinating_conjunction_pool)))
        # print("comma_span_pool : "+str(len(coordinating_conjunction_span_pool)))
        auxiliary_coordinating_conjunction_pool, auxiliary_coordinating_conjunction_span_pool = self.list_auxiliary_spacy(coordinating_conjunction_pool, coordinating_conjunction_span_pool)
        # print("split_auxiliary_coordinating_conjunction")
        # print("auxiliary_coordinating_conjunction_pool : "+str(auxiliary_coordinating_conjunction_pool))
        # print("auxiliary_coordinating_conjunction_span_pool : "+str(auxiliary_coordinating_conjunction_span_pool))
        # print("comma_pool : "+str(len(auxiliary_coordinating_conjunction_pool)))
        # print("comma_span_pool : "+str(len(auxiliary_coordinating_conjunction_span_pool)))
        noun_chunks_pool, noun_chunks_span_pool=self.noun_chunks(text)
        # noun_chunks_pool, noun_chunks_span_pool=self.noun_chunks(text)
        # print("noun_chunks")
        # print("noun_chunks_pool : "+str(noun_chunks_pool))
        # print("noun_chunks_span_pool : "+str(noun_chunks_span_pool))
        # print("comma_pool : "+str(len(noun_chunks_pool)))
        # print("comma_span_pool : "+str(len(noun_chunks_span_pool)))
        sentence_spacy_pool, sentence_spacy_span_pool = self.sentence_spacy(doc)
        # print("sentence_spacy")
        # print("sentence_spacy_pool : "+str(sentence_spacy_pool))
        # print("sentence_spacy_span_pool : "+str(sentence_spacy_span_pool))
        # print("comma_pool : "+str(len(sentence_spacy_pool)))
        # print("comma_span_pool : "+str(len(sentence_spacy_span_pool)))
 
        sentence_pool.extend(comma_pool)
        span_sentence_pool.extend(comma_span_pool)
        # sentence_pool.extend(pron_chunks_pool)
        # span_sentence_pool.extend(pron_chunks_span_pool)
        sentence_pool.extend(coordinating_conjunction_pool)
        span_sentence_pool.extend(coordinating_conjunction_span_pool)
        sentence_pool.extend(auxiliary_coordinating_conjunction_pool)
        span_sentence_pool.extend(auxiliary_coordinating_conjunction_span_pool)
        sentence_pool.extend(noun_chunks_pool)
        span_sentence_pool.extend(noun_chunks_span_pool)
        sentence_pool.extend(sentence_spacy_pool)
        span_sentence_pool.extend(sentence_spacy_span_pool)
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool
    def list_auxiliary_spacy(self, text_list, span_pool):
        sentence_pool = []
        span_sentence_pool = []
        for i in range(len(text_list)):
            doc = self.nlp(text_list[i])
            sentence_spacy_pool, sentence_spacy_span_pool = self.sentence_spacy(doc, span_pool[i][0])
            # print("list_auxiliary_spacy")
            # print("sentence_spacy_pool : "+str(sentence_spacy_pool))
            # print("sentence_spacy_span_pool : "+str(sentence_spacy_span_pool))
        sentence_pool.extend(sentence_spacy_pool)
        span_sentence_pool.extend(sentence_spacy_span_pool)
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool


    def auxiliary_spacy(self, text):
        sentence_pool = []
        span_sentence_pool = []
        doc = self.nlp(text)
        sentence_spacy_pool, sentence_spacy_span_pool = self.sentence_spacy(doc)
        sentence_pool.extend(sentence_spacy_pool)
        span_sentence_pool.extend(sentence_spacy_span_pool)
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool
    
    
    # def list_pron_chunks(self, text_list, text, ids):

    def pron_chunks(self, text):
        sentence_pool = []
        span_sentence_pool = []
        doc = self.nlp(text)
        ss = 0
        words= []
        for i in range(len(doc)):
            # Check if the token is a pronoun
            if doc[i].pos_ == 'PRON':
                se = i+1
                span = ' '.join(words)
                # print("span: "+span)
                span = self.remove_punctuation(span)
                new_span = self.span(text, span, ss)
                if span == None:
                    continue
                ss, se, new_span = new_span
                sentence_pool.append(span)
                span_sentence_pool.append((ss, se))
                words = [doc[i].text]
            else:
                words.append(doc[i].text)
                

        span = " ".join(words)
        new_span = self.remove_punctuation(span)
        new_span = self.span(text, new_span)
        if span == None:
            assert len(sentence_pool) == len(span_sentence_pool)
            return sentence_pool, span_sentence_pool
        ss, se, new_span = new_span
        sentence_pool.append(span)
        span_sentence_pool.append((ss, se))
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool

    def noun_chunks(self, text):
        doc = self.nlp(text)

        temp = [chunk.text for chunk in doc.noun_chunks]
        PRON = [token.lemma_ for token in doc if token.pos_ == "PRON"]
        VERB = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        NOUN = [token.lemma_ for token in doc if token.pos_ == "NOUN"]
        PROPN = [token.lemma_ for token in doc if token.pos_ == "PROPN"]
        values_to_remove = PRON + VERB + NOUN + PROPN
        temp = list(filter(lambda x: x not in values_to_remove, temp))
        pool = temp
        new_pool = []
        span_pool = []
        for subtext in pool:
            span = self.span(text, subtext)
            if span == None:
                continue
            ss, se, new_sub_text = span
            new_pool.append(subtext)
            span_pool.append((ss, se))
        assert len(new_pool) == len(span_pool)
        return new_pool, span_pool
    def sentence_spacy(self, doc, ids=0):
        sentence_pool = []
        span_sentence_pool = []
        for sent in doc.sents:
            sentence_pool.append(str(sent))
            span_sentence_pool.append((sent.start+ids, sent.end+ids))
        new_sentence_pool, new_span_sentence_pool = self.concat_sentennce(sentence_pool, span_sentence_pool)

        assert len(new_sentence_pool) == len(new_span_sentence_pool)
        return new_sentence_pool, new_span_sentence_pool
    def concat_sentennce(self, sentence_pool, span_sentence_pool):
        # print(sentence_pool)
        # print("=============================concat_sentennce=======================")
        new_sentence_pool = []
        new_span_sentence_pool = []
        for i in range(len(sentence_pool)):
            # print("============================ sentence pool ========================")
            for j in range(i, len(sentence_pool)):
                # print("============================ span sentence pool ========================")
                ss = span_sentence_pool[i][0]
                se = span_sentence_pool[j][1]
                span = ' '.join(sentence_pool[i:j+1])
                new_span = self.remove_punctuation(span)
                ss, se, new_span = self.span(span, new_span, ss)
                # print("span:"+span)
                # print("ss:"+str(ss))
                # print("se:"+str(se))
                # print("new_span:"+str(new_span))
                new_sentence_pool.append(span)
                new_span_sentence_pool.append((ss, se))
        # print("=============================concat_sentennce=======================")
        assert len(new_sentence_pool) == len(new_span_sentence_pool)
        return new_sentence_pool, new_span_sentence_pool
    def remove_duplicate(self, sentence_pool, span_sentence_pool):
        dup_index = [1] * len(sentence_pool)
        for i in range(len(sentence_pool)-1):
            for j in range(i+1, len(sentence_pool)):
                if sentence_pool[i] == sentence_pool[j]:
                    dup_index[j] = 0
        sentence_pool = [sentence_pool[index] for index in range(len(dup_index)) if dup_index[index]]
        span_sentence_pool = [span_sentence_pool[index] for index in range(len(dup_index)) if dup_index[index]]
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool
    def remove_unclear_content(self, pool, span_pool):
        sentence_pool=[]
        span_sentence_pool=[]
        for i in range(len(pool)):
            words = pool[i].split()
            doc = self.nlp(pool[i])
        
            if doc[len(doc)-1].pos_ != "PROPN":
                if words[len(words)-1][0].isupper():
                    continue
            if doc[0].pos_ == "VERB":
                if words[len(words)-1][0].islower():
                    continue
                if words[0][0].islower():
                    continue
            if (words[len(words)-1] not in self.unclear_eos_words) and (words[0] not in self.unclear_bos_words):
                sentence_pool.append(pool[i])
                span_sentence_pool.append(span_pool[i])

        pool = sentence_pool
        span_pool = span_sentence_pool
        # print(pool)
        index_list = [1] * len(pool)
        for i in range(len(pool)):
            # words = pool[i].split()
            doc = self.nlp(pool[i])
            strin = pool[i]
            for j in range(i+1, len(pool)):
                rep_strin = pool[j]

                if len(rep_strin) < len(strin):
                    break
                
                dup = rep_strin[0:len(strin)]

                if dup != strin:
                    break

                other = rep_strin[len(strin):]
                # rep_words = pool[j].split()
                # print("resul:" + other)
                rep_doc = self.nlp(other)
                # lll = [(rep_doc[k].pos_, rep_doc[k].lemma_) for k in range(len(rep_doc))]
                # print(lll)
                rep_verb_num = 0
                rep_adj_num = 0
                rep_noun_num = 0
                rep_adv_num = 0
                for rep_token in rep_doc:
                    # print(rep_token.pos_)
                    if rep_token.pos_ == 'NOUN':
                        rep_noun_num+=1
                    if rep_token.pos_ == 'ADJ':
                        rep_adj_num+=1
                    if rep_token.pos_ == 'VERB':
                        rep_verb_num+=1
                    if rep_token.pos_ == 'ADV':
                        rep_adv_num+=1
                rep_sum_num = rep_verb_num + rep_adj_num + rep_noun_num + rep_adv_num
                # print("i: "+str(i)+"  "+str(sum_num) + "   "+ str(j)+"   "+str(rep_sum_num))
                if rep_sum_num == 0:
                    index_list[j] = 0

            verb_num = 0
            adj_num = 0
            pron_num = 0
            adv_num = 0
            for token in doc:
                # print(rep_token.pos_)
                if token.pos_ == 'PRON':
                    pron_num+=1
                if token.pos_ == 'ADJ':
                    adj_num+=1
                if token.pos_ == 'VERB':
                    verb_num+=1
                if token.pos_ == 'ADV':
                    adv_num+=1

            if pron_num == adv_num and verb_num == 0 and adj_num==0:
                    index_list[i] = 0


        # print(index_list)
        sentence_pool = []
        span_sentence_pool = []

        for i in range(len(index_list)):
            if index_list[i] == 1:
                sentence_pool.append(pool[i])
                span_sentence_pool.append(span_pool[i])
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool

    def remove_end_by_unclear_sentence(self, pool, span_pool):
        index_list = [1] * len(pool)

        for i in range(len(index_list)):
            for j in range(len(self.unclear_eos_words)):
                if pool[i].endswith(self.unclear_eos_words[j]):
                    index_list[i] = 0
                    break

        sentence_pool = []
        span_sentence_pool = []

        for i in range(len(index_list)):
            if index_list[i] == 1:
                sentence_pool.append(pool[i])
                span_sentence_pool.append(span_pool[i])
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool

    def remove_dup_words(self, pool, span_pool):
        sentence_pool=[]
        span_sentence_pool=[]
        index_list = [1] * len(pool)
        for i in range(len(pool)-1):
            strin = pool[i]
            words = strin.split()
            for j in range(i+1, len(pool)):
                rep_strin = pool[j]

                if len(rep_strin) < len(strin):
                    break
                
                dup = rep_strin[0:len(strin)]

                if dup != strin:
                    break

                other = rep_strin[len(strin):]
                other = self.remove_punctuation(other)
                if other == words[len(words)-1]:
                    index_list[i] = 0


        sentence_pool = []
        span_sentence_pool = []

        for i in range(len(index_list)):
            if index_list[i] == 1:
                sentence_pool.append(pool[i])
                span_sentence_pool.append(span_pool[i])
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool

    def remove_dup_sentences(self, pool, span_pool):
        sentence_pool=[]
        span_sentence_pool=[]
        index_list = [1] * len(pool)
        for i in range(len(pool)-1):
            strin = pool[i]
            sentences = re.split(' ; | , | ! ', strin)
            for j in range(i+1, len(pool)):
                rep_strin = pool[j]

                if len(rep_strin) < len(strin):
                    break
                
                dup = rep_strin[0:len(strin)]

                if dup != strin:
                    break

                other = rep_strin[len(strin):]
                other = self.remove_punctuation(other)
                if other == sentences[len(sentences)-1]:
                    index_list[i] = 0


        sentence_pool = []
        span_sentence_pool = []

        for i in range(len(index_list)):
            if index_list[i] == 1:
                sentence_pool.append(pool[i])
                span_sentence_pool.append(span_pool[i])
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool

    def remove_unclear_first_sentence(self, pool, span_pool):
        index_list = [1] * len(pool)
        delimiters = " ? ", " ! ", " . ", " , "
        regex_pattern = '|'.join(map(re.escape, delimiters))

        for i in range(len(pool)):
            # print(pool[i])
            sentences = re.split(regex_pattern, pool[i])
            if len(sentences) == 1:
                continue
            # print(sentences)
            sentence = sentences[0]
            rep_doc = self.nlp(sentence)
            # lll = [(rep_doc[k].pos_, rep_doc[k].lemma_) for k in range(len(rep_doc))]
            # print(lll)
            rep_verb_num = 0
            rep_adj_num = 0
            rep_noun_num = 0
            rep_adv_num = 0
            rep_num_num = 0
            for rep_token in rep_doc:
                # print(rep_token.pos_)
                if rep_token.pos_ == 'NOUN':
                    rep_noun_num+=1
                if rep_token.pos_ == 'ADJ':
                    rep_adj_num+=1
                if rep_token.pos_ == 'VERB':
                    rep_verb_num+=1
                if rep_token.pos_ == 'ADV':
                    rep_adv_num+=1
                if rep_token.pos_ == 'NUM':
                    rep_num_num+=1
            rep_sum_num = rep_verb_num + rep_adj_num + rep_noun_num + rep_adv_num + rep_num_num
            if rep_sum_num == rep_num_num and rep_num_num>0:
                index_list[i] = 0

        sentence_pool = []
        span_sentence_pool = []

        for i in range(len(index_list)):
            if index_list[i] == 1:
                sentence_pool.append(pool[i])
                span_sentence_pool.append(span_pool[i])
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool

    def remove_unclear_last_sentence(self, pool, span_pool):
        index_list = [1] * len(pool)
        # print(pool)
        delimiters = " ? ", " ! ", " . "
        regex_pattern = '|'.join(map(re.escape, delimiters))

        for i in range(len(pool)):
            # print(pool[i])
            sentences = re.split(regex_pattern, pool[i])
            if len(sentences) == 1:
                continue

            sentence = sentences[len(sentences)-1]
            rep_doc = self.nlp(sentence)
            # lll = [(rep_doc[k].pos_, rep_doc[k].lemma_) for k in range(len(rep_doc))]
            # print(lll)
            rep_verb_num = 0
            rep_adj_num = 0
            rep_noun_num = 0
            rep_adv_num = 0
            rep_pron_num = 0 
            rep_aux_num = 0
            for rep_token in rep_doc:
                # print(rep_token.pos_)
                if rep_token.pos_ == 'PRON':
                    rep_pron_num+=1
                if rep_token.pos_ == 'NOUN':
                    rep_noun_num+=1
                if rep_token.pos_ == 'ADJ':
                    rep_adj_num+=1
                if rep_token.pos_ == 'VERB':
                    rep_verb_num+=1
                if rep_token.pos_ == 'ADV':
                    rep_adv_num+=1
                if rep_token.pos_ == 'AUX':
                    rep_aux_num+=1
            rep_sum_num = rep_verb_num + rep_adj_num + rep_noun_num + rep_adv_num + rep_pron_num
            if rep_sum_num < 2:
                index_list[i] = 0

        sentence_pool = []
        span_sentence_pool = []

        for i in range(len(index_list)):
            if index_list[i] == 1:
                sentence_pool.append(pool[i])
                span_sentence_pool.append(span_pool[i])
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool
    def remove_transition_sentence(self, pool, span_pool):
        index_list = [1] * len(pool)

        for i in range(len(pool)-1):
            sentences = pool[i].split(" . ")
            sentence = sentences[len(sentences)-1]
            # print("sentence: "+sentence)
            for j in range(i+1, len(pool)):

                if pool[j].startswith(sentence):
                    rep_doc = self.nlp(sentence)
                    rep_verb_num = 0
                    rep_pron_num = 0
                    # rep_adv_num = 0
                    for rep_token in rep_doc:
                        # print(rep_token.pos_)
                        if rep_token.pos_ == 'PRON':
                            rep_pron_num+=1
                        if rep_token.pos_ == 'VERB':
                            rep_verb_num+=1
                        # if rep_token.pos_ == 'ADV':
                        #     rep_adv_num+=1
                    
                    if rep_pron_num == 1 and rep_verb_num == 1:
                        index_list[i] = 0
                    # if rep_pron_num == rep_adv_num and rep_verb_num == 0:
                    #     index_list[i] = 0

        sentence_pool = []
        span_sentence_pool = []

        for i in range(len(index_list)):
            if index_list[i] == 1:
                sentence_pool.append(pool[i])
                span_sentence_pool.append(span_pool[i])
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool
    def remove_interjections_in_the_end(self, pool, span_pool):
        index_list = [1] * len(pool)

        for i in range(len(index_list)):
            doc = self.nlp(pool[i])
            if doc[len(doc)-1].pos_ == 'INTJ':
                index_list[i] = 0

        sentence_pool = []
        span_sentence_pool = []

        for i in range(len(index_list)):
            if index_list[i] == 1:
                sentence_pool.append(pool[i])
                span_sentence_pool.append(span_pool[i])
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool

    def remove_more_adj(self, pool, span_pool):
        index_list = [1] * len(pool)

        for i in range(len(index_list)):
            doc = self.nlp(pool[i])
            
            # rep_verb_num = 0
            # rep_adj_num = 0
            # rep_noun_num = 0
            # rep_adv_num = 0
            # rep_intj_num = 0
            # rep_pron_num = 0
            # rep_aux_num = 0
            # for rep_token in doc:
            #     # print(rep_token.pos_)
            #     if rep_token.pos_ == 'PRON':
            #         rep_pron_num+=1
            #     if rep_token.pos_ == 'NOUN':
            #         rep_noun_num+=1
            #     if rep_token.pos_ == 'ADJ':
            #         rep_adj_num+=1
            #     if rep_token.pos_ == 'VERB':
            #         rep_verb_num+=1
            #     if rep_token.pos_ == 'ADV':
            #         rep_adv_num+=1
            #     if rep_token.pos_ == 'INTJ':
            #         rep_intj_num+=1
            #     if rep_token.pos_ == 'AUX':
            #         rep_aux_num+=1
            # # print("rep_adj_num: "+str(rep_adj_num))
            # # print("len: "+str(len(doc)))
            # sum_num = rep_verb_num + rep_adj_num + rep_noun_num + rep_adv_num + rep_intj_num + rep_pron_num
            # if rep_adj_num > 0.7*sum_num and rep_adj_num > 1:
            #     index_list[i] = 0

            # if rep_pron_num == 0 and rep_noun_num == 0 and rep_adj_num>0 and rep_verb_num>0:
            #     index_list[i] = 0
            # if rep_pron_num == 0 and rep_noun_num == 1 and rep_adj_num>0 and rep_aux_num==0:
            #     index_list[i] = 0

            rep_adj_num = 0
            rep_adv_num = 0
            rep_intj_num = 0
            rep_noun_num = 0
            index = 0
            while (index < len(doc)):
                if doc[index].pos_ == 'ADJ':
                    rep_adj_num+=1
                if doc[index].pos_ == 'ADV':
                    rep_adv_num+=1
                if doc[index].pos_ == 'NOUN':
                    break

                if doc[index].pos_ == 'PROPN':
                    break
                # if doc[index].pos_ == 'INTJ':
                #     break
                index+=1

            if rep_adj_num > 2 or rep_adv_num > 2:
                index_list[i] = 0


        sentence_pool = []
        span_sentence_pool = []

        for i in range(len(index_list)):
            if index_list[i] == 1:
                sentence_pool.append(pool[i])
                span_sentence_pool.append(span_pool[i])
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool

    def remove_more_intj(self, pool, span_pool):
        index_list = [1] * len(pool)

        for i in range(len(index_list)):
            doc = self.nlp(pool[i])
            
            # rep_verb_num = 0
            # rep_adj_num = 0
            # rep_noun_num = 0
            # rep_adv_num = 0
            # rep_intj_num = 0
            # rep_pron_num = 0
            # rep_aux_num = 0
            # for rep_token in doc:
            #     # print(rep_token.pos_)
            #     if rep_token.pos_ == 'PRON':
            #         rep_pron_num+=1
            #     if rep_token.pos_ == 'NOUN':
            #         rep_noun_num+=1
            #     if rep_token.pos_ == 'ADJ':
            #         rep_adj_num+=1
            #     if rep_token.pos_ == 'VERB':
            #         rep_verb_num+=1
            #     if rep_token.pos_ == 'ADV':
            #         rep_adv_num+=1
            #     if rep_token.pos_ == 'INTJ':
            #         rep_intj_num+=1
            #     if rep_token.pos_ == 'AUX':
            #         rep_aux_num+=1
            # # print("rep_adj_num: "+str(rep_adj_num))
            # # print("len: "+str(len(doc)))
            # sum_num = rep_verb_num + rep_adj_num + rep_noun_num + rep_adv_num + rep_intj_num + rep_pron_num
            # if rep_adj_num > 0.7*sum_num and rep_adj_num > 1:
            #     index_list[i] = 0

            # if rep_pron_num == 0 and rep_noun_num == 0 and rep_adj_num>0 and rep_verb_num>0:
            #     index_list[i] = 0
            # if rep_pron_num == 0 and rep_noun_num == 1 and rep_adj_num>0 and rep_aux_num==0:
            #     index_list[i] = 0

            rep_adj_num = 0
            rep_adv_num = 0
            rep_intj_num = 0
            index = 0
            while (index < len(doc)):
                if doc[index].pos_ == 'INTJ':
                    rep_intj_num+=1

                if doc[index].pos_ == 'PROPN':
                    break
                # if doc[index].pos_ == 'INTJ':
                #     break
                index+=1

            if rep_intj_num > 2:
                index_list[i] = 0


        sentence_pool = []
        span_sentence_pool = []

        for i in range(len(index_list)):
            if index_list[i] == 1:
                sentence_pool.append(pool[i])
                span_sentence_pool.append(span_pool[i])
        assert len(sentence_pool) == len(span_sentence_pool)
        return sentence_pool, span_sentence_pool