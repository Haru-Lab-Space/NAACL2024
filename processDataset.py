import torch
from torch import nn
import json
from tqdm import tqdm
from augment import AugmentData
import string
import numpy as np

class ProcessDataset(nn.Module):
    def __init__(self):
        super(ProcessDataset, self).__init__()
        self.build()
    def build(self):
        config = self.read_json("config.json")
        self.id2label = config['id2label']
        self.label2id = config['label2id']
        self.train = config['train']
        self.test = config['test']
        self.left_padding = config['left_padding']
        self.right_padding = config['right_padding']
        self.dataset_folder = config['dataset folder']
        self.candidate_num = config['candidate num']
        self.overlap_threshold = config['overlap_threshold']
        self.punctuation_error = config['punctuation error']
        self.augmentModel = AugmentData()

    def forward(self, type="train"):
        new_dataset = self.convert2process_dataset(type)
        self.convertprocess2train_dataset(new_dataset, type=type)

    def convertprocess2train_dataset(self, dataset, type="train"):
        if type == "train":
            new_dataset = []
            mean_match = []
            max_span_label = 0
            max_word_paragraph = 0
            max_conv = 0
            max_utter = 0
            logger_message = f'Convert process dataset to training dataset'
            progress_bar = tqdm(dataset,
                                desc=logger_message, initial=0, dynamic_ncols=True)
            for i, _ in enumerate(progress_bar):
                sample = dataset[i]
                conversation_ID = sample["conversation_ID"]
                conversation = sample["conversation"]
                # print(len(conversation))
                # print(self.right_padding)
                text = []
                utter = []
                speaker = []
                emotion = []
                casual_text = []
                casual_span = []
                casual = []
                for j in range(len(conversation.keys())):
                    # print(conversation[j])
                    text.append(conversation[j+1]['text'])
                    utter.append(j+1)
                    speaker.append(conversation[j+1]['speaker'])
                    emotion.append(conversation[j+1]['emotion'])
                    casual_text.append(conversation[j+1]['casual text'])
                    casual_span.append(conversation[j+1]['casual span'])
                    casual.append(conversation[j+1]['casual'])
                # print(utter)
                # print(text)
                for j in range(len(conversation)):
                    start_ids = max(0, j-self.left_padding)
                    end_ids = min(j+self.right_padding+1, len(conversation))
                    if start_ids > end_ids:
                        break
                    # print(str(start_ids) + "    " + str(end_ids))

                    casual_text_sample = []
                    casual_span_sample = []
                    for k in range(start_ids, end_ids):
                        # print(casual_text[k])
                        casual_text_sample.extend(casual_text[k])
                        casual_span_sample.extend(casual_span[k])

                    match = 0
                    # print(casual[j])
                    span_label = [0] * len(casual_span_sample)
                    for span in casual[j]:
                        for k in range(len(casual_span_sample)):
                            utter_ID = casual_span_sample[k][0]
                            span_label[k] = self.overlap(span, casual_text_sample[k], text[utter_ID-1])
                            match += span_label[k]

                    paragraph = " ".join(text[start_ids:end_ids])


                    if len(casual[j])==0:
                        match = 1
                    else:
                        match = match/len(casual[j])
                        mean_match.append(match)

                    for k in range(len(span_label)):
                        new_dataset.append({
                            "conversation_ID": conversation_ID,
                            "utterance_ID": utter[j],
                            "paragraph": paragraph,
                            "match": match,
                            "speaker_pool": speaker[start_ids:end_ids],
                            "emotion": emotion[j],
                            "casual_pool": casual_text_sample[k],
                            "casual_span_pool": casual_span_sample[k],
                            "span_label": span_label[k]
                        })

                    if len(casual_span_sample)>max_span_label:
                        max_span_label = len(casual_span_sample)
                    words = paragraph.split()
                    if len(words) > max_word_paragraph:
                        max_word_paragraph = len(words)
                        max_utter = utter[j]
                        max_conv = conversation_ID
                    if len(casual_span_sample) > 32:
                        print("conver: "+str(conversation_ID)+"     "+str(utter[j]) +"    " + str(len(casual_span_sample)))
            print("Mean match: " + str(sum(mean_match) / len(mean_match)))
            print("Max span_label: " + str(max_span_label))
            print("Max max_word_paragraph: " + str(max_word_paragraph))
            print("Max max_utter: " + str(max_utter))
            print("Max max_conv: " + str(max_conv))
            train_dataset_path = self.dataset_folder + type + "/"  + type + ".json"
            self.write_json(train_dataset_path, new_dataset)
        else:
            new_dataset = []
            mean_match = []
            max_span_label = 0
            max_word_paragraph = 0
            max_conv = 0
            max_utter = 0
            logger_message = f'Convert process dataset to testing dataset'
            progress_bar = tqdm(dataset,
                                desc=logger_message, initial=0, dynamic_ncols=True)
            for i, _ in enumerate(progress_bar):
                sample = dataset[i]
                conversation_ID = sample["conversation_ID"]
                conversation = sample["conversation"]
                index_list = list(range(-self.left_padding, len(conversation) + self.right_padding))
                text = []
                utter = []
                speaker = []
                casual_text = []
                casual_span = []
                for j in range(len(conversation.keys())):
                    text.append(conversation[j+1]['text'])
                    utter.append(j+1)
                    speaker.append(conversation[j+1]['speaker'])
                    casual_text.append(conversation[j+1]['casual text'])
                    casual_span.append(conversation[j+1]['casual span'])
                for j in range(len(conversation)):
                    start_ids = max(0, j-self.left_padding)
                    end_ids = min(j+self.right_padding+1, len(conversation))
                    if start_ids > end_ids:
                        break

                    casual_text_sample = []
                    casual_span_sample = []
                    for k in range(start_ids, end_ids):
                        casual_text_sample.extend(casual_text[k])
                        casual_span_sample.extend(casual_span[k])
                        assert len(casual_text_sample) == len(casual_span_sample)

                    # print("===== k: "+str(k))
                    # print("===== casual_span_sample: "+str(casual_span_sample))
                    # print("===== casual_text_sample: "+str(casual_text_sample))
                    if len(casual_span_sample)>max_span_label:
                        max_span_label = len(casual_span_sample)
                    if len(casual_span_sample) > 32:
                        print("conver: "+str(conversation_ID)+"     "+str(utter[j]) +"    " + str(len(casual_span_sample)))
                    paragraph = " ".join(text[start_ids:end_ids])

                    for k in range(len(casual_text_sample)):
                        new_dataset.append({
                            "conversation_ID": conversation_ID,
                            "utterance_ID": utter[j],
                            "paragraph": paragraph,
                            "speaker_pool": speaker[start_ids:end_ids],
                            "casual_pool": casual_text_sample[k],
                            "casual_span_pool": casual_span_sample[k],
                        })

                    
                    words = paragraph.split()
                    if len(words) > max_word_paragraph:
                        max_word_paragraph = len(words)
                        max_utter = utter[j]
                        max_conv = conversation_ID
            # print("Mean match: " + str(sum(mean_match) / len(mean_match)))
            print("Max span_label: " + str(max_span_label))
            print("Max max_word_paragraph: " + str(max_word_paragraph))
            print("Max max_utter: " + str(max_utter))
            print("Max max_conv: " + str(max_conv))
            train_dataset_path = self.dataset_folder + type + "/"  + type + ".json"
            self.write_json(train_dataset_path, new_dataset)

    def convert2process_dataset(self, type="train"):
        dataset_path = self.dataset_folder + type + "/Subtask_1_" + type + ".json"
        raw_dataset_path = self.dataset_folder + type + "/raw_"  + type + ".json"
        data = self.read_json(dataset_path)
        new_data = []
        max_casual_span = 0
        mean_casual_span = 0
        max_casual_span_pool = []
        mean_casual_span_pool = []
        m_count=0
        n_count=0
        logger_message = f'Convert raw dataset to process dataset'
        progress_bar = tqdm(data,
                            desc=logger_message, initial=0, dynamic_ncols=True)
        for i, _ in enumerate(progress_bar):
            new_conversation = {}
            conversation_ID = data[i]['conversation_ID']
            # print("conversation_ID: "+str(conversation_ID))
            casual_text_pool = []
            casual_span_pool = []
            sum_casual_span = []
            for j in range(len(data[i]['conversation'])):
                index_of_emotion_utterance = data[i]['conversation'][j]['utterance_ID']
                text = data[i]['conversation'][j]['text']
                
                text = text.replace('\u2019', "'")
                speaker = data[i]['conversation'][j]['speaker']
                casual_text = self.augmentModel.augment_sentence(text)
                casual_span = []
                for k in range(len(casual_text)):
                    span = self.span(text, casual_text[k])
                    if span != None:
                        ss, se, _ = span
                        # casual_text.append(casual_text[k])
                        # casual_span_pool.append((ss, se))
                        casual_span.append((ss, se))
                if len(casual_text) > max_casual_span:
                    max_casual_span = len(casual_text)
                casual_span = self.casual_convert(index_of_emotion_utterance, casual_span)

                if "emotion" in data[i]['conversation'][j].keys():
                    emotion = data[i]['conversation'][j]['emotion']
                    new_utter = {
                        "text": text,
                        "speaker": speaker,
                        "emotion": emotion,
                        "casual text": casual_text,
                        "casual span": casual_span,
                        "casual": []
                    }
                else:
                    new_utter = {
                        "text": text,
                        "speaker": speaker,
                        "casual text": casual_text,
                        "casual span": casual_span,
                        "casual": []
                    }
                new_conversation[index_of_emotion_utterance] = new_utter
                casual_text_pool.extend(casual_text)
                casual_span_pool.extend(casual_span)
                sum_casual_span.append(len(casual_text))
                # print("Max casual span utter: "+str(len(casual_text)))
            max_casual_span_pool.append(sum(sum_casual_span))
            mean_casual_span_pool.append(sum(sum_casual_span) / len(sum_casual_span))

            if 'emotion-cause_pairs' in data[i].keys():
                cause_pairs = data[i]['emotion-cause_pairs']
                m, n = self.matching_cause_pairs(casual_text_pool, casual_span_pool, cause_pairs)
                # if m < len(cause_pairs):
                #     print("===============")
                #     print(conversation_ID)
                #     # print("casual: "+str(m)+" , annotated: "+str(len(cause_pairs)))
                #     print(cause_pairs)

                for j in range(len(data[i]['emotion-cause_pairs'])):
                    utter_id, emotion_category = data[i]['emotion-cause_pairs'][j][0].split("_")
                    casual_utter_id, casual = data[i]['emotion-cause_pairs'][j][1].split("_")
                    utter_id = int(utter_id)
                    # casual = self.remove_punctuation(casual)
                    # if "casual" not in new_conversation[utter_id].keys():
                    #     new_conversation[utter_id]["casual"] = [casual]
                    # else:
                    new_conversation[utter_id]["casual"].append(casual)
                    
                m_count+=m
                n_count+=n
            new_data.append({
                "conversation_ID": conversation_ID,
                "conversation": new_conversation
            })
        self.write_json(raw_dataset_path, new_data)
        # print("Max casual span: "+str(max_casual_span))
        # print("Mean casual span: "+str(mean_casual_span))
        # print("Max casual span pool: "+str(sum(max_casual_span_pool)/len(max_casual_span_pool)))
        # print("Mean casual span pool: "+str(sum(mean_casual_span_pool)/len(mean_casual_span_pool)))
        # print("Max casual span conversation: "+str(len(casual_span_pool)))
        # print("Matching casual span: "+str(m_count/n_count))
        return new_data
    def overlap(self, text, sub_text, utter_text):
        # print("text: "+text)
        # print("sub_text: "+sub_text)
        # print("utter_text: "+utter_text)
        span_text = self.span(utter_text, text)
        if real == None:
            return 0
        start_span_text,end_span_text,_ = text
        start_sub_span_text, end_sub_span_text,_ = self.span(utter_text, sub_text)
        duplicate_span = np.abs(start_span_text - start_sub_span_text) + np.abs(end_span_text - end_sub_span_text)
        span = end_span_text - start_span_text
        if duplicate_span/span > self.overlap_threshold:
            return duplicate_span/span
        else:
            return 0
    def span(self, text, sub_text, s_ind=0):
        sub_text = self.remove_punctuation(sub_text)
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

                            if new_sub_text[j]+punctuation == new_text[i + j]:
                                # print(new_sub_text)
                                new_sub_text[j] = new_sub_text[j]+punctuation 
                                ss = i
                                se = i+j
                                print(new_sub_text)
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
    def matching_cause_pairs(self, cause_pairs, casual_span, annotated_cause_pairs):
        # print("cause_pairs: "+str(cause_pairs))
        # print("casual_span: "+str(casual_span))
        # print("annotated_cause_pairs: "+str(annotated_cause_pairs))
        # print("cause_pairs: "+str(len(cause_pairs)))
        # print("casual_span: "+str(len(casual_span)))
        # print("annotated_cause_pairs: "+str(len(annotated_cause_pairs)))
        m = 0
        for i in range(len(annotated_cause_pairs)):
            # print("Macthing")
            ind, casual = annotated_cause_pairs[i][1].split("_")
            casual = self.remove_punctuation(casual)
            # print(ind + "   "+casual)
            for j in range(len(cause_pairs)):
                # print(str(casual_span[j][0]) + " " +cause_pairs[j])
                if np.logical_and((casual == cause_pairs[j]), (int(ind) == casual_span[j][0])):
                    m+=1
                    break
                # else:
                #     if j == len(cause_pairs) -1:
                #         print(annotated_cause_pairs[i])


        return m, len(annotated_cause_pairs)
    def remove_punctuation(self, text):
        # Create a string of all punctuation characters
        punctuation_chars = string.punctuation

        # Remove punctuation at the beginning and end of the string
        cleaned_text = text.strip(punctuation_chars).strip().strip(punctuation_chars).strip().strip(punctuation_chars).strip()

        return cleaned_text
    def casual_convert(self, index, casual_span):
        new_casual_span = []
        # print(casual_span)
        for i in range(len(casual_span)):
            # print(casual_span[i])
            ss, se = casual_span[i]
            new_casual_span.append([index, ss, se])
        return new_casual_span
    def read_json(self, path):
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        return data
    def write_json(self, path, data):
        with open(path, 'w') as file:
            # Write the Python data structure as JSON to the file
            json.dump(data, file, indent=2)