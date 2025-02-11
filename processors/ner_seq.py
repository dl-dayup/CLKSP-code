""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import copy
import json
from .utils_ner import DataProcessor
from itertools import permutations
import numpy as np
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len,segment_ids, label_ids, cl_labels, inputs_embeds):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len
        self.cl_labels = cl_labels
        self.inputs_embeds = inputs_embeds

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels, all_cl_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:,:max_len]
    all_cl_labels = all_cl_labels[:max_len,:max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_cl_labels, all_lens

def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer, ee, prompt,
                                 cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=1,
                                 sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                 sequence_a_segment_id=0,mask_padding_with_zero=True,):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    n_prompt = 0 # prompt.shape[0]
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        if isinstance(example.text_a,list):
            example.text_a = " ".join(example.text_a)
        tokens = tokenizer.tokenize(example.text_a)
        label_ids = [label_map[x] for x in example.labels]
        #if len(example.text_a) != len(example.labels):
        #    continue

        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2 + n_prompt 
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [label_map['O']]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [label_map['O']]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [label_map['O']] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)

        if len(input_ids) != len(label_ids):
            continue
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids) - n_prompt
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token] * padding_length

        # prompt
        tem = torch.tensor(input_ids).unsqueeze(0)
        device = torch.device('cpu')
        ee=ee.to(device)
        embed = ee(tem).squeeze(0)
        inputs_embeds = ee(tem).squeeze(0)
        #inputs_embeds = torch.concat([embed,prompt],0)
        #print('feature', embed.shape, inputs_embeds.shape)
        '''
        # 原始label数据位移后移
        for i in range(n_prompt):
            #label_ids.insert(i,label_map['X']) 
            label_ids.append(label_map['X']) 
        
        #input_mask = ([0] * n_prompt) + input_mask
        input_mask = input_mask+([0] * n_prompt)
        #segment_ids = ([pad_token_segment_id] * n_prompt) + segment_ids
        segment_ids = segment_ids+([pad_token_segment_id] * n_prompt)
        '''
        # cl
        cl_labels = np.zeros((len(input_mask), len(input_mask)), dtype=int)
        # knowledge cl
        # token cl 
        # label 按类型放入dict中，key 是 label_id，value 是此 label_id 对应的下标列表
        flag_dict={}
        start_index = max_seq_length - n_prompt
        for index, l in enumerate(label_ids):
            '''
            if l in (label_map['B-tes'],label_map['I-tes'],label_map['S-tes']):
                    cl_labels[start_index+0, index]=l
                    cl_labels[index,start_index+0]=l
            if l in (label_map['B-exa'],label_map['I-exa'],label_map['S-exa']):
                    cl_labels[start_index+1, index]=l
                    cl_labels[index,start_index+1]=l
            if l in (label_map['B-dru'],label_map['I-dru'],label_map['S-dru']):
                    cl_labels[start_index+2, index]=l
                    cl_labels[index,start_index+2]=l
            if l in (label_map['B-sit'],label_map['I-sit'],label_map['S-sit']):
                    cl_labels[start_index+3, index]=l
                    cl_labels[index,start_index+3]=l
            if l in (label_map['B-sur'],label_map['I-sur'],label_map['S-sur']):
                    cl_labels[start_index+4, index]=l
                    cl_labels[index,start_index+4]=l
            if l in (label_map['B-dis'],label_map['I-dis'],label_map['S-dis']):
                    cl_labels[start_index+5, index]=l
                    cl_labels[index,start_index+5]=l
            '''

            if index < start_index:
                if l in flag_dict.keys():
                    flag_dict[l].append(index)
                else:
                    flag_dict[l]=[index]
        #print('kCL:', cl_labels.sum())
        #将除"O"以外的其他label的下标列表中元素进行两两组合，并赋值为相应的label_id，此为构建token级的对比学习正样本对
        #'''

        for i, t in flag_dict.items():
            if i not in (label_map['O'],label_map['X'],label_map["[START]"], label_map["[END]"] ) and len(t)>1: #去掉"O"label和其他标识label
                for (c, d) in permutations(t, 2):
                    cl_labels[c,d]=i
        #'''
        #print('tCL:', cl_labels.sum())
        #cl_labels = cl_labels - np.eye(len(input_mask)) * 1e12
        input_len = len(input_mask)
        #print(len(input_ids),len(input_mask),len(segment_ids),len(label_ids))
        #assert len(input_ids) == max_seq_length
        #assert len(input_mask) == max_seq_length+n_prompt
        #assert len(segment_ids) == max_seq_length+n_prompt
        #assert len(label_ids) == max_seq_length+n_prompt
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,input_len = input_len,
                                      segment_ids=segment_ids, label_ids=label_ids, cl_labels=cl_labels, inputs_embeds=inputs_embeds))
    return features


class CnerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        """See base class."""
        return ["X",'B-CONT','B-EDU','B-LOC','B-NAME','B-ORG','B-PRO','B-RACE','B-TITLE',
                'I-CONT','I-EDU','I-LOC','I-NAME','I-ORG','I-PRO','I-RACE','I-TITLE',
                'O','S-NAME','S-ORG','S-RACE',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class CluenerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["X", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name',
                'B-organization', 'B-position','B-scene',"I-address",
                "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name',
                'I-organization', 'I-position','I-scene',
                "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie',
                'S-name', 'S-organization', 'S-position',
                'S-scene','O',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class CMeEEProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["X", "B-dep", "B-pro", "B-ite", "B-equ", "B-bod", "B-dis", "B-sym", "B-mic", "B-dru",
                "I-dep", "I-pro", "I-ite", "I-equ", "I-bod", "I-dis", "I-sym", "I-mic", "I-dru",
                "S-dep", "S-pro", "S-ite", "S-equ", "S-bod", "S-dis", "S-sym", "S-mic", "S-dru",
               'O',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class CNMERProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_text(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_text(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_text(os.path.join(data_dir, "test.txt")), "test")
    def get_labels(self):
        """See base class."""
        return ["X", "B-dis", "B-sur", "B-sit", "B-dru", "B-exa", "B-tes", 
                "I-dis", "I-sur", "I-sit", "I-dru", "I-exa", "I-tes",
                "S-dis", "S-sur", "S-sit", "S-dru", "S-exa", "S-tes", 
               'O',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

ner_processors = {
    "cner": CnerProcessor,
    "cmeee": CMeEEProcessor,
    "cnmer": CNMERProcessor,
    'cluener':CluenerProcessor
}
