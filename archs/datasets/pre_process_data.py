import spacy 
import os
from tqdm import tqdm
import json
nlp = spacy.load('en_core_web_lg')

def get_verb_noun(description,query_length=None):
    sentence=nlp(description)
    if query_length is not None:
        if len(sentence) > query_length:
            sentence = sentence[:query_length]
    sent_pos = [token.pos_ for token in sentence]
    output= [] 
    verbs = [] 
    nouns = [] 
    for token in sentence:
        if token.pos_ == "NOUN":
            nouns.append(token.lemma_)
            output.append(token.lemma_)
        elif token.pos_ == "VERB":
            verbs.append(token.lemma_)
            output.append(token.lemma_)
    return verbs, nouns, output   

def get_sentence(description):
    sentence=nlp(description)
    output= [] 
    for token in sentence:
        if token.pos_ in ["SPACE"] :
            continue
        output.append(token.lemma_)
    return output   

def description_translate_word(data_path, dataset_type,query_length=None,sos_id=1,eos_id=2):
    if dataset_type == 'charades':

        train_file =  open(os.path.join(data_path, "charades_sta_train.txt")).readlines()
        test_file =  open(os.path.join(data_path, "charades_sta_test.txt")).readlines()
        vocabs = [] 
        progress_bar = tqdm(total=len(test_file))
        max_length = 0
        progress_bar.set_description('test_translate_process') 
        for line in test_file:
            anno, sent = line.split("##")
            verbs, nouns, output  = get_verb_noun(sent)
            vocabs.extend(output)
            max_length = max(max_length,len(output))
            progress_bar.update(1)
        progress_bar.close()
        progress_bar = tqdm(total=len(train_file)) 
        progress_bar.set_description('train_translate_process') 
        for line in train_file:
            anno, sent = line.split("##")
            verbs, nouns, output  = get_verb_noun(sent)
            vocabs.extend(output)
            max_length = max(max_length,len(output))
            progress_bar.update(1)
        progress_bar.close()
        # print('max_length:',max_length)
        vocabs = set(vocabs)
    elif dataset_type == 'activitynet':
        with open(os.path.join(data_path, "train.json"),'r') as f:
            train_file = json.load(f)
        # with open(os.path.join(data_path, "val.json"),'r') as f:
        #     val_file = json.load(f)
        # with open(os.path.join(data_path, "test.json"),'r') as f:
        #     test_file = json.load(f)
        vocabs = [] 
        max_length = 0
        # progress_bar = tqdm(total=len(test_file))
        # progress_bar.set_description('test_translate_process') 
        # for vid, video_anno in test_file.items():
        #     for sent in video_anno['sentences']:
        #         verbs, nouns, output  = get_verb_noun(sent,query_length)
        #         vocabs.extend(output)
        #         max_length = max(max_length,len(output))
        #     progress_bar.update(1)
        # progress_bar.close()
        # progress_bar = tqdm(total=len(val_file))
        # progress_bar.set_description('val_translate_process') 
        # for vid, video_anno in val_file.items():
        #     for sent in video_anno['sentences']:
        #         verbs, nouns, output  = get_verb_noun(sent,query_length)
        #         vocabs.extend(output)
        #         max_length = max(max_length,len(output))
        #     progress_bar.update(1)
        # progress_bar.close()
        progress_bar = tqdm(total=len(train_file))
        progress_bar.set_description('train_translate_process') 
        for vid, video_anno in train_file.items():
            for sent in video_anno['sentences']:
                verbs, nouns, output  = get_verb_noun(sent,query_length)
                vocabs.extend(output)
                max_length = max(max_length,len(output))
            progress_bar.update(1)
        progress_bar.close()
        vocabs_unique = set(vocabs)
        vocabs_bak = []
        for item in vocabs_unique:
            if vocabs.count(item) >=10:
                vocabs_bak.append(item)
        vocabs = set(vocabs_bak)

    else:
        print(dataset_type,'is an invalid dataset_name!!!')
    
    vocab_idx = {k:list(vocabs).index(k)+3 for k in vocabs}
    vocab_idx['PAD'] = 0 
    vocab_idx['<sos>'] = sos_id
    vocab_idx['<eos>'] = eos_id
    idx_vocab = {v:k for k,v in vocab_idx.items()}
    print('max_length:',max_length)
    return vocab_idx,idx_vocab,max_length

    

if __name__ == "__main__":
    data_path = '/raid/file3/code3/DTGv2/data/Charades-STA/I3D'
    dataset_type = 'charades'
    # print(os.path.join(data_path+'/{}_vocab_translate.json'.format(dataset_type)))
    # if not os.path.isfile(os.path.join(data_path+'/{}_vocab_translate.json'.format(dataset_type))):    
    #     with open(data_path+'/{}_vocab_translate.json'.format(dataset_type), 'w') as f:
    #         vocab_idx,idx_vocab,max_length  = description_translate(data_path,dataset_type)
    #         json.dump([vocab_idx, idx_vocab,max_length], f)
    # with open(os.path.join(data_path,'{}_vocab_translate.json'.format(dataset_type)), 'r') as f:
    #     vocab_idx,idx_vocab,max_length = json.load(f)
    # print(vocab_idx)