import pandas as pd 

def read_dialogues(file, split_by='__eou__'):
    # read dialogues by line from a txt file
    utterances = []
    with open(file, 'r') as f:
        dialogues = f.readlines()
    for dialogue in dialogues:
        # split by '__eou__' and remove the last empty string
        dialogue = dialogue.strip().split(split_by)[:-1]
        utterances.extend(dialogue)
    return utterances

def read_dialogue_context(file, split_by='__eou__'):
    # read dialogues by line from a txt file
    contexts = []
    with open(file, 'r') as f:
        dialogues = f.readlines()
    for dialogue in dialogues:
        # split by '__eou__' and remove the last empty string
        dialogue = dialogue.strip().split(split_by)[:-1]
        for i in range(len(dialogue)):
            context = '__eou__'.join(dialogue[:i+1])
            contexts.append(context)
    return contexts

def get_emotion_labels(file):
    # read emotion labels by line from a txt file
    aggregate_labels = []
    with open(file, 'r') as f:
        emotion_labels = f.readlines()
    for each_dialogue in emotion_labels:
        each_dialogue = each_dialogue.strip().split()
        # convert string to int
        each_dialogue = [int(x) for x in each_dialogue]
        aggregate_labels.extend(each_dialogue)
    return aggregate_labels

def build_instructed_finetune_data(input, label_index, label_dict, instruction, save_path):

    fine_tune_data = pd.DataFrame({
        'input': input,
        'output': [label_dict[x] for x in label_index],
    })

    fine_tune_data['instruction'] = instruction
    output_df = fine_tune_data[['instruction', 'input', 'output']]
    output_df.to_csv(save_path, index=False)

if __name__ == "__main__":
    # read dialogues from a txt file
    dialogues_train = read_dialogues('dataset/dailydialog/train/dialogues_train.txt')
    dialogues_valid = read_dialogues('dataset/dailydialog/validation/dialogues_validation.txt')
    dialogues_test = read_dialogues('dataset/dailydialog/test/dialogues_test.txt')
    
    dialogue_contexts_train = read_dialogue_context('dataset/dailydialog/train/dialogues_train.txt')
    dialogue_contexts_valid = read_dialogue_context('dataset/dailydialog/validation/dialogues_validation.txt')
    dialogue_contexts_test = read_dialogue_context('dataset/dailydialog/test/dialogues_test.txt')

    # read emotion labels from a txt file
    emotion_labels_train = get_emotion_labels('dataset/dailydialog/train/dialogues_emotion_train.txt')
    emotion_labels_valid = get_emotion_labels('dataset/dailydialog/validation/dialogues_emotion_validation.txt')
    emotion_labels_test = get_emotion_labels('dataset/dailydialog/test/dialogues_emotion_test.txt')
    
    print('Number of dialogues in train set:', len(dialogues_train))
    print('Number of dialogues in validation set:', len(dialogues_valid))
    print('Number of dialogues in test set:', len(dialogues_test))

    print('Number of dialogue contexts in train set:', len(dialogue_contexts_train))
    print('Number of dialogue contexts in validation set:', len(dialogue_contexts_valid))
    print('Number of dialogue contexts in test set:', len(dialogue_contexts_test))

    print('Number of emotion labels in train set:', len(emotion_labels_train))
    print('Number of emotion labels in validation set:', len(emotion_labels_valid))
    print('Number of emotion labels in test set:', len(emotion_labels_test))

    assert len(dialogues_train) == len(dialogue_contexts_train) == len(emotion_labels_train) 
    assert len(dialogues_valid) == len(dialogue_contexts_valid) == len(emotion_labels_valid)
    assert len(dialogues_test) == len(dialogue_contexts_test) == len(emotion_labels_test)

    label_dict = { 0: 'no emotion', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'happiness', 5: 'sadness', 6: 'surprise'}
    instruction_on_utterance = 'Analyse the emotion of the utterance and predict the emotion label.'
    instruction_on_context = 'Analyse the emotion of the utternace, use the previous conversations as context if needed, and predict the emotion label.'

    build_instructed_finetune_data(dialogues_train, emotion_labels_train, label_dict, instruction_on_utterance, 'dataset/dailydialog/train/emotion_finetune_train.csv')
    build_instructed_finetune_data(dialogues_valid, emotion_labels_valid, label_dict, instruction_on_utterance, 'dataset/dailydialog/validation/emotion_finetune_validation.csv')
    build_instructed_finetune_data(dialogues_test, emotion_labels_test, label_dict, instruction_on_utterance, 'dataset/dailydialog/test/emotion_finetune_test.csv')

    build_instructed_finetune_data(dialogue_contexts_train, emotion_labels_train, label_dict, instruction_on_context, 'dataset/dailydialog/train/emotion_finetune_context_train.csv')
    build_instructed_finetune_data(dialogue_contexts_valid, emotion_labels_valid, label_dict, instruction_on_context, 'dataset/dailydialog/validation/emotion_finetune_context_validation.csv')
    build_instructed_finetune_data(dialogue_contexts_test, emotion_labels_test, label_dict, instruction_on_context, 'dataset/dailydialog/test/emotion_finetune_context_test.csv')
