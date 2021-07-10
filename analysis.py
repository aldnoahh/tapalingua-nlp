import os
import sys
from itertools import groupby
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from pydub import AudioSegment

# files                                                                         
src = sys.argv[1]
nam= src.rsplit('.', 1)[0]
dst= nam+ ".wav"

# convert wav to mp3                                                            
sound =  AudioSegment.from_file(src, "mp4")
sound.export(dst, format="wav")

audio, rate = librosa.load(dst, sr = 16000)

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


input_values = tokenizer(audio, return_tensors = "pt").input_values
logits = model(input_values).logits
prediction = torch.argmax(logits, dim = -1)
transcription = tokenizer.batch_decode(prediction)[0]


# this is where the logic starts to get the start and end timestamp for each word
words = [w for w in transcription.split(' ') if len(w) > 0]
prediction = prediction[0].tolist()
duration_sec = input_values.shape[1] / rate


ids_w_time = [(i / len(prediction) * duration_sec, _id) for i, _id in enumerate(prediction)]
# remove entries which are just "padding" (i.e. no characers are recognized)
ids_w_time = [i for i in ids_w_time if i[1] != tokenizer.pad_token_id]
# now split the ids into groups of ids where each group represents a word
split_ids_w_time = [list(group) for k, group
                    in groupby(ids_w_time, lambda x: x[1] == tokenizer.word_delimiter_token_id)
                    if not k]

assert len(split_ids_w_time) == len(words)  # make sure that there are the same number of id-groups as words. Otherwise something is wrong

word_start_times = []
word_end_times = []
for cur_ids_w_time, cur_word in zip(split_ids_w_time, words):
    _times = [_time for _time, _id in cur_ids_w_time]
    word_start_times.append(min(_times))
    word_end_times.append(max(_times))
    
#print(len(words))
#print(len(word_start_times))
#print(len(word_end_times))
#print(transcription)
#for i in range(len(words)):
#    print("{}      {}     {}".format(words[i], word_start_times[i], word_end_times[i]))
    
fname= nam+ ".txt"
with open(fname, 'w') as f:
    print(transcription,file=f)
    for i in range(len(words)):
        print("{}      {}     {}".format(words[i], word_start_times[i], word_end_times[i]),file=f)
    
    