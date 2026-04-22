from funasr import AutoModel

model = AutoModel(model="iic/emotion2vec_plus_large")

wav_file = "/cluster/home/bekadb/Speech-Emotion-Recognition/datasets/IEMOCAP/Session1/sentences/wav/Ses01F_script02_1/Ses01F_script02_1_F001.wav"
res = model.generate(wav_file, output_dir="./outputs", granularity="utterance", extract_embedding=False)
print(res)


# module load FFmpeg/6.0-GCCcore-13.2.0