import numpy as np
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import spacy
from sentence_encoder import SentenceEncoder

def process_data():
    ...

def save_data(filepath, name: str, data: pd.DataFrame):
    numRows = data.shape[0]
    save_file = filepath + f"twitter.{numRows}.{name}.json"
    print(f"Saving: {name} at {save_file}")
    data.to_json(save_file)

def load_dataset(filepath, rownum, start_point, posneg_split=800000):
    halfed_data = rownum // 2
    start_point2 = (start_point // 2) + posneg_split

    df1 = pd.read_csv(filepath, nrows=halfed_data, skiprows=range(1, start_point))
    df2 = pd.read_csv(filepath, nrows=halfed_data, skiprows=range(1, start_point2))
    return pd.concat([df1, df2], ignore_index=True)

def encode_data(model: SentenceEncoder, text: str):
    #print("Encoding data")
    return model.encode(text)[0].tolist()

def preprocess_and_save(model: SentenceEncoder, save_file_path: str, filename: str, df: pd.DataFrame):
    print(f"Encoding: {filename}")
    df["encodings"] = df[" text"].apply(lambda x: encode_data(model, x))
   
    save_data(save_file_path, filename, df)


def main():
    #nlp = spacy.load("en_core_web_sm")
    #model = SentenceTransformer("all-mpnet-base-v2")
    
    data_path = "./Data/twitter_sentiment/training.1600000.processed.noemoticon.csv"
    train_num = 5000
    test_num = 1000
    valid_num = 500

    # Retrieve data
    train_data = load_dataset(data_path, train_num, 0)
    test_data = load_dataset(data_path, test_num, train_num)
    valid_data = load_dataset(data_path, valid_num, train_num + test_num)

    # Initialize Sentence encoder
    encoder = SentenceEncoder()
    save_file_path = "./Data/twitter_sentiment/"
    preprocess_and_save(encoder, save_file_path, "train", train_data)
    preprocess_and_save(encoder, save_file_path, "test", test_data)
    preprocess_and_save(encoder, save_file_path, "valid", valid_data)

if __name__ == "__main__":
    main()