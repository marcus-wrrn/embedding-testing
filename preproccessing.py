import pandas as pd
from dataloader import UnprocessedTwitterDataset
from model_parts.sentence_encoder import SentenceEncoder
import torch
from torch.utils.data import DataLoader

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

def get_chunks(df: pd.DataFrame, chunk_size: int):
    return [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

def preprocess_and_save(model: SentenceEncoder, save_directory: str, df: pd.DataFrame, chunk_size=1000):
    print(f"Encoding to: {save_directory}")
    chunks = get_chunks(df, chunk_size)
    for i, chunk in enumerate(chunks):
        print(f"Encoding Chunk: {i}/{len(chunks)}")
        chunk = chunk.copy()
        # Convert tensor encodings to lists
        chunk["encodings"] = chunk[" text"].apply(lambda x: model.encode(x).cpu().numpy().tolist())
        save_path = save_directory + f"_batch.{i}.json"
        chunk.to_json(save_path)

    # df["encodings"] = df[" text"].apply(lambda x: encode_data(model, x))
   
    # save_data(save_file_path, filename, df)

def preproccessing(dataset: UnprocessedTwitterDataset, model: SentenceEncoder, batch_size=1000):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch_data, batch_target in dataloader:
        ...

def merge_data(dir_path):
    ...

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = "./Data/twitter_sentiment/training.1600000.processed.noemoticon.csv"
    
    train_num = 200000
    test_num = 5000
    valid_num = 2000

    # Retrieve data
    train_data = load_dataset(data_path, train_num, 0)
    test_data = load_dataset(data_path, test_num, train_num)
    valid_data = load_dataset(data_path, valid_num, train_num + test_num)

    # Initialize Sentence encoder
    encoder = SentenceEncoder(device)
    
    save_file_path = "./Data/twitter_sentiment/preprocessing/"
    preprocess_and_save(encoder, save_file_path + "train_batch/", train_data)
    preprocess_and_save(encoder, save_file_path + "test_batch/", test_data, chunk_size=2500)
    preprocess_and_save(encoder, save_file_path + "validation_batch/", valid_data, chunk_size=1000)

if __name__ == "__main__":
    main()