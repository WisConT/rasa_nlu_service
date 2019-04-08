import pandas as pd

def read_numberbatch_to_csv(txt_path, csv_path):
    read = pd.read_csv(txt_path, sep=" ", skiprows=[0], names=["id"] + range(300))
    read.to_csv(path_or_buf=csv_path, index=False)

def get_english_numberbatch_csv(in_csv_path, out_csv_path):
    df = pd.read_csv(in_csv_path)
    df = df[df['id'].str.contains('/c/en/')]
    df.to_csv(path_or_buf=out_csv_path, index=False)

def get_embeddings(df, words):
    ids = ['/c/en/' + word for word in words]
    embeddings = [e[1:] for e in df[df['id'].isin(ids)].values]
    return embeddings
    
    
if __name__ == "__main__":
    # read_numberbatch_to_csv('../embeddings/numberbatch-17.06.txt', '../embeddings/numberbatch.csv')
    # get_english_numberbatch_csv('../embeddings/numberbatch.csv', '../embeddings/numberbatch_eng.csv')
    # print(pd.read_csv('../embeddings/numberbatch_eng.csv')[0:10])
    df = pd.read_csv('../embeddings/numberbatch_eng.csv')
    print(get_embeddings(df, ["hello", "world"]))