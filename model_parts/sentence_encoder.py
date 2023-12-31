from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SentenceEncoder:
    """
    Sentence Transformer used for encoding input sentences

    TODO: Add max_token length
    """
    def __init__(self, 
                 device,
                 tokenizer_path="sentence-transformers/all-mpnet-base-v2", 
                 model_path="sentence-transformers/all-mpnet-base-v2" 
                ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.device = device

    def tokenize_sentences(self, sentences):
        return self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
    
    @torch.no_grad()
    def get_token_embeddings(self, tokenized_sentences):
        return self.model(**tokenized_sentences)

    def encode(self, sentences, normalize=True) -> torch.Tensor:
        tokenized_sents = self.tokenize_sentences(sentences)
        token_embeddings = self.get_token_embeddings(tokenized_sents)
        sentence_embeddings = mean_pooling(token_embeddings, tokenized_sents['attention_mask'])
        # Consider not adding normalization (does that improve performance?)
        return F.normalize(sentence_embeddings, p=2, dim=1) if normalize else sentence_embeddings

def main():
   # Sentences we want sentence embeddings for
    sentences = ['This is an example sentence', 'Each sentence is converted to an encoding', 'Hello World']

    model = SentenceEncoder()
    sentence_embeddings = model.encode(sentences)

    print("Sentence embeddings:")
    print(sentence_embeddings.shape)

if __name__ == "__main__":
    main()