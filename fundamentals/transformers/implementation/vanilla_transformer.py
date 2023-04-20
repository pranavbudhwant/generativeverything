import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()

        # embed_size: Embedding size of each token
        # heads: Number of heads in the multi-head attention
        self.embed_size = embed_size
        self.heads = heads
        # We split & divide the embedding and send these subtokens to different attention heads, so that each attention head can learn a different representation of the input
        
        self.head_dim = embed_size // heads # Embedding size of each head

        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads" # Check if embedding size is divisible by heads or not (if not, then we can't split the embedding into equal parts, so we throw an error)

        # We have 3 linear layers for each head, one for keys, one for values and one for queries
        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False) # We don't use bias here because we are using layer normalization later on (similar to batch norm, which is a form of bias)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        
        # We have a linear layer to combine the outputs of the different heads
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size) # We multiply heads * head_dim because we are concatenating the outputs of the different heads 

    def forward(self, values, keys, query, mask):
        # mask: Mask to mask out the padding tokens
        # values: Values of the input (size: batch_size, value_len, embed_size)
        # keys: Keys of the input (size: batch_size, key_len, embed_size)
        # query: Query of the input (size: batch_size, query_len, embed_size)

        # keys, values can be thought of as containting "this is what I have", "this is what I can give" information
        # query is "this is what I want" information 

        batch_size = query.shape[0] # Batch size of the input

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1] # Length of the values, keys and query

        # We split the embedding into self.heads different pieces and send each piece to a different attention head
        # We do this by reshaping the embedding into (batch_size, [value/key/query]_len, heads, head_dim)
        # Basically, the embed_size dimension is now split into heads and head_dim dimensions. 

        values = values.reshape(batch_size, value_len, self.heads, self.head_dim) # Spilt values
        keys = keys.reshape(batch_size, key_len, self.heads, self.head_dim) # Spilt keys
        queries = query.reshape(batch_size, query_len, self.heads, self.head_dim) # Spilt queries

        # We send the splitted embeddings through the linear layers
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # We multiply the query, and keys and softmax the result to get the attention weights; basically for a query, which key should it pay attention to. Then we multiply the attention weights with the values to get the output.

        # Multiply queries and keys 
        #   query size: (batch_size, query_len, heads, head_dim)
        #   key size: (batch_size, key_len, heads, head_dim)
        #   QK size: (batch_size, heads, query_len, key_len), i.e. for each head, we have a query_len * key_len matrix of dot 
        #                                                          product between query & key vectors, which gives the 
        #                                                          attention weights for each query and key: which query 
        #                                                          should pay attention to which key.
        QK = torch.einsum("bqhd,bkhd->bhqk", [queries, keys]) 

        # Mask: if we are using a mask, we will mask out the tokens which are 0s so that the model doesn't pay attention to them
        if mask is not None:
            QK = QK.masked_fill(mask == 0, float("-1e20"))

        # Getting the attention weights by applying softmax on QK:
        attention = torch.softmax(QK / (self.embed_size ** (1/2)), dim = 3) 
        # We divide by the square root of the embedding size because we want to scale the values down (for numerical stability). We do this because we are using the dot product between the query and key vectors to get the attention weights. The dot product between two vectors is equal to the product of their magnitudes multiplied by the cosine of the angle between them. So, if the magnitudes of the vectors are large, then the dot product will be large, and the softmax function will saturate, which will make the gradients very small, which will make the model converge very slowly. So, we scale the values down by dividing by the square root of the embedding size.
        # We apply softmax on the last dimension because we want to normalize across the key_len dimension, i.e. for each query, we want to normalize the attention weights across all the keys, such that for a given query, the attention weights for all the keys sum to 1.

        # Multiply the attention weights with the values to get the output
        #   attention size: (batch_size, heads, query_len, key_len)
        #   values size: (batch_size, value_len, heads, head_dim)
        #   Here key_len and value_len are always going to be the same; we multiply across that dimension to get the output
        #   output size: (batch_size, query_len, heads, head_dim) i.e. for each query, we get a vector of size head_dim from each head 

        out = torch.einsum("bhql,blhd->bqhd", [attention, values])

        # Concatenate the outputs of the different heads to get the final output:
        out = out.reshape(batch_size, query_len, self.heads * self.head_dim)

        # Finally, we pass the output through a linear layer to get the final output
        out = self.fc_out(out) # out size: (batch_size, query_len, embed_size). fc_out doesn't change the size of the input

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        """
        The transformer block is:
            1. Multi-head attention (self.attention)
            2. Add & Norm (self.norm1)
            3. Feed forward (self.feed_forward)
            4. Add & Norm (self.norm2)
        """

        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size) # Layer normalization (normalization across the channels). Layer norm is like batch norm, but instead of normalizing across the batch dimension, it normalizes across the channels dimension. Each channel is a feature map. So, layer norm normalizes across the feature maps (of size embed_size).
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        # The feed forward network is a simple 2 layer network with ReLU activation in between. 
        # The hidden layer has size forward_expansion * embed_size. 
        # The output layer has size embed_size.
        # The feed_forward network doesn't change the size of the input; 
        # forward_expansion is a hyperparameter that controls the size of the hidden layer conditioned on the size of the input, and is usually set to 4.

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask) # Get the attention weights
        x = self.norm1(attention + query) # Add the query to the attention weights and normalize. Addition is for the residual connection.
        x = self.dropout(x) # Dropout
        forward = self.feed_forward(x) # Pass the output through the feed forward network
        out = self.norm2(forward + x) # Add the output of the feed forward network to the output of the attention layer and normalize. Addition is for the residual connection.
        out = self.dropout(out) # Dropout
        return out


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self. device = device 
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size) # Maps each token in the input to a vector of size embed_size; since there are src_vocab_size tokens, the embedding matrix will have size (src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size) # Maps each position in the input to a vector of size embed_size; since there are max_length positions, the embedding matrix will have size (max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size = embed_size, 
                                 heads = heads, 
                                 dropout = dropout, 
                                 forward_expansion = forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        batch_size, seq_length = x.shape # Get the batch size and sequence length of the input

        # Get the positions for each token in the input
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(self.device) # positions size: (batch_size, seq_length) so that for each input in the batch, we have the positions of all the tokens in that input

        out = self.word_embedding(x) + self.position_embedding(positions) # Add the word embeddings and the position embeddings to get the input embeddings. out size: (batch_size, seq_length, embed_size)
        out = self.dropout(out) # Dropout

        for layer in self.layers:
            out = layer(out, out, out, mask) # Since we are in the Encoder, the value, key and query are all the same (out).
        
        return out


class DecoderBlock(nn.Module):
    def __init__(self, target_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, target_mask):
        # src_mask is the mask for the source input; If the source input is padded, we don't want to pay attention to the padded tokens, so we mask them out.
        # target_mask is the mask for the target input; since the target can only attend to the previous tokens, we mask out the tokens that are ahead of the current token.

        # x is the input to the DecoderBlock, which serves as the query. 
        # value and key are the outputs of the encoder, which serve as the value and key respectively for the attention layer.

        # The DecoderBlock is essentially:
        # 1. The self-attention layer
        # 2. The transformer block, with cross-attention, which is the attention layer with value and key from the encoder and query from the DecoderBlock.

        attention = self.attention(x, x, x, target_mask) # Get the attention weights for the self-attention layer
        query = self.norm(attention + x) # Add the query to the attention weights and normalize. Addition is for the residual connection.
        query = self.dropout(query) # Dropout
        out = self.transformer_block(value, key, query, src_mask) # Pass the output through the transformer block, which has cross-attention
        return out


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size) # Maps each token in the input to a vector of size embed_size; since there are target_vocab_size tokens, the embedding matrix will have size (target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size) # Maps each position in the input to a vector of size embed_size; since there are max_length positions, the embedding matrix will have size (max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(target_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout) 
                for _ in range(num_layers)]
        
        )

        self.fc_out = nn.Linear(embed_size, target_vocab_size) # Maps the output of the decoder to the target vocabulary size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, target_mask):
        batch_size, seq_length = x.shape # Get the batch size and sequence length of the input
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(self.device) # positions size: (batch_size, seq_length) so that for each input in the batch, we have the positions of all the tokens in that input
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions)) # Add the word embeddings and the position embeddings to get the input embeddings. x size: (batch_size, seq_length, embed_size)

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, target_mask) # Pass the output through the DecoderBlock
        
        out = self.fc_out(x) 
        return out


class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 target_vocab_size, 
                 src_pad_idx,
                 target_pad_idx,
                 embed_size=256,
                 num_layers=6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0,
                 device="cuda",
                 max_length=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(target_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # src_mask size: (batch_size, 1, 1, src_len)
        # i.e. if the token is not a padding token, then src_mask = 1, else src_mask = 0
        return src_mask.to(self.device)
    
    def make_trg_mask(self, target):
        batch_size, target_len = target.shape
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(batch_size, 1, target_len, target_len) # target_mask size: (batch_size, 1, target_len, target_len)
        # i.e. if the target token is ahead of the current token, then target_mask = 0, else target_mask = 1
        # torch.tril returns the lower triangular part of a matrix (2D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
        # We expand the target_mask so that we have one target_mask for each input in the batch.
        return target_mask.to(self.device)
    
    def forward(self, src, target):
        src_mask = self.make_src_mask(src)
        target_mask = self.make_trg_mask(target)
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(target, enc_out, src_mask, target_mask)
        return out


if __name__ == '__main__':
    # Toy example to test the Transformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor([[1, 2, 3, 0, 0, 0], [3, 4, 5, 6, 7, 0]]).to(device)
    target = torch.tensor([[1, 2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8]]).to(device)
    src_pad_idx = 0
    target_pad_idx = 0
    src_vocab_size = 10
    target_vocab_size = 10
    model = Transformer(src_vocab_size, target_vocab_size, src_pad_idx, target_pad_idx).to(device)

    out = model(x, target[:, :-1]) # The target is shifted to the left by 1 position, since the last token is not used as input to the decoder

    print(out.shape) # (batch_size, target_len - 1, target_vocab_size)
