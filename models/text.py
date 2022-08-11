from torch import nn


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, num_class=4, embedding=None, freeze=False):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        if num_class == 2:
            num_class = 1
        self.linear = nn.Linear(embed_dim, num_class)
        self.init_weights(embedding)
        if freeze:
            self.embedding.weight.requires_grad = False

    def init_weights(self, embedding):
        initrange = 0.5
        if embedding is None:
            self.embedding.weight.data.uniform_(-initrange, initrange)
        else:
            self.embedding.weight.data = embedding
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()

    def forward(self, text_and_offsets):
        embedded = self.embedding(text_and_offsets[0], text_and_offsets[1])
        return self.linear(embedded),  embedded
