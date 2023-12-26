import torch
from torch import nn
from einops import rearrange

import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import networkx as nx
from bs4 import BeautifulSoup

class ImgToEmbeddings(nn.Module):
    """ImgToEmbeddings

    Args:
        patches (int): Number of patches to divide the image into
        patch_size (int): Size of the patches
        transformer_dim (int): Dimension of the transformer
        img_channels (int): Number of channels in the image
        seq_len (int): Length of the sequence
        reduced_dim (int): Dimension of the reduced embedding

    Returns:
        torch.Tensor: The output of the model

    Input shape:
        (batch, channels, height, width)

    Output shape:
        (batch, seq_len, reduced_dim)

    Example:
        >>> import torch
        >>> from geminix import ImgToEmbeddings
        >>> model = ImgToEmbeddings(
        ...     patches=16,
        ...     patch_size=16,
        ...     transformer_dim=512,
        ...     img_channels=3,
        ...     seq_len=128,
        ...     reduced_dim=128
        ... )
        >>> x = torch.randn(1, 3, 256, 256)
        >>> y = model(x)
        >>> y.shape
        torch.Size([1, 128, 128])
    """

    def __init__(
        self,
        patches: int,
        patch_size: int,
        transformer_dim: int,
        img_channels: int,
        seq_len: int,
        reduced_dim: int,
        *args,
        **kwargs,
    ):
        super(ImgToEmbeddings, self).__init__()
        self.patches = patches
        self.patch_size = patch_size
        self.transformer_dim = transformer_dim
        self.img_channels = img_channels
        self.seq_len = seq_len
        self.reduced_dim = reduced_dim

        # Img is a square, cal number of apthces
        self.num_patches_side = int(patches**0.5)

        # Patch embedding layer
        self.patch_embedding = nn.Linear(
            patch_size * patch_size * img_channels, transformer_dim
        )

        # Dim reduction
        self.dim_reduction = nn.Linear(transformer_dim, reduced_dim)

        # Batch Norm and relu
        self.norm = nn.BatchNorm1d(patches)
        self.activate = nn.ReLU()

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, patches, reduced_dim))

        # Token mixing
        self.token_mixer = nn.Linear(patches * reduced_dim, patches * reduced_dim)

        # Linear layer to expand the seq to vocab
        self.seq_expansion = nn.Linear(patches * reduced_dim, seq_len * reduced_dim)

    def forward(self, x: torch.Tensor):
        """Forward pass

        Args:
            x (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        batch, channels, height, width, height = x.shape

        # Check if img can be evenly divided into patches
        assert (
            height % self.num_patches_side == 0 and width % self.num_patches_side == 0
        ), "Image dimensions must be divisivle by the square root of patches"

        # Reshpe the img to patches
        x = x.unfold(
            2,
            self.patch_size,
        ).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch, channels, self.num_patches, -1)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, self.num_patches, -1)

        # Apply patch embedding
        x = self.patch_embedding(x)

        # Dim reduction
        x = self.dim_reduction(x)

        # Batch norm
        x = self.norm(x)
        x = self.activate(x)

        # Add positional encoding
        x = x.view(batch, -1)
        x = self.token_mixer(x)
        x = x.view(batch, self.num_patches, -1)

        # Expand the seq to match vocab
        x = self.seq_expansion(x)
        x = x.view(batch, self.seq_len, -1)

        return x
class TreeEmbedding(nn.Module):
    """TreeEmbedding for HTML DOM Tree

    Args:
        dim (int): Dimension of the embedding
        vocab_size (int): Size of the vocabulary

    Returns:
        torch.Tensor: The output of the model

    Input shape:
        (batch, seq_len)  # assuming each sequence is a list of HTML tokens

    Output shape:
        (batch, seq_len, dim)

    Example:
        >>> import torch
        >>> from geminix import TreeEmbedding
        >>> model = TreeEmbedding(dim=128, vocab_size=256)
        >>> x = [["<html>", "<body>", "<p>", "text", "</p>", "</body>", "</html>"]]
        >>> offsets = torch.tensor([0, 7])
        >>> y = model(x, offsets)
        >>> y.shape
        torch.Size([1, 7, 128])
    """

    def __init__(self, dim: int, vocab_size: int, *args, **kwargs):
        super(TreeEmbedding, self).__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.embedding = nn.EmbeddingBag(vocab_size, dim, sparse=True)

    def forward(self, sequences, offsets):
        """Forward pass

        Args:
            sequences (List[List[str]]): List of lists representing sequences of HTML tokens
            offsets (torch.Tensor): Offsets for each sequence in the batch

        Returns:
            torch.Tensor: Embedded representations of the HTML tokens
        """
        token_indices_list = [
            torch.tensor([max(0, min(hash(token), self.vocab_size - 1)) for token in sequence])
            for sequence in sequences
        ]

        sequence_lengths = [len(token_indices) for token_indices in token_indices_list]

        token_indices_padded = nn.utils.rnn.pad_sequence(token_indices_list, batch_first=True, padding_value=0)

        return self.embedding(token_indices_padded.flatten(), offsets).view(
            token_indices_padded.size(0), token_indices_padded.size(1), -1
        )

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.6):
        super(GraphAttentionLayer, self).__init__()
        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.gat_conv(x, edge_index)
        return x

class GraphEmbeddingModel(nn.Module):
    def __init__(self, input_dim, hidden_channels, out_channels):
        super(GraphEmbeddingModel, self).__init__()
        self.gat1 = GraphAttentionLayer(input_dim, hidden_channels)
        self.gat2 = GraphAttentionLayer(hidden_channels * self.gat1.gat_conv.heads, out_channels)

    def forward(self, node_features, edge_index):
        h = self.gat1(node_features, edge_index)
        h = nn.functional.elu(h)
        h = self.gat2(h, edge_index)
        return torch.mean(h, dim=0)

class HtmlToGraphEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_channels=128, out_channels=64):
        super(HtmlToGraphEmbedding, self).__init__()
        self.graph_embedding_model = GraphEmbeddingModel(input_dim, hidden_channels, out_channels)

    def forward(self, html):
        graph = self.construct_graph_from_html(html)
        graph_embedding = self.generate_graph_embedding(graph)
        return graph_embedding

    def construct_graph_from_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        return self.construct_graph_from_tree(soup.body)

    def construct_graph_from_tree(self, tree):
        graph = nx.DiGraph()
        mapping = {}

        def add_edges_recursive(parent_id, node):
            nonlocal graph
            nonlocal mapping

            if isinstance(node, str):
                pass
            else:
                for child in node.children:
                    child_id = mapping.setdefault(id(child), len(mapping))
                    graph.add_edge(parent_id, child_id)
                    add_edges_recursive(child_id, child)

        root_id = mapping.setdefault(id(tree), len(mapping))
        add_edges_recursive(root_id, tree)
        return graph

    def generate_graph_embedding(self, graph):
        num_nodes = len(graph.nodes())
        node_features = torch.eye(num_nodes)
        edge_index = torch.tensor(list(graph.edges)).t().contiguous()

        return self.graph_embedding_model(torch.tensor(node_features) if node_features is not None else torch.tensor(edge_index),
                                           edge_index)
class AudioToEmbeddings(nn.Module):
    """AudioToEmbeddings

    Args:
        audio_seq_len (int): Length of the audio sequence
        seqlen (int): Length of the sequence
        dim (int): Embedding dimension

    Example:
        >>> import torch
        >>> from geminix import AudioToEmbeddings
        >>> model = AudioToEmbeddings(
        ...     audio_seq_len=32000,
        ...     seqlen=512,
        ...     dim=512
        ... )
        >>> x = torch.randn(1, 32000)
        >>> y = model(x)
        >>> y.shape
        torch.Size([1, 512, 512])
    """

    def __init__(self, audio_seq_len, seqlen, dim):
        super(AudioToEmbeddings, self).__init__()
        self.audio_seq_len = audio_seq_len
        self.seqlen = seqlen
        self.dim = dim
        # Initialize a linear layer to project the 2D audio input to the desired 3D shape
        self.projection = nn.Linear(audio_seq_len, seqlen * dim)

    def forward(self, x):
        """Forward pass

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # x shape: [batch, audio_seq_len] - 2D input
        batch, audio_seq_len = x.shape

        # Project the audio tensor to match the seqlen and dim
        x = self.projection(x)  # x shape: [batch, seqlen * dim]

        # Reshape to the target shape: [batch, seqlen, dim]
        x = rearrange(x, "b (s d) -> b s d", s=self.seqlen, d=self.dim)

        return x


# # Example usage
# audio_seq_len = 32000  # Input audio sequence length
# seqlen = 512  # Sequence length to align with the language transformer
# dim = 512  # Embedding dimension

# model = AudioToEmbeddings(audio_seq_len, seqlen, dim)
# audio_input = torch.randn(1, audio_seq_len)  # Example input tensor
# output = model(audio_input)

# print("Output shape:", output.shape)  # Should be [1, 512, 512]
