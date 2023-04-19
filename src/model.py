import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_rating_sparse_tensor(ratings_df, users_df, items_df):
    """
    Args:
      ratings_df: a pd.DataFrame with `user_id`, `news_id` and `rating` columns.
    Returns:
      A tf.SparseTensor representing the ratings matrix.
    """
    return (
        torch.sparse_coo_tensor(
            indices=ratings_df[["user_id", "news_id"]].values.T.astype(int),
            values=ratings_df["rating"].values.astype("float32"),
            size=[users_df.shape[0], items_df.shape[0]],
        )
        .coalesce()
        .to(device)
    )


def sparse_mean_square_error(true_ratings, predictions):
    """
    Args:
      true_ratings: Ground truth i.e flattened vector of ratings by the user
      predictions: Predicted ratings of the users for the items
    Returns:
      A scalar Tensor representing the MSE between the true ratings and the
        model's predictions.
    """
    # predictions = torch.sum(
    #     torch.gather(input=user_embeddings, index=sparse_ratings.indices[:, 0]) *
    #     torch.gather(input=movie_embeddings, index=sparse_ratings.indices[:, 1]),
    #     axis=1)
    loss = torch.nn.functional.mse_loss(predictions, true_ratings)
    return loss


class CFModel(torch.nn.Module):
    """Simple class that represents a collaborative filtering model"""

    def __init__(
        self, embedding_dim, user_size, item_size, init_stddev=1.0, metrics=None
    ):
        """Initializes a CFModel.
        Args:
          embedding_vars: A dictionary of tf.Variables.
          loss: A float Tensor. The loss to optimize.
          metrics: optional list of dictionaries of Tensors. The metrics in each
            dictionary will be plotted in a separate figure during training.
        """
        super().__init__()
        self._metrics = metrics

        # Initialize the embeddings using a normal distribution.
        self.U = (
            (torch.randn([user_size, embedding_dim]) * init_stddev)
            .to(device)
            .requires_grad_()
        )
        self.V = (
            (torch.randn([item_size, embedding_dim]) * init_stddev)
            .to(device)
            .requires_grad_()
        )

    @property
    def embeddings(self):
        """The embeddings dictionary."""
        return {
            "user_embed": self.U.detach().cpu().numpy(),
            "item_embed": self.V.detach().cpu().numpy(),
        }

    def forward(self, sparse_ratings):
        predictions = torch.sum(
            torch.index_select(
                input=model.U, dim=0, index=sparse_ratings.indices()[0, :]
            )
            * torch.index_select(
                input=model.V, dim=0, index=sparse_ratings.indices()[1, :]
            ),
            dim=1,
        )
        return predictions


def train(model: torch.nn.Module, ratings_mat, optimizer):
    """Trains the model.
    Args:
      model: model being trained
      rating_mat: sparse rating matrix of dense shape [N, M]
      optimizer: the optimizer to use
    Returns:
      The calculated loss during training
    """
    model.train()
    pred = model.forward(ratings_mat)
    # print(pred)
    loss = sparse_mean_square_error(ratings_mat.values(), pred)
    # print("****loss", loss)
    # Backpropagate error
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def build_model(ratings, embedding_dim=3, init_stddev=1.0):
    """
    Args:
      ratings: a DataFrame of the ratings
      embedding_dim: the dimension of the embedding vectors.
      init_stddev: float, the standard deviation of the random initial embeddings.
    Returns:
      model: a CFModel.
    """
    # Split the ratings DataFrame into train and test.
    train_ratings, test_ratings = split_dataframe(ratings)
    # SparseTensor representation of the train and test datasets.
    A_train = build_rating_sparse_tensor(train_ratings, users, movies)
    A_test = build_rating_sparse_tensor(test_ratings, users, movies)

    # metrics = {
    #     'train_error': train_loss,
    #     'test_error': test_loss
    # }
    # embeddings = {
    #     "user_id": U,
    #     "movie_id": V
    # }
    return (
        CFModel(embedding_dim, A_train.size()[0], A_train.size()[1], init_stddev).to(
            device
        ),
        A_train,
        A_test,
    )


def gravity(U, V):
    """Creates a gravity loss given two embedding matrices."""
    return (
        1.0
        / (U.shape[0] * V.shape[0])
        * torch.sum(torch.matmul(U.T, U) * torch.matmul(V.T, V))
    )


def get_square_norm(X):
    return (1.0 / X.shape[0]) * torch.sum(X * X)


def train_with_regularization(
    model: torch.nn.Module,
    ratings_mat,
    optimizer,
    regularization_coeff=0.1,
    gravity_coeff=1.0,
):
    """Trains the model.
    Args:
      model: model being trained
      rating_mat: sparse rating matrix of dense shape [N, M]
      optimizer: the optimizer to use
    Returns:
      The calculated loss during training
    """
    model.train()
    pred = model.forward(ratings_mat)
    # print(pred)
    loss = sparse_mean_square_error(ratings_mat.values(), pred)
    gravity_loss = gravity_coeff * gravity(model.U, model.V)
    regularization_loss = regularization_coeff * (
        get_square_norm(model.U) + get_square_norm(model.V)
    )
    total_loss = loss + gravity_loss + regularization_loss
    # print("****loss", loss)
    # Backpropagate error
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return loss