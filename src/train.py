import json
from os import path
import sys

import pandas as pd
from model import build_model, train_with_regularization, test, compute_metrics, train
from matplotlib import pyplot as plt
import torch
import time


def get_file_path(file_name):
    return path.join(DATA_PATH, file_name)


def run_training_session(
    A_train,
    A_test,
    model,
    train,
    test,
    optimizer_fn=torch.optim.SGD,
    num_iterations=100,
    learning_rate=1.0,
    plot_results=True,
):
    optimizer = optimizer_fn([model.U, model.V], lr=learning_rate)
    train_losses = []
    test_losses = []
    iterations = []
    for i in range(num_iterations + 1):
        loss = train(model, A_train, optimizer)
        if (i % 10 == 0) or (i == num_iterations):
            test_loss = test(model, A_test, optimizer)
            print(
                f"\r iteration {i}: train_error={loss} test_error={test_loss}", end=""
            )
            iterations.append(i)
            # store the results in the plot
            train_losses.append(loss.detach().cpu().numpy())
            test_losses.append(test_loss.detach().cpu().numpy())
    # TODO: plot the losses vs iterations
    num_subplots = 1
    if plot_results:
        fig = plt.figure()
        fig.set_size_inches(num_subplots * 10, 8)
        ax = fig.add_subplot(1, num_subplots, 1)
        ax.plot(iterations, train_losses, label="train_loss")
        ax.plot(iterations, test_losses, label="test_loss")
        ax.set_xlim([1, num_iterations])
        ax.legend()
        fig.savefig(get_file_path("losses.png"))
        # save train and test losses
        train_losses_df = pd.DataFrame(train_losses)
        test_losses_df = pd.DataFrame(test_losses)
        train_losses_df.to_csv(get_file_path("train_losses.csv"))
        test_losses_df.to_csv(get_file_path("test_losses.csv"))
        # compute metrics
        scores = compute_metrics(model, A_test)
        # save metrics to json
        metrics = {
            "train_loss": train_losses[-1].astype(float).item(),
            "test_loss": test_losses[-1].astype(float).item(),
            "mse": float(scores[0]),
            "mae": float(scores[1]),
            "r2": float(scores[2]),
            "max_error": float(scores[3]),
        }
        # print(type(metrics["max_error"]))
        json.dump(metrics, open(get_file_path("metrics.json"), "w"))


if __name__ == "__main__":
    DATA_PATH = path.abspath(sys.argv[1])

    # Load preprocessed data
    ratings_df = pd.read_csv(get_file_path("ratings_processed.csv"))
    news_df = pd.read_csv(get_file_path("news_processed.csv"))
    users_df = pd.read_csv(get_file_path("users_processed.csv"))
    # print(ratings_df.shape)
    # print(news_df.shape)
    # print(users_df.shape)

    start = time.time()
    # Build the CF model and train it.
    # model, A_train, A_test = build_model(ratings_df, news_df, users_df embedding_dim=15, init_stddev=0.05)
    model, A_train, A_test = build_model(
        ratings_df,
        users_df=users_df,
        news_df=news_df,
        embedding_dim=15,
        init_stddev=0.05,
    )
    # print(model)
    run_training_session(
        A_train,
        A_test,
        model,
        # train=train_with_regularization,
        train= train,
        test=test,
        num_iterations=2000,
        learning_rate=5.0,
        plot_results=True,
    )
    print(f"\ncompleted in {(time.time() - start)}")
    torch.save({"embedding_dict": model.embeddings}, get_file_path("embeddings.pt"))
