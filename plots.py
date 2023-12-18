import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pandas as pd
import yaml


def load_data(directory: str):
    """
    Loads the data from the given directory.
    """
    data = []
    params = {
        "BERT-base": 110,
        "BERT-large": 336,
        "DistilBERT": 67,
        "MiniLM-L6": 22.7,
        "MiniLM-L12": 33.4,
    }
    for filename, params in params.items():
        with open(os.path.join(directory, f"{filename}.json")) as f:
            model_output = json.load(f)
            model_output["model_name"] = filename
            model_output["params"] = params
        with open(os.path.join(directory, f"{filename}.config.yaml")) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            model_output["lr"] = config["lr"]["value"]
            model_output["batch_size"] = config["batch_size"]["value"]
            model_output["num_steps"] = config["max_steps"]["value"]
            model_output["threshold"] = config["threshold"]["value"]
            model_output["mapped_threshold"] = config["mapped_threshold"]["value"]

        data.append(model_output)

    return pd.DataFrame.from_records(data)


if __name__ == "__main__":
    # Load the data
    data = load_data("models/model_outputs")

    # Plot the data
    with plt.style.context("seaborn-v0_8"):
        fig, ax = plt.subplots(layout="constrained", figsize=(42 // 2, 28 // 2))
        font = {'size': 32}

        plt.rc('font', **font)
        #fig.tight_layout()

        min_f1, max_f1 = data["test_mapped_f1"].min(), data["test_mapped_f1"].max()
        min_params, max_params = data["params"].min(), data["params"].max()

        X = data["params"].values.reshape(-1, 1)
        y = data["test_mapped_f1"].values.reshape(-1, 1)

        #plt.xlim((min_params - 20, max_params + 100))
        #plt.ylim(min_f1 - 0.05, max_f1 + 0.05)

        fig.suptitle("Model Size VS F1-Score", fontsize=56)

        # Change ticks
        plt.yticks(np.arange(0.54, 0.6 + 0.005, 0.005), fontsize=32)
        xticks = np.arange(25, np.ceil(max_params) + 25, 25)
        plt.ylabel("F1-Score", fontsize=32)

        plt.xticks(xticks, rotation=-45, fontsize=32)
        ax.set_xticklabels([f"{v}M" for v in xticks])
        plt.xlabel("No. params", fontsize=32)

        ax.scatter(data["params"], data["test_mapped_f1"], label="F1", s=200, c="red")

        text_positions = {
            "BERT-base": {"ha": "left"},
            "BERT-large": {"ha": "right"},
            "DistilBERT": {"ha": "left"},
            "MiniLM-L6": {"ha": "left"},
            "MiniLM-L12": {"ha": "left"},
        }

        # Add Model name to each dot
        for i in range(len(data)):
            row = data.iloc[i]
            ax.annotate(f"{row['model_name']} ({row['params']}M)", (row["params"], row["test_mapped_f1"]), fontsize=45, **text_positions[row["model_name"]])

    fig.savefig("model-vs-f1.png", dpi=600)
