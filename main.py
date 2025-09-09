import os
import wandb
import torch
import hydra
import pickle
import datetime
import itertools
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf

from core.layers import get_fc_network
from core.dataloaders import get_dataloaders
from utils.train_utils import train_and_evaluate, save_experiment_results

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    cfg.device = cfg.device[0]

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    # Initialize Weights & Biases.
    # The run name will include the Hydra run id and a timestamp.
    run = wandb.init(
        project=cfg.get("wandb_project", "my_benchmark_project"),
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"benchmark_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        reinit=True  # Allows multiple runs within the same script if desired.
    )

    all_results = defaultdict(list)

    experiment_configs = list(itertools.product(cfg.batch_sizes, cfg.learning_rates, cfg.prob_of_using_new_values,
                                                cfg.num_layers, cfg.feature_pass, cfg.l2_lambda, cfg.use_bn,
                                                cfg.dataset_name, cfg.data_dims, cfg.seeds, cfg.num_epochs))

    for bs, lr, prob_new, num_layers, feature_pass, l2_lambda, use_bn, dataset_name, data_dims, seed, num_epochs in experiment_configs:

        train_loader, val_loader, test_loader = get_dataloaders(dataset_name, bs)

        input_dim, num_classes = data_dims

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

        model_old_w, model_new_w = get_fc_network(input_dim=input_dim, num_layers=num_layers,
                                                  feature_pass=feature_pass, num_classes=num_classes,
                                                  use_bn=use_bn, device=device)

        experiment_name = f"BS:{bs}, Layers:{num_layers}, LR:{lr}, L2Decay:{l2_lambda}, UseNewProb:{prob_new}, Ds:{dataset_name}"

        print(f"Running experiment: {experiment_name}")

        loss_fn = nn.CrossEntropyLoss()

        optimizer_old = optim.SGD(model_old_w.parameters(), lr=lr)
        optimizer_new = optim.SGD(model_new_w.parameters(), lr=lr)

        wandb.config.update({
            "batch_size": bs,
            "learning_rate": lr,
            "prob_of_using_new": prob_new,
            "ds_name": dataset_name,
            "experiment_name": experiment_name,
            "l2_lambda": l2_lambda,
            "seed": seed
        }, allow_val_change=True)

        _, _, results = train_and_evaluate(
            model_old_w=model_old_w, model_new_w=model_new_w,
            train_loader=train_loader, val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer_old=optimizer_old,
            optimizer_new=optimizer_new,
            device=device, num_epochs=num_epochs,
            experiment_name=experiment_name,
            custom_params={'lr': lr, 'prob_of_using_new': prob_new}
        )

        key = f"{dataset_name}_layers{num_layers}_bs{bs}_seed{seed}"

        all_results[key].append({
            "results": results,
            "config": cfg,
            "experiment_name": experiment_name
        })

        # Save individual experiment results
        save_experiment_results(
            results,
            cfg,
            experiment_name,
            save_dir=os.path.join(cfg.get("results_dir", "./results"), dataset_name))


    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    all_results_path = os.path.join(cfg.get("results_dir", "./results"), f"all_experiments_{timestamp}.pkl")
    with open(all_results_path, "wb") as f:
        pickle.dump(dict(all_results), f)

    run.finish()

main()
