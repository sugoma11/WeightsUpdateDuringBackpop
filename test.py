import os
import pickle
import datetime
import itertools
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from my_modules import get_fc_network, get_dataloaders


import pickle
import os
from collections import defaultdict

# Add this function to your code
def save_experiment_results(results, config, experiment_name, save_dir="./results"):
    """
    Save experiment results to pickle file with structured organization.
    
    Args:
        results: Dictionary of experiment results
        config: Configuration parameters used
        experiment_name: Name of the experiment
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a filename based on experiment config
    dataset_name = config["dataset_name"][0]
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = f"{dataset_name}_{timestamp}.pkl"
    
    # Full structured data to save
    data_to_save = {
        "results": results,
        "config": config,
        "experiment_name": experiment_name,
        "timestamp": timestamp
    }
    
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data_to_save, f)
    
    print(f"Results saved to {filepath}")
    
    return filepath

def train_and_evaluate(model_old_w, model_new_w, 
                       train_loader, val_loader,
                       loss_fn, optimizer_old, optimizer_new,
                       device, num_epochs, experiment_name, custom_params=None):
    
    model_old_w.to(device)
    model_new_w.to(device)

    loss_fn_new = deepcopy(loss_fn)

    global_step = 0

    results = {
        "train_loss_old": [],
        "val_loss_old": [],
        "train_accuracy_old": [],
        "val_accuracy_old": [],
        "train_loss_new": [],
        "val_loss_new": [],
        "train_accuracy_new": [],
        "val_accuracy_new": [],
        "epochs": []
    }

    for epoch in range(num_epochs):
        model_old_w.train()
        model_new_w.train()

        running_loss_old = 0.0
        running_loss_new = 0.0

        correct_new = 0
        correct_old = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.flatten(1)

            optimizer_old.zero_grad()
            optimizer_new.zero_grad()

            outputs_old_w = model_old_w.forward(inputs, lr=custom_params['lr'], prob_of_using_new=0.0)
            outputs_new_w = model_new_w.forward(inputs, **custom_params)
            
            loss_old = loss_fn(outputs_old_w, targets)
            loss_old.backward()
            optimizer_old.step()

            loss_new = loss_fn_new(outputs_new_w, targets)
            loss_new.backward()
            optimizer_new.step()

            running_loss_old += loss_old.item()
            running_loss_new += loss_new.item()

            _, predicted_new = torch.max(outputs_new_w.data, 1)
            _, predicted_old = torch.max(outputs_old_w.data, 1)

            correct_new += (predicted_new == targets).sum().item()
            correct_old += (predicted_old == targets).sum().item()
            total += targets.size(0)

            global_step += 1

        train_accuracy_new = 100 * correct_new / total
        train_accuracy_old = 100 * correct_old / total

        avg_train_loss_old = running_loss_old / len(train_loader)
        avg_train_loss_new = running_loss_new / len(train_loader)

        # Evaluation phase
        model_old_w.eval()
        model_new_w.eval()

        correct_new = 0
        correct_old = 0
        total = 0

        val_loss_old = 0.0
        val_loss_new = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.flatten(1)

                outputs_new = model_new_w.forward(inputs, **custom_params)
                outputs_old = model_old_w.forward(inputs, lr=custom_params['lr'], prob_of_using_new=0.0)

                loss_new = loss_fn_new(outputs_new, targets)
                loss_old = loss_fn(outputs_old, targets)

                val_loss_new += loss_new.item()
                val_loss_old += loss_old.item()

                _, predicted_new = torch.max(outputs_new.data, 1)
                _, predicted_old = torch.max(outputs_old.data, 1)

                correct_new += (predicted_new == targets).sum().item()
                correct_old += (predicted_old == targets).sum().item()

                total += targets.size(0)

        avg_val_loss_old = val_loss_old / len(val_loader)
        avg_val_loss_new = val_loss_new / len(val_loader)


        val_accuracy_new = 100 * correct_new / total
        val_accuracy_old = 100 * correct_old / total


        print(f"[OLD] Epoch {epoch}: Train Loss: {avg_train_loss_old:.4f}, Val Loss: {avg_val_loss_old:.4f}, Accuracy: {val_accuracy_old:.2f}%")
        print(f"[NEW] Epoch {epoch}: Train Loss: {avg_train_loss_new:.4f}, Val Loss: {avg_val_loss_new:.4f}, Accuracy: {val_accuracy_new:.2f}%")

        wandb.log({
            f"{experiment_name}/train_loss_old": avg_train_loss_old,
            f"{experiment_name}/val_loss_old": avg_val_loss_old,
            f"{experiment_name}/train_accuracy_old": train_accuracy_old,
            f"{experiment_name}/val_accuracy_old": val_accuracy_old,

            f"{experiment_name}/train_loss_new": avg_train_loss_new,
            f"{experiment_name}/val_loss_new": avg_val_loss_new,
            f"{experiment_name}/train_accuracy_new": train_accuracy_new,
            f"{experiment_name}/val_accuracy_new": val_accuracy_new,

            "epoch": epoch
        })

        results["train_loss_old"].append(avg_train_loss_old)
        results["val_loss_old"].append(avg_val_loss_old)
        results["train_accuracy_old"].append(train_accuracy_old)
        results["val_accuracy_old"].append(val_accuracy_old)

        results["train_loss_new"].append(avg_train_loss_new)
        results["val_loss_new"].append(avg_val_loss_new)
        results["train_accuracy_new"].append(train_accuracy_new)
        results["val_accuracy_new"].append(val_accuracy_new)
        results["epochs"].append(epoch)

    return model_old_w, model_new_w, results

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
                
        experiment_name = f"BS:{bs}, LR:{lr}, L2Decay:{l2_lambda}, UseNewProb:{prob_new}, Ds:{dataset_name}"

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
