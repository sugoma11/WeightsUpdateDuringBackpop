import os
import torch
import wandb
import pickle
import datetime
from copy import deepcopy

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
