import copy
import os.path as osp

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from utils import debug


def evaluate_loss(model, loss_function, num_batches, data_iter, cuda=False):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        return np.mean(_loss).item(), f1_score(all_targets, all_predictions) * 100
    pass


def fpr_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=np.int32), np.array(y_pred, dtype=np.int32)
    TP = sum((y_true == 1) & (y_pred == 1))
    FP = sum((y_true == 0) & (y_pred == 1))
    TN = sum((y_true == 0) & (y_pred == 0))
    FN = sum((y_true == 1) & (y_pred == 0))

    if (FP + TN) == 0:
        return 0.
    else:
        FPR = FP / (FP + TN)
        return FPR


def evaluate_metrics(model, loss_function, num_batches, data_iter, csv_path):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().astype(np.int32).tolist())
        model.train()

        results_df = pd.DataFrame({'Label': all_targets, 'Prediction': all_predictions})
        results_df.to_csv(csv_path, index=False)

        return accuracy_score(all_targets, all_predictions) * 100, \
               precision_score(all_targets, all_predictions) * 100, \
               recall_score(all_targets, all_predictions) * 100, \
               f1_score(all_targets, all_predictions) * 100, \
               fpr_score(all_targets, all_predictions) * 100
    pass


def train(model, dataset, max_steps, loss_function, optimizer, model_dir, dev_every=128, log_every=50, max_patience=5):
    debug('Start Training')
    train_losses = []
    best_model = None
    patience_counter = 0
    best_f1 = 0
    try:
        for step_count in range(max_steps):
            model.train()
            model.zero_grad()
            graph, targets = dataset.get_next_train_batch()
            #graph = graph.to("cuda:0")
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            if log_every is not None and (step_count % log_every == log_every - 1):
                debug('Step %d\t\tTrain Loss %10.3f' % (step_count, batch_loss.detach().cpu().item()))
            train_losses.append(batch_loss.detach().cpu().item())
            batch_loss.backward()
            optimizer.step()
            if step_count % dev_every == (dev_every - 1):
                valid_loss, valid_f1 = evaluate_loss(model, loss_function, dataset.initialize_valid_batch(),
                                                     dataset.get_next_valid_batch)
                if valid_f1 > best_f1:
                    patience_counter = 0
                    best_f1 = valid_f1
                    best_model = copy.deepcopy(model.state_dict())
                    _save_file = open(osp.join(model_dir, 'model.bin'), 'wb')
                    torch.save(model.state_dict(), _save_file)
                    _save_file.close()
                else:
                    patience_counter += 1
                debug('Step %d\t\tTrain Loss %10.3f\tValid Loss%10.3f\tf1: %5.2f\tPatience %d' % (
                    step_count, np.mean(train_losses).item(), valid_loss, valid_f1, patience_counter))
                debug('=' * 100)
                train_losses = []
                if patience_counter == max_patience:
                    break
    except KeyboardInterrupt:
        debug('Training Interrupted by user!')
        raise

    if best_model is not None:
        model.load_state_dict(best_model)
    _save_file = open(osp.join(model_dir, 'model.bin'), 'wb')
    torch.save(model.state_dict(), _save_file)
    _save_file.close()

    return model
