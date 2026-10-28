import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from math import floor, ceil
from stan import stan_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, average_precision_score


def to_labels(logits: torch.Tensor) -> list:
    # Trả về nhãn cứng (0 hoặc 1) để tính F1-Score
    with torch.no_grad():
        pred = F.softmax(logits, dim=1).cpu()
        pred = pred.argmax(dim=1)
    return pred.numpy().tolist()

def to_probs(logits: torch.Tensor) -> list:
    # Lấy xác suất của class 1 (dành cho AUC, AP)
    probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist()
    return probs

def att_train(
    x_train,
    y_train,
    x_test,
    y_test,
    num_classes: int = 2,
    epochs: int = 18,
    batch_size: int = 256,
    attention_hidden_dim: int = 150,
    lr: float = 3e-3,
    device: str = "cpu"
):
    model = stan_model(
        time_windows_dim=x_train.shape[1],
        spatio_windows_dim=x_train.shape[2],
        feat_dim=x_train.shape[3],
        num_classes=num_classes,
        attention_hidden_dim=attention_hidden_dim,
        mamba_d_model=128,
    )
    model.to(device)

    nume_feats = x_train
    labels = y_train

    nume_feats.requires_grad = False
    labels.requires_grad = False

    nume_feats.to(device)
    labels = labels.to(device)

    dummy_input = nume_feats[:2]
    _ = model(dummy_input)

    # Tính toán trọng số phạt (Class Weights) cho hàm loss. Phạt nặng lớp thiểu số
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    class_counts[class_counts == 0] = 1.0 # Tránh lỗi chia cho 0
    weights = (1.0 / class_counts) * len(labels) / num_classes
    weights = weights.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Sử dụng Weighted Cross-Entropy Loss
    loss_func = torch.nn.CrossEntropyLoss(weight=weights)

    batch_num = ceil(len(labels) / batch_size)
    for epoch in range(epochs):
        model.train()
        loss = 0.
        pred_labels = []
        pred_probs = []

        for batch in (range(batch_num)):
            optimizer.zero_grad()
            batch_mask = list(range(batch*batch_size, min((batch+1)*batch_size, len(labels))))

            output = model(nume_feats[batch_mask])
            batch_loss = loss_func(output, labels[batch_mask])
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            pred_labels.extend(to_labels(output))
            pred_probs.extend(to_probs(output))

        true = labels.cpu().numpy()
        pred_labels = np.array(pred_labels)
        pred_probs = np.array(pred_probs)

        print(f"Epoch: {epoch}, loss: {(loss / batch_num):.4f}, "
              f"auc: {roc_auc_score(true, pred_probs):.4f}, "
              f"F1: {f1_score(true, pred_labels, average='macro'):.4f}, "
              f"AP: {average_precision_score(true, pred_probs):.4f}")


# Test Set
    model.eval()
    feats_test = x_test
    labels_test = y_test
    batch_num_test = ceil(len(labels_test) / batch_size)

    with torch.no_grad():
        pred_labels = []
        pred_probs = []
        for batch in range(batch_num_test):
            batch_mask = list(range(batch * batch_size, min((batch + 1) * batch_size, len(labels_test))))
            output = model(feats_test[batch_mask])

            pred_labels.extend(to_labels(output))
            pred_probs.extend(to_probs(output))

        true = labels_test.cpu().numpy()
        pred_labels = np.array(pred_labels)
        pred_probs = np.array(pred_probs)

        print(f"test set | auc: {roc_auc_score(true, pred_probs):.4f}, "
              f"F1: {f1_score(true, pred_labels, average='macro'):.4f}, "
              f"AP: {average_precision_score(true, pred_probs):.4f}")


def stan_main(
        X_train, y_train, X_test, y_test,  # Nhận trực tiếp mảng Numpy
        epochs=20,
        batch_size=256,
        attention_hidden_dim=150,
        lr=0.003,
        device="cpu",
):
    # Chuyển đổi dữ liệu trực tiếp sang Tensor
    train_feature = torch.from_numpy(X_train).float().to(device)
    train_label = torch.from_numpy(y_train).long().to(device)

    test_feature = torch.from_numpy(X_test).float().to(device)
    test_label = torch.from_numpy(y_test).long().to(device)

    att_train(
        train_feature,
        train_label,
        test_feature,
        test_label,
        epochs=epochs,
        batch_size=batch_size,
        attention_hidden_dim=attention_hidden_dim,
        lr=lr,
        device=device
    )