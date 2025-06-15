import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights, VGG11_Weights
import torch_pruning as tp
import torch.optim as optim
from tqdm import tqdm


def simple_pruning(conv, amount=0.1):
    strategy = tp.strategy.L1Strategy()
    pruning_index = strategy(conv.weight, amount=amount)
    plan = DG.get_pruning_plan(conv, tp.prune_conv_out_channel, pruning_index)
    plan.exec()

def compute_importance(layer):
    with torch.no_grad():
        activations = layer.weight * layer.weight.grad
    return torch.mean(torch.abs(activations), dim=(1,2,3))

def global_pruning(model, sensitivity, target_sparsity):
    all_filters = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            importance = compute_importance(module)
            all_filters.extend([(name, i, score) for i, score in enumerate(importance)])
    
    all_filters.sort(key=lambda x: x[2])
    
    prune_target = int(len(all_filters) * target_sparsity)
    return {name: [] for name, _, _ in all_filters[:prune_target]}


def low_rank_factor(module, rank=0.7):
    if isinstance(module, nn.Conv2d):
        weight = module.weight.data
        print('Before Low-Rank Factorization: ', weight.size())

        U, S, Vh = torch.svd(weight, some=False)
        appr_weight = torch.matmul(
            U[:, :int(rank * weight.size(1))],
            torch.matmul(torch.diag(S[:int(rank * weight.size(1))]), Vh[:int(rank * weight.size(1)), :])
        )
        module.weight.data = appr_weight
        print('After Low-Rank Factorization: ', module.weight.data.size())

def quantize_8bit(model):
    model.fuse_model()
    model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    torch.ao.quantization.prepare(model, inplace=True)
    torch.ao.quantization.convert(model, inplace=True)
    return model


def knowledge_distillation(teacher_model, student_model, train_loader, device='cuda', temperature=3.0, alpha=0.5):
    teacher_model.eval()
    student_model.train()

    optimizer = optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9)
    ce_loss = nn.CrossEntropyLoss()
    kd_loss = nn.KLDivLoss(reduction='batchmean')

    progress_bar = tqdm(train_loader, desc="Distilling", leave=True)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.long().to(device)

        with torch.no_grad():
            teacher_outputs = teacher_model(images)

        student_outputs = student_model(images)

        soft_targets = torch.nn.functional.softmax(teacher_outputs / temperature, dim=1)
        student_log_probs = torch.nn.functional.log_softmax(student_outputs / temperature, dim=1)

        loss = (1 - alpha) * ce_loss(student_outputs, labels) + \
               alpha * kd_loss(student_log_probs, soft_targets) * (temperature ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

    return student_model
