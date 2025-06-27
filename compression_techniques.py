import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights
import torch_pruning as tp
import torch.ao.quantization as quantization
import torch.optim as optim
from tqdm import tqdm
import tltorch as tlt
import copy

# After several model changes and updates to main, it is quite possible that some of these functions currently dont work
# The only currently guaranteed to work is knowledge_distillation
# global pruning will most likely work as well but needs some updates

# def simple_pruning(conv, amount=0.1): # Not applicable for CNN
#     strategy = tp.strategy.L1Strategy()
#     pruning_index = strategy(conv.weight, amount=amount)
#     plan = DG.get_pruning_plan(conv, tp.prune_conv_out_channel, pruning_index)
#     plan.exec()

def compute_importance(layer):
    with torch.no_grad():
        activations = layer.weight * layer.weight.grad
    return torch.mean(torch.abs(activations), dim=(1,2,3))

def global_structured_pruning(model, device, target_sparsity=0.3, input_size=(224, 224)):
    all_filters = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            importance = compute_importance(module)
            all_filters.extend([(name, i, score) for i, score in enumerate(importance)])
    
    all_filters.sort(key=lambda x: x[2])
    prune_target = int(len(all_filters) * target_sparsity)
    print()
    
    filters_to_prune = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            filters_to_prune[name] = set()
    for name, idx, _ in all_filters[:prune_target]:
        filters_to_prune[name].add(idx)
    
    prev_pruned = None
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            keep_indices = [i for i in range(module.out_channels) 
                           if i not in filters_to_prune[name]]
            
            new_conv = nn.Conv2d(
                in_channels=module.in_channels if prev_pruned is None else len(prev_pruned),
                out_channels=len(keep_indices),
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None
            )
            
            new_conv.weight.data = module.weight.data[keep_indices]
            if module.bias is not None:
                new_conv.bias.data = module.bias.data[keep_indices]
            
            if prev_pruned is not None:
                new_conv.weight.data = new_conv.weight.data[:, prev_pruned]
            
            parent = model
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, new_conv)
            else:
                setattr(model, name, new_conv)
            
            prev_pruned = keep_indices
 
    
    if hasattr(model, 'classifier') and prev_pruned is not None:
        first_linear = model.classifier[0]
        
        spatial_size = 7 * 7
        
        expected_original_channels = first_linear.in_features // spatial_size
        
        new_in_features = len(prev_pruned) * spatial_size
        
        if first_linear.in_features % expected_original_channels != 0:
            raise ValueError("Original weight dimensions are not divisible by expected channels")
        
        new_linear = nn.Linear(
            in_features=new_in_features,
            out_features=first_linear.out_features,
            bias=first_linear.bias is not None
        ).to(device)
        
        orig_weight = first_linear.weight.data
        
        if expected_original_channels == len(prev_pruned):
            new_linear.weight.data = orig_weight
        else:
            with torch.no_grad():
                reshaped = orig_weight.view(
                    first_linear.out_features,
                    expected_original_channels,
                    spatial_size
                )
                
                mask = torch.zeros(expected_original_channels, dtype=torch.bool)
                mask[prev_pruned] = True
                
                new_weight = reshaped[:, mask].contiguous()
                new_weight = new_weight.view(first_linear.out_features, -1)
                
                new_linear.weight.data = new_weight
        
        if first_linear.bias is not None:
            new_linear.bias.data = first_linear.bias.data
            
        model.classifier[0] = new_linear
    
    return model

def low_rank_factor(model, rank=0.3, factorization='tucker', device='cpu'):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            fact_conv = tlt.FactorizedConv.from_conv(
                module,
                rank=rank,
                decompose_weights=True,
                factorization=factorization
            ).to(device)
            setattr(model, name, fact_conv)
        else:
            low_rank_factor(module, rank, factorization, device)
    return model

def fuse_vgg_model(model):
    for module_name, module in model.named_children():
        if module_name == "features":
            for i in range(len(module)):
                if isinstance(module[i], torch.nn.Conv2d) and \
                   i + 1 < len(module) and isinstance(module[i + 1], torch.nn.ReLU):
                    quantization.fuse_modules(module, [str(i), str(i + 1)], inplace=True)
        elif module_name == "classifier":
            for i in range(len(module)):
                if isinstance(module[i], torch.nn.Linear) and \
                   i + 1 < len(module) and isinstance(module[i + 1], torch.nn.ReLU):
                    quantization.fuse_modules(module, [str(i), str(i + 1)], inplace=True)

def quantize_8bit(model, device, calibration_loader=None):
    model.eval()
    device = "cpu"
    fuse_vgg_model(model)

    per_tensor_qconfig = quantization.QConfig(
        activation=quantization.FakeQuantize.with_args(observer=quantization.MinMaxObserver, qscheme=torch.per_tensor_affine),
        weight=quantization.default_weight_fake_quant
    )
    model.qconfig = per_tensor_qconfig

    quantization.prepare(model, inplace=True)

    model.to(device)

    if calibration_loader is not None:
        with torch.no_grad():
            for images, _ in calibration_loader:
                images = images.to(device)
                model(images)
    else:
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            model(dummy_input)

    quantization.convert(model, inplace=True)
    return model


def knowledge_distillation(teacher_model, student_model, train_loader, val_loader, 
                          device='cuda', temperature=4.0, alpha=0.5, epochs=50,
                          lr=0.001, momentum=0.7):

    teacher_model.eval()
    optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    ce_loss = nn.CrossEntropyLoss()
    kd_loss = nn.KLDivLoss(reduction='batchmean')
    
    best_val_acc = 0.0
    early_stop_counter = 0
    patience = 5

    for epoch in range(epochs):
        student_model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.long().to(device)
            
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            
            student_outputs = student_model(images)
            
            soft_targets = torch.softmax(teacher_outputs / temperature, dim=1)
            student_log_probs = torch.log_softmax(student_outputs / temperature, dim=1)
            
            loss = (1 - alpha) * ce_loss(student_outputs, labels) + \
                   alpha * kd_loss(student_log_probs, soft_targets) * (temperature ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        student_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = student_model(images)
                loss = ce_loss(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = f"models/mobilenet_v3_l_dist_from_vgg19_epoch_{epoch+1}.pth"
            torch.save(student_model, best_model)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            student_model.load_state_dict(torch.load(best_model))
            break
    
    return student_model

