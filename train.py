import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
import logging

# Import the model components
from mcran import MCRAN, preprocess_dna_for_mcran
from birc_mamba import BiRCMamba, dna_to_onehot
from mffs import MFFS, MDNetDBP

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mdnet_dbp_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DNABindingDataset(Dataset):
    """Dataset for DNA-protein binding site prediction"""

    def __init__(self, dna_sequences, labels):
        """
        Args:
            dna_sequences: List of DNA sequences (strings)
            labels: List of binding labels (0 or 1)
        """
        self.dna_sequences = dna_sequences
        self.labels = labels

    def __len__(self):
        return len(self.dna_sequences)

    def __getitem__(self, idx):
        sequence = self.dna_sequences[idx]
        label = self.labels[idx]

        # Prepare inputs for both models
        sce_encoded = np.array(coden1(sequence))
        one_hot = dna_to_onehot(sequence).squeeze(0).numpy()  # Remove batch dimension

        return {
            'sequence': sequence,
            'sce_encoded': sce_encoded,
            'one_hot': one_hot,
            'label': label
        }


def collate_fn(batch):
    """Custom collate function to handle different encodings"""
    sequences = [item['sequence'] for item in batch]
    sce_encoded = torch.FloatTensor([item['sce_encoded'] for item in batch])
    one_hot = torch.FloatTensor([item['one_hot'] for item in batch])
    labels = torch.FloatTensor([item['label'] for item in batch]).unsqueeze(1)

    return {
        'sequences': sequences,
        'sce_encoded': sce_encoded,
        'one_hot': one_hot,
        'labels': labels
    }


def evaluate_model(model, dataloader, device):
    """Evaluate model performance with comprehensive metrics"""
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            sce_encoded = batch['sce_encoded'].to(device)
            one_hot = batch['one_hot'].to(device)
            labels = batch['labels'].to(device)

            if isinstance(model, MDNetDBP):
                outputs, _ = model(sce_encoded, one_hot)
            elif isinstance(model, MCRAN):
                outputs = model(sce_encoded)
            elif isinstance(model, BiRCMamba):
                # Process features and get predictions
                features = model(one_hot)
                # Apply global pooling
                global_avg = torch.mean(features, dim=1)
                global_max, _ = torch.max(features, dim=1)
                combined = torch.cat([global_avg, global_max], dim=1)
                # Apply final classifier
                outputs = torch.sigmoid(model.fc(combined))

            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds)
            all_probs.extend(probs)

    # Calculate comprehensive metrics
    all_labels = np.array(all_labels).flatten()
    all_predictions = np.array(all_predictions).flatten()
    all_probs = np.array(all_probs).flatten()

    # Calculate all metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'pr_auc': pr_auc
    }


def train_model(model, train_loader, val_loader, optimizer, criterion, device,
                num_epochs, patience=10, scheduler=None):
    """Train model with early stopping"""

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            sce_encoded = batch['sce_encoded'].to(device)
            one_hot = batch['one_hot'].to(device)
            labels = batch['labels'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            if isinstance(model, MDNetDBP):
                outputs, _ = model(sce_encoded, one_hot)
            elif isinstance(model, MCRAN):
                outputs = model(sce_encoded)
            elif isinstance(model, BiRCMamba):
                # Process features and get predictions
                features = model(one_hot)
                # Apply global pooling
                global_avg = torch.mean(features, dim=1)
                global_max, _ = torch.max(features, dim=1)
                combined = torch.cat([global_avg, global_max], dim=1)
                # Apply final classifier (assuming there's a fc layer in BiRCMamba)
                outputs = torch.sigmoid(model.fc(combined))

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Calculate average training loss for this epoch
        avg_train_loss = sum(train_losses) / len(train_losses)
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                sce_encoded = batch['sce_encoded'].to(device)
                one_hot = batch['one_hot'].to(device)
                labels = batch['labels'].to(device)

                if isinstance(model, MDNetDBP):
                    outputs, _ = model(sce_encoded, one_hot)
                elif isinstance(model, MCRAN):
                    outputs = model(sce_encoded)
                elif isinstance(model, BiRCMamba):
                    features = model(one_hot)
                    global_avg = torch.mean(features, dim=1)
                    global_max, _ = torch.max(features, dim=1)
                    combined = torch.cat([global_avg, global_max], dim=1)
                    outputs = torch.sigmoid(model.fc(combined))

                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

        # Calculate average validation loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        history['val_loss'].append(avg_val_loss)

        # Evaluate model
        val_metrics = evaluate_model(model, val_loader, device)
        history['val_metrics'].append(val_metrics)

        # Step the learning rate scheduler if it exists
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Log progress with all metrics
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"Val ACC: {val_metrics['accuracy']:.4f}, "
                    f"Val AUC: {val_metrics['auc']:.4f}, "
                    f"Val PR-AUC: {val_metrics['pr_auc']:.4f}, "
                    f"Val F1: {val_metrics['f1']:.4f}, "
                    f"Val Recall: {val_metrics['recall']:.4f}, "
                    f"Val Precision: {val_metrics['precision']:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            # Save best model
            torch.save(best_model_state, f"{model.__class__.__name__}_best.pth")
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Load best model weights
    model.load_state_dict(best_model_state)
    return model, history


def plot_training_history(history, model_name):
    """Plot training history with all metrics"""

    # Create directory for plots if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{model_name}_loss.png")
    plt.close()

    # Plot all metrics in a comprehensive way
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'pr_auc']
    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metrics):
        values = [epoch_metrics[metric] for epoch_metrics in history['val_metrics']]
        plt.subplot(2, 3, i + 1)
        plt.plot(values, linewidth=2)
        plt.title(f'Validation {metric.replace("_", "-").upper()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace("_", "-").upper())
        plt.grid(True, alpha=0.3)
        # Add max value annotation
        max_val = max(values)
        max_epoch = values.index(max_val)
        plt.annotate(f'Max: {max_val:.3f}', 
                    xy=(max_epoch, max_val), 
                    xytext=(max_epoch, max_val + 0.02),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()


def three_stage_training(train_data, val_data, test_data):
    """
    Three-stage training process:
    1. Pretrain individual modules (MCRAN and BiRC-Mamba)
    2. Load pretrained weights and train fusion layers
    3. Fine-tune the entire model
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create datasets
    train_dataset = DNABindingDataset(train_data['sequences'], train_data['labels'])
    val_dataset = DNABindingDataset(val_data['sequences'], val_data['labels'])
    test_dataset = DNABindingDataset(test_data['sequences'], test_data['labels'])

    # Hyperparameters
    pretrain_batch_size = 64
    fusion_batch_size = 32
    pretrain_lr = 1e-3
    fusion_lr = 5e-4
    finetune_lr = 1e-5
    weight_decay = 1e-4

    # Create dataloaders
    pretrain_train_loader = DataLoader(
        train_dataset,
        batch_size=pretrain_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    fusion_train_loader = DataLoader(
        train_dataset,
        batch_size=fusion_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Define loss function
    criterion = nn.BCELoss()

    # Stage 1: Pretrain MCRAN and BiRC-Mamba individually
    logger.info("Stage 1: Pretraining individual modules")

    # Initialize models
    mcran = MCRAN(seq_length=101, dropout_rate=0.3).to(device)
    bircmamba = BiRCMamba(d_model=128, n_layer=3, d_state=16, dropout=0.3).to(device)

    # Add a classifier head to BiRCMamba
    bircmamba.fc = nn.Sequential(
        nn.Linear(256, 64),  # 256 = 128*2 (after global pooling and concatenation)
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1)
    ).to(device)

    # Pretrain MCRAN
    mcran_optimizer = optim.Adam(mcran.parameters(), lr=pretrain_lr, weight_decay=weight_decay)
    mcran_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        mcran_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    logger.info("Pretraining MCRAN...")
    pretrained_mcran, mcran_history = train_model(
        mcran,
        pretrain_train_loader,
        val_loader,
        mcran_optimizer,
        criterion,
        device,
        num_epochs=50,
        patience=10,
        scheduler=mcran_scheduler
    )

    plot_training_history(mcran_history, "MCRAN_Pretrain")

    # Pretrain BiRC-Mamba
    bircmamba_optimizer = optim.Adam(bircmamba.parameters(), lr=pretrain_lr, weight_decay=weight_decay)
    bircmamba_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        bircmamba_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    logger.info("Pretraining BiRC-Mamba...")
    pretrained_bircmamba, bircmamba_history = train_model(
        bircmamba,
        pretrain_train_loader,
        val_loader,
        bircmamba_optimizer,
        criterion,
        device,
        num_epochs=50,
        patience=10,
        scheduler=bircmamba_scheduler
    )

    plot_training_history(bircmamba_history, "BiRCMamba_Pretrain")

    # Stage 2: Train fusion layers
    logger.info("Stage 2: Training fusion layers")

    # Initialize MDNetDBP with pretrained weights
    mdnetdbp = MDNetDBP(
        seq_length=101,
        mcran_dim=128,
        bircmamba_dim=128,
        fusion_dim=128,
        dropout_rate=0.3
    ).to(device)

    # Load pretrained weights
    mdnetdbp.mcran.load_state_dict(pretrained_mcran.state_dict())
    mdnetdbp.bircmamba.load_state_dict(pretrained_bircmamba.state_dict())

    # Freeze pretrained modules
    for param in mdnetdbp.mcran.parameters():
        param.requires_grad = False

    for param in mdnetdbp.bircmamba.parameters():
        param.requires_grad = False

    # Only train fusion layers
    fusion_params = list(mdnetdbp.mcran_feature_extractor.parameters()) + \
                    list(mdnetdbp.mffs.parameters())

    fusion_optimizer = optim.Adam(fusion_params, lr=fusion_lr, weight_decay=weight_decay)
    fusion_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        fusion_optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    logger.info("Training fusion layers...")
    fusion_trained_model, fusion_history = train_model(
        mdnetdbp,
        fusion_train_loader,
        val_loader,
        fusion_optimizer,
        criterion,
        device,
        num_epochs=30,
        patience=8,
        scheduler=fusion_scheduler
    )

    plot_training_history(fusion_history, "MDNetDBP_Fusion")

    # Stage 3: Fine-tune the entire model
    logger.info("Stage 3: Fine-tuning the entire model")

    # Unfreeze all parameters
    for param in mdnetdbp.parameters():
        param.requires_grad = True

    finetune_optimizer = optim.Adam(mdnetdbp.parameters(), lr=finetune_lr, weight_decay=weight_decay)
    finetune_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        finetune_optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    logger.info("Fine-tuning MDNetDBP...")
    final_model, finetune_history = train_model(
        mdnetdbp,
        fusion_train_loader,
        val_loader,
        finetune_optimizer,
        criterion,
        device,
        num_epochs=20,
        patience=5,
        scheduler=finetune_scheduler
    )

    plot_training_history(finetune_history, "MDNetDBP_Finetune")

    # Final evaluation on test set
    logger.info("Evaluating final model on test set...")
    test_metrics = evaluate_model(final_model, test_loader, device)

    # Log all test metrics
    logger.info("="*50)
    logger.info("FINAL TEST RESULTS:")
    logger.info("="*50)
    logger.info(f"Test Accuracy:   {test_metrics['accuracy']:.4f}")
    logger.info(f"Test AUC:        {test_metrics['auc']:.4f}")
    logger.info(f"Test PR-AUC:     {test_metrics['pr_auc']:.4f}")
    logger.info(f"Test F1 Score:   {test_metrics['f1']:.4f}")
    logger.info(f"Test Recall:     {test_metrics['recall']:.4f}")
    logger.info(f"Test Precision:  {test_metrics['precision']:.4f}")
    logger.info("="*50)

    # Save final model
    torch.save(final_model.state_dict(), "MDNetDBP_final.pth")
    logger.info("Final model saved as MDNetDBP_final.pth")

    return final_model, test_metrics


if __name__ == "__main__":

    def generate_dummy_data(num_samples, seq_length=101):
        bases = ['A', 'T', 'C', 'G']
        sequences = [''.join(np.random.choice(bases) for _ in range(seq_length)) for _ in range(num_samples)]
        labels = np.random.randint(0, 2, num_samples)
        return {'sequences': sequences, 'labels': labels}

    train_data = generate_dummy_data(1000)
    val_data = generate_dummy_data(200)
    test_data = generate_dummy_data(300)

    # Run three-stage training
    start_time = time.time()
    final_model, test_metrics = three_stage_training(train_data, val_data, test_data)
    training_time = time.time() - start_time

    logger.info(f"Total training time: {training_time / 60:.2f} minutes")
    logger.info(f"Final test metrics: {test_metrics}")
