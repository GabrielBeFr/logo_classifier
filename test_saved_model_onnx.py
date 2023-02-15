import torch
import tqdm
from train import SaveBestModel, compute_metrics
from dataset import get_dataloader
import pathlib
from utils import get_labels
import onnxruntime as ort
import numpy as np
import sklearn

def test_model(ort_session, prohibited_classes):

    train_path = pathlib.Path("datasets/train_dataset.hdf5")
    val_path = pathlib.Path("datasets/val_dataset.hdf5")
    test_path = pathlib.Path("datasets/test_dataset.hdf5")
    _, _, test_loader = get_dataloader(train_path, val_path, test_path, prohibited_classes = prohibited_classes, test = True)

    classes_str, classes_ids = get_labels("datasets/class_infos.jsonl", prohibited_classes=prohibited_classes)

    # Evaluate the model on validation set

    correct = 0
    total = 0
    missed_logos = {}
    y_test = []
    y_pred = []
    with torch.no_grad():
        print("Entering testing loop")
        for embeddings, labels, ids in tqdm.tqdm(test_loader):
            outputs = ort_session.run(None,{"embeddings":embeddings.detach().cpu().numpy()})[0]
            predicted = np.argmax(outputs,1)
            total += labels.size(0)
            ground_truth = np.array([np.where(classe==1)[0][0] for classe in labels])
            correct += (predicted == ground_truth).sum().item()
            for indice in np.where((predicted == ground_truth)==False)[0].tolist():
                missed_logos[str(ids[indice].item())]=[ground_truth[indice].item(),predicted[indice].item()]
            y_pred += predicted.tolist()
            y_test += ground_truth.tolist()

    # Compute metrics and save model
    report = sklearn.metrics.classification_report(y_test, y_pred, labels=classes_ids, target_names=classes_str)
    f1_micro = sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred, labels = classes_ids, average="micro")
    f1_macro = sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred, labels = classes_ids, average="macro")
    f1_classes = sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred, labels = classes_ids, average=None)
    print(report)
    print(f1_micro)
    print(f1_macro)
    print(f1_classes)

if __name__ == "__main__":
    ort_session = ort.InferenceSession("wandb/run-20230214_164701-z2sdrr0a/files/logos_classifier.onnx")
    test_model(ort_session=ort_session,prohibited_classes=[])
