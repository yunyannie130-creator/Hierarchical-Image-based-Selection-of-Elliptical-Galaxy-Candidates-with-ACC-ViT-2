from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from MODEL import VisionTransformer
from utils_p import get_config

FINAL_CATEGORIES = ["0_com", "1_inb", "2_cig_and_edg", "3_spi"]

class UnlabeledImageDataset(Dataset):
    """读取一个目录下没有真实标签的图像。"""

    SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"找不到无标签图像目录：{self.root.resolve()}"
            )

        self.image_paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in self.SUPPORTED_SUFFIXES
        )

        if not self.image_paths:
            raise RuntimeError(
                f"目录中没有 jpg、jpeg 或 png 图像：{self.root.resolve()}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, str(image_path)

def load_state_dict_safely(
    model: torch.nn.Module,
    model_path,
    device,
):
    """
    加载常见格式的 PyTorch 模型权重。
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{model_path.resolve()}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    else:
        state_dict = checkpoint

    # 兼容 DataParallel 保存的 module. 前缀
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

def combine_branch_outputs(outputs, beta):
    """
    对 ViT 多分支输出进行加权求和。

    如果模型只返回一个 Tensor，则直接使用。
    """
    if torch.is_tensor(outputs):
        return outputs

    if not isinstance(outputs, (list, tuple)):
        raise TypeError(
            f"模型输出类型不支持：{type(outputs)}"
        )

    if len(outputs) != len(beta):
        raise ValueError(
            f"模型输出有 {len(outputs)} 个分支，"
            f"但 beta 有 {len(beta)} 个权重。"
        )

    weighted_output = torch.zeros_like(outputs[0])

    for output, weight in zip(outputs, beta):
        weighted_output += output * float(weight)

    return weighted_output

def predict_unlabeled_two_layer_model(
    data_loader,
    first_layer_model_path,
    second_layer_model_path,
    device,
    beta1=None,
    beta2=None,
    output_csv="./results/non_predictions.csv",
):
    """对无标签图像执行两阶段推理并保存最终四分类预测。"""
    if beta1 is None:
        beta1 = [1.0, 1.0, 1.0, 1.0]

    if beta2 is None:
        beta2 = [1.0, 1.0, 1.0, 1.0]

    model1 = VisionTransformer(
        get_config(), img_size=224, num_classes=3
    ).to(device)
    load_state_dict_safely(
        model1, first_layer_model_path, device
    )
    model1.eval()

    model2 = VisionTransformer(
        get_config(), img_size=224, num_classes=2
    ).to(device)
    load_state_dict_safely(
        model2, second_layer_model_path, device
    )
    model2.eval()

    records = []
    total_batches = len(data_loader)

    with torch.no_grad():
        for batch_index, (images, image_paths) in enumerate(
            data_loader, start=1
        ):
            images = images.to(device, non_blocking=True)

            logits1 = combine_branch_outputs(
                model1(images, swap=False), beta1
            )
            probabilities1 = torch.softmax(logits1, dim=1)
            predictions1 = logits1.argmax(dim=1)

            final_predictions = torch.empty_like(predictions1)
            final_predictions[predictions1 == 1] = 2
            final_predictions[predictions1 == 2] = 3

            second_probabilities = torch.full(
                (images.size(0), 2),
                float("nan"),
                device=device,
            )
            ell_mask = predictions1 == 0

            if ell_mask.any():
                logits2 = combine_branch_outputs(
                    model2(images[ell_mask], swap=False), beta2
                )
                probabilities2 = torch.softmax(logits2, dim=1)
                predictions2 = logits2.argmax(dim=1)
                final_predictions[ell_mask] = predictions2
                second_probabilities[ell_mask] = probabilities2

            for index, image_path_text in enumerate(image_paths):
                image_path = Path(image_path_text)
                first_label = int(predictions1[index].item())
                final_label = int(final_predictions[index].item())

                records.append({
                    "filename": image_path.name,
                    "object_id": image_path.stem,
                    "full_path": str(image_path.resolve()),
                    "first_stage_label": first_label,
                    "first_stage_category": [
                        "ELL", "CIG+EDG", "SPI"
                    ][first_label],
                    "first_stage_confidence": float(
                        probabilities1[index, first_label].item()
                    ),
                    "second_stage_confidence": (
                        float(
                            second_probabilities[
                                index, final_label
                            ].item()
                        )
                        if first_label == 0
                        else pd.NA
                    ),
                    "predicted_label": final_label,
                    "predicted_category": FINAL_CATEGORIES[
                        final_label
                    ],
                })

            if (
                batch_index == 1
                or batch_index % 10 == 0
                or batch_index == total_batches
            ):
                print(
                    f"无标签推理进度：{batch_index}/"
                    f"{total_batches} batch"
                )

    result = pd.DataFrame.from_records(records)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"预测结果已保存：{output_path.resolve()}")
    print("各预测类别数量：")
    print(result["predicted_category"].value_counts())

    return result
