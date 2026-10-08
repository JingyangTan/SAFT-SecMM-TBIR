import os
import random
from collections import Counter
from dataclasses import dataclass

import mobileclip
import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    GaussianBlur,
    Normalize,
    RandomApply,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from torchvision.transforms.functional import InterpolationMode

from src.models.TinyCLIP.src import open_clip as open_clip_tiny

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


class ClipDistillData(Dataset):
    CAPTIONS_PER_IMAGE = 5

    def __init__(
        self,
        model_arch=None,
        model_type=None,
        train_image_root=None,
        train_annotation_files=None,
        val_image_root=None,
        val_annotation_files=None,
        train=True,
        *,
        train_augmentation,
    ):
        self.train_mode = train
        if train:
            self.data = self.load_data(train_image_root, train_annotation_files)
            self.image_transforms = self._build_train_transform(model_type, train_augmentation)
        else:
            self.data = self.load_data(val_image_root, val_annotation_files)
            self.image_transforms = self._build_val_transform(model_type)

        self.tokenizer = self._build_tokenizer(model_type, model_arch)

    @staticmethod
    def _build_train_transform(model_type, train_augmentation):
        if model_type == "mobileclip":
            image_size = 256
        elif model_type in ("tinyclip", "openclip"):
            image_size = 224
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        crop = train_augmentation["random_resized_crop"]
        color_jitter = train_augmentation["color_jitter"]
        blur = train_augmentation["gaussian_blur"]
        process_list = [
            RandomResizedCrop(
                size=(image_size, image_size),
                scale=tuple(crop["scale"]),
                ratio=tuple(crop["ratio"]),
            ),
            RandomHorizontalFlip(p=train_augmentation["horizontal_flip_probability"]),
            ColorJitter(
                brightness=color_jitter["brightness"],
                contrast=color_jitter["contrast"],
                saturation=color_jitter["saturation"],
                hue=color_jitter["hue"],
            ),
            RandomApply(
                [GaussianBlur(kernel_size=blur["kernel_size"], sigma=tuple(blur["sigma"]))],
                p=blur["probability"],
            ),
            ToTensor(),
        ]
        if model_type in ("tinyclip", "openclip"):
            process_list.append(Normalize(mean=CLIP_MEAN, std=CLIP_STD))
        return Compose(process_list)

    @staticmethod
    def _build_val_transform(model_type):
        if model_type == "mobileclip":
            process_list = [
                Resize(
                    size=256,
                    interpolation=INTERPOLATION_MODES["bilinear"],
                    max_size=None,
                    antialias=True,
                ),
                CenterCrop(size=(256, 256)),
                ToTensor(),
            ]
        elif model_type in ("tinyclip", "openclip"):
            process_list = [
                Resize(
                    size=224,
                    interpolation=INTERPOLATION_MODES["bicubic"],
                    max_size=None,
                    antialias=True,
                ),
                CenterCrop(size=(224, 224)),
                ToTensor(),
                Normalize(
                    mean=CLIP_MEAN,
                    std=CLIP_STD,
                ),
            ]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return Compose(process_list)

    @staticmethod
    def _build_tokenizer(model_type, model_arch):
        if model_type == "mobileclip":
            return mobileclip.get_tokenizer(model_arch)
        if model_type == "tinyclip":
            return open_clip_tiny.get_tokenizer(model_arch)
        if model_type == "openclip":
            return open_clip.get_tokenizer(model_arch)
        raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def _read_annotations(root_annotations):
        records = []
        with open(root_annotations, "r", encoding="utf-8") as annotation_file:
            for line_number, raw_line in enumerate(annotation_file, start=1):
                line = raw_line.strip()
                if not line:
                    raise ValueError(
                        f"Empty annotation line in {root_annotations}:{line_number}."
                    )
                if "\t" not in line:
                    raise ValueError(
                        f"Expected a tab-separated annotation in "
                        f"{root_annotations}:{line_number}."
                    )
                image_name, caption = line.split(None, 1)
                records.append((image_name.split("#")[0], caption))
        if not records:
            raise ValueError(f"No annotations found in {root_annotations}.")
        return records

    @staticmethod
    def _validate_image_files(root_image, image_names, root_annotations):
        missing_paths = [
            os.path.join(root_image, image_name)
            for image_name in image_names
            if not os.path.isfile(os.path.join(root_image, image_name))
        ]
        if missing_paths:
            preview = ", ".join(missing_paths[:5])
            raise FileNotFoundError(
                f"{len(missing_paths)} images referenced by {root_annotations} are missing; "
                f"first entries: {preview}"
            )

    @staticmethod
    def _print_dataset_summary(split, root_annotations, image_count, caption_count, **details):
        summary = [
            f"[{split}] {root_annotations}",
            f"images={image_count}",
            f"captions={caption_count}",
        ]
        summary.extend(f"{name}={value}" for name, value in details.items())
        print("Dataset audit: " + ", ".join(summary))

    @staticmethod
    def _group_annotations(records):
        grouped_records = {}
        for line_index, (image_name, caption) in enumerate(records):
            grouped_records.setdefault(image_name, []).append((line_index, caption))
        return grouped_records

    def _load_image_text_samples(self, root_image, root_annotations):
        records = self._read_annotations(root_annotations)
        grouped_records = self._group_annotations(records)

        self._validate_image_files(root_image, grouped_records, root_annotations)
        caption_counts = [len(annotations) for annotations in grouped_records.values()]
        self._print_dataset_summary(
            "val" if not self.train_mode else "train",
            root_annotations,
            len(grouped_records),
            len(records),
            captions_per_image=f"{min(caption_counts)}-{max(caption_counts)}",
        )
        return [
            ImageTextSample(
                os.path.join(root_image, image_name),
                [caption for _, caption in annotations],
            )
            for image_name, annotations in grouped_records.items()
        ]

    def _load_distillation_samples(self, root_image, root_annotations):
        annotation_stem, _ = os.path.splitext(root_annotations)
        text_emb_path = f"{annotation_stem}_text_emb.npy"
        teacher_text_embeddings = np.load(text_emb_path)

        image_emb_path = f"{annotation_stem}_image_emb.npy"
        teacher_image_embeddings = np.load(image_emb_path)

        for embedding_path, embeddings in (
            (text_emb_path, teacher_text_embeddings),
            (image_emb_path, teacher_image_embeddings),
        ):
            if not np.issubdtype(embeddings.dtype, np.floating):
                raise ValueError(
                    f"Teacher embeddings in {embedding_path} must have a floating-point "
                    f"dtype; got {embeddings.dtype}."
                )
            if not np.isfinite(embeddings).all():
                raise ValueError(
                    f"Teacher embeddings in {embedding_path} contain NaN or Inf."
                )

        records = self._read_annotations(root_annotations)
        grouped_records = self._group_annotations(records)
        if teacher_text_embeddings.ndim != 2 or teacher_image_embeddings.ndim != 2:
            raise ValueError(
                f"Teacher embeddings for {root_annotations} must be two-dimensional; "
                f"got text={teacher_text_embeddings.shape}, image={teacher_image_embeddings.shape}."
            )
        if len(records) != teacher_text_embeddings.shape[0]:
            raise ValueError(
                f"Caption count and text embedding count differ for {root_annotations}: "
                f"{len(records)} captions vs {teacher_text_embeddings.shape[0]} text embeddings."
            )
        if teacher_text_embeddings.shape[1] != teacher_image_embeddings.shape[1]:
            raise ValueError(
                f"Teacher embedding dimensions differ for {root_annotations}: "
                f"text={teacher_text_embeddings.shape[1]}, "
                f"image={teacher_image_embeddings.shape[1]}."
            )

        image_names = [image_name for image_name, _ in records]
        captions_per_image = Counter(image_names)
        if any(count != self.CAPTIONS_PER_IMAGE for count in captions_per_image.values()):
            invalid_counts = sorted(set(captions_per_image.values()))
            raise ValueError(
                f"Distillation data requires exactly {self.CAPTIONS_PER_IMAGE} captions per image "
                f"for the current image-embedding alignment, but found counts "
                f"{invalid_counts} in {root_annotations}."
            )

        unique_image_names = list(grouped_records)
        if len(unique_image_names) != teacher_image_embeddings.shape[0]:
            raise ValueError(
                f"Image count and image embedding count differ for {root_annotations}: "
                f"{len(unique_image_names)} images vs "
                f"{teacher_image_embeddings.shape[0]} image embeddings."
            )
        for index, image_name in enumerate(image_names):
            expected_image_name = unique_image_names[index // self.CAPTIONS_PER_IMAGE]
            if image_name != expected_image_name:
                raise ValueError(
                    f"Captions for {image_name} are not a contiguous group of "
                    f"{self.CAPTIONS_PER_IMAGE} lines in {root_annotations}; "
                    "the current image embedding alignment would be invalid."
                )

        self._validate_image_files(root_image, unique_image_names, root_annotations)
        self._print_dataset_summary(
            "train",
            root_annotations,
            len(unique_image_names),
            len(records),
            captions_per_image=self.CAPTIONS_PER_IMAGE,
            text_embeddings=teacher_text_embeddings.shape,
            image_embeddings=teacher_image_embeddings.shape,
        )

        return [
            ImageTextEmbeddingSample(
                os.path.join(root_image, image_name),
                [caption for _, caption in annotations],
                [teacher_text_embeddings[line_index] for line_index, _ in annotations],
                [teacher_image_embeddings[annotations[0][0] // self.CAPTIONS_PER_IMAGE]],
            )
            for image_name, annotations in grouped_records.items()
        ]

    def load_data(self, image_root_list, text_list):
        if len(image_root_list) != len(text_list):
            raise ValueError(
                "The number of image roots and annotation files must match; "
                f"got {len(image_root_list)} and {len(text_list)}."
            )

        data_all = []
        for image_root, annotation_path in zip(image_root_list, text_list):
            if self.train_mode:
                data_list = self._load_distillation_samples(
                    root_image=image_root, root_annotations=annotation_path
                )
            else:
                data_list = self._load_image_text_samples(
                    root_image=image_root, root_annotations=annotation_path
                )
            data_all.extend(data_list)
        return data_all

    def data_transforms(self, image_path, caption):
        image = Image.open(image_path).convert("RGB")
        transformed_image = self.image_transforms(image)

        tokenized_text = self.tokenizer(caption).squeeze(0)

        return transformed_image, tokenized_text

    def __getitem__(self, index):
        item = self.data[index]
        if self.train_mode:
            caption_index = random.randrange(len(item.captions))
        else:
            caption_index = index % len(item.captions)
        caption = item.captions[caption_index]
        transformed_image, tokenized_text = self.data_transforms(item.image_path, caption)
        if self.train_mode:
            teacher_text_embedding = torch.from_numpy(item.text_embeddings[caption_index])
            teacher_image_embedding = torch.from_numpy(item.image_embedding[0])
            return (
                transformed_image,
                tokenized_text,
                teacher_text_embedding,
                teacher_image_embedding,
            )
        return transformed_image, tokenized_text

    def __len__(self):
        return len(self.data)


@dataclass
class ImageTextSample:
    image_path: str = ""
    captions: list[str] | None = None


@dataclass
class ImageTextEmbeddingSample:
    image_path: str = ""
    captions: list[str] | None = None
    # One teacher text embedding per caption, in the same order.
    text_embeddings: list[np.ndarray] | None = None
    # The existing data layout stores one teacher image embedding in a list.
    image_embedding: list[np.ndarray] | None = None
