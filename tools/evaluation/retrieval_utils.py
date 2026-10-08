import json
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class RetrievalImageDataset(Dataset):
    """Load evaluation images and fail with the path of an unreadable image."""

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            raise RuntimeError(f"Failed to load image: {image_path}") from error
        return self.transform(image), image_path


def _normalize_query_key(query):
    return " ".join(query.split())


def _load_queries(query_file):
    with open(query_file, encoding="utf-8") as stream:
        queries = [line.strip() for line in stream]
    if not queries:
        raise ValueError(f"No queries found in {query_file}.")
    if any(not query for query in queries):
        raise ValueError(f"Empty query found in {query_file}.")
    if len(queries) != len(set(queries)):
        raise ValueError(f"Duplicate queries found in {query_file}.")
    return queries


def _load_ground_truth(label_file):
    with open(label_file, encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Ground truth in {label_file} must be a non-empty list.")

    ground_truth = {}
    for record in records:
        if "text_query" not in record or "image_name" not in record:
            raise ValueError(
                f"Each ground-truth record in {label_file} must contain "
                "text_query and image_name."
            )
        query = _normalize_query_key(
            " ".join(record["text_query"].split("_")) + "."
        )
        image_names = record["image_name"]
        if query in ground_truth:
            raise ValueError(f"Duplicate ground-truth query in {label_file}: {query}")
        if not isinstance(image_names, list) or not image_names:
            raise ValueError(f"Ground-truth query has no images: {query}")
        ground_truth[query] = set(image_names)
    return ground_truth


def _average_precision_at_k(predictions, relevant_images, top_k):
    relevant_count = 0
    precision_sum = 0.0
    for rank, image_name in enumerate(predictions[:top_k], start=1):
        if image_name in relevant_images:
            relevant_count += 1
            precision_sum += relevant_count / rank
    return precision_sum / min(len(relevant_images), top_k)


def compute_map_at_k(scores, image_paths, queries, label_file, top_k):
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if scores.ndim != 2:
        raise ValueError(f"Scores must be two-dimensional; got {tuple(scores.shape)}.")
    if scores.shape != (len(queries), len(image_paths)):
        raise ValueError(
            "Score shape does not match query and image counts: "
            f"scores={tuple(scores.shape)}, queries={len(queries)}, "
            f"images={len(image_paths)}."
        )
    if top_k > len(image_paths):
        raise ValueError(
            f"top_k={top_k} exceeds the image count {len(image_paths)}."
        )

    ground_truth = _load_ground_truth(label_file)
    query_keys = [_normalize_query_key(query) for query in queries]
    missing_queries = [query for query in query_keys if query not in ground_truth]
    if missing_queries:
        preview = ", ".join(missing_queries[:5])
        raise ValueError(
            f"{len(missing_queries)} queries are missing from {label_file}; "
            f"first entries: {preview}"
        )

    image_names = [os.path.basename(image_path) for image_path in image_paths]
    available_images = set(image_names)
    missing_images = sorted(
        {
            image_name
            for query in query_keys
            for image_name in ground_truth[query]
            if image_name not in available_images
        }
    )
    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise ValueError(
            f"{len(missing_images)} ground-truth images are missing from the image root; "
            f"first entries: {preview}"
        )

    _, top_indices = torch.topk(scores, k=top_k, dim=1)
    map_sum = 0.0
    for row, query in enumerate(query_keys):
        predictions = [image_names[index] for index in top_indices[row].tolist()]
        map_sum += _average_precision_at_k(
            predictions, ground_truth[query], top_k
        )
    return map_sum / len(query_keys), top_indices


def extract_image_embeddings(
    model,
    preprocess,
    image_root,
    evaluation_device,
    batch_size,
    num_workers,
    preserve_model_dtype=False,
):
    image_paths = sorted(
        os.path.join(image_root, filename)
        for filename in os.listdir(image_root)
        if filename.endswith(".jpg")
    )
    if not image_paths:
        raise ValueError(f"No .jpg images found in {image_root}.")

    dataset = RetrievalImageDataset(image_paths, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    image_feature_batches = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_index, (images, _paths) in enumerate(dataloader, start=1):
            images = images.to(evaluation_device, non_blocking=True)
            image_features = model.encode_image(images)
            image_norms = image_features.norm(p=2, dim=-1, keepdim=True)
            if preserve_model_dtype:
                image_features /= image_norms
            else:
                image_features = image_features / image_norms
            image_feature_batches.append(image_features.cpu())
            if batch_index % 20 == 0 or batch_index == len(dataloader):
                completed = min(batch_index * batch_size, len(dataset))
                print(f"Images encoded: {completed}/{len(dataset)}", flush=True)

    image_embeddings = torch.cat(image_feature_batches, dim=0)
    return image_paths, image_embeddings


def encode_queries(model, tokenizer, query_file, evaluation_device):
    queries = _load_queries(query_file)
    text_feature_batches = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for query in queries:
            tokenized_text = tokenizer([query]).to(evaluation_device)
            text_features = model.encode_text(tokenized_text)
            text_features = text_features / text_features.norm(
                p=2, dim=-1, keepdim=True
            )
            text_feature_batches.append(text_features.cpu())
    return queries, torch.cat(text_feature_batches, dim=0).float()


def _load_caption_records(label_file):
    records = []
    with open(label_file, encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                raise ValueError(f"Empty caption line in {label_file}:{line_number}.")
            fields = line.split("\t", 1)
            if len(fields) != 2 or not fields[0] or not fields[1]:
                raise ValueError(
                    f"Expected an image ID and caption separated by a tab in "
                    f"{label_file}:{line_number}."
                )
            image_name = fields[0].split("#", 1)[0]
            records.append((image_name, fields[1]))
    if not records:
        raise ValueError(f"No captions found in {label_file}.")
    return records


def _validate_five_caption_order(image_paths, caption_records, label_file):
    captions_per_image = 5
    expected_caption_count = len(image_paths) * captions_per_image
    if len(caption_records) != expected_caption_count:
        raise ValueError(
            f"Recall evaluation expects {captions_per_image} captions per image, "
            f"but {label_file} contains {len(caption_records)} captions for "
            f"{len(image_paths)} images."
        )

    for image_index, image_path in enumerate(image_paths):
        expected_name = os.path.basename(image_path)
        group_start = image_index * captions_per_image
        group_names = {
            caption_records[index][0]
            for index in range(group_start, group_start + captions_per_image)
        }
        if group_names != {expected_name}:
            raise ValueError(
                f"Caption group {image_index} in {label_file} does not match "
                f"the sorted image {expected_name}; found {sorted(group_names)}."
            )
    return captions_per_image


def evaluate_text_to_image_recall(
    model,
    tokenizer,
    label_file,
    image_paths,
    image_embeddings,
    evaluation_device,
    batch_size,
    top_k,
):
    caption_records = _load_caption_records(label_file)
    captions_per_image = _validate_five_caption_order(
        image_paths,
        caption_records,
        label_file,
    )
    if top_k > len(image_paths):
        raise ValueError(
            f"top_k={top_k} exceeds the image count {len(image_paths)}."
        )

    captions = [caption for _image_name, caption in caption_records]
    hit_count = 0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for start in range(0, len(captions), batch_size):
            caption_batch = captions[start : start + batch_size]
            tokenized_text = tokenizer(caption_batch).to(evaluation_device)
            text_features = model.encode_text(tokenized_text)
            text_features /= text_features.norm(
                p=2,
                dim=-1,
                keepdim=True,
            )
            scores = torch.matmul(text_features.cpu(), image_embeddings.t())
            top_indices = torch.topk(scores, k=top_k, dim=1).indices
            target_indices = (
                torch.arange(start, start + len(caption_batch))
                // captions_per_image
            )
            matches = top_indices.eq(target_indices.unsqueeze(1))
            hit_count += matches.any(dim=1).sum().item()

            completed = start + len(caption_batch)
            if completed % (batch_size * 20) == 0 or completed == len(captions):
                print(f"Captions encoded: {completed}/{len(captions)}", flush=True)

    return {
        "image_count": len(image_paths),
        "caption_count": len(captions),
        "captions_per_image": captions_per_image,
        "top_k": top_k,
        "text_to_image_recall_at_k": hit_count / len(captions),
    }


def evaluate_map(
    model,
    preprocess,
    tokenizer,
    query_file,
    image_root,
    label_file,
    evaluation_device,
    batch_size,
    num_workers,
    top_k,
):
    image_paths, image_embeddings = extract_image_embeddings(
        model,
        preprocess,
        image_root,
        evaluation_device,
        batch_size,
        num_workers,
    )
    image_embeddings = image_embeddings.float()
    queries, text_embeddings = encode_queries(
        model, tokenizer, query_file, evaluation_device
    )
    if text_embeddings.shape[1] != image_embeddings.shape[1]:
        raise ValueError(
            "Text and image embedding dimensions differ: "
            f"{text_embeddings.shape[1]} vs {image_embeddings.shape[1]}."
        )

    scores = torch.matmul(text_embeddings, image_embeddings.t())
    map_at_k, top_indices = compute_map_at_k(
        scores, image_paths, queries, label_file, top_k
    )
    return {
        "image_count": len(image_paths),
        "query_count": len(queries),
        "matched_query_count": len(queries),
        "top_k": top_k,
        "map_at_k": map_at_k,
    }, top_indices


def validate_map_evaluation_config(config):
    for field in ("evaluation", "benchmarks"):
        if field not in config:
            raise ValueError(f"Missing evaluation config field: {field}")

    evaluation = config["evaluation"]
    for field in ("device", "batch_size", "num_workers", "top_k", "result_file"):
        if field not in evaluation:
            raise ValueError(f"Missing evaluation config field: evaluation.{field}")
    if not isinstance(evaluation["device"], str) or not evaluation["device"]:
        raise ValueError("evaluation.device must be a non-empty string.")
    if not isinstance(evaluation["result_file"], str) or not evaluation["result_file"]:
        raise ValueError("evaluation.result_file must be a non-empty string.")
    if (
        isinstance(evaluation["batch_size"], bool)
        or not isinstance(evaluation["batch_size"], int)
        or evaluation["batch_size"] <= 0
    ):
        raise ValueError("evaluation.batch_size must be a positive integer.")
    if (
        isinstance(evaluation["num_workers"], bool)
        or not isinstance(evaluation["num_workers"], int)
        or evaluation["num_workers"] < 0
    ):
        raise ValueError("evaluation.num_workers must be a non-negative integer.")
    if (
        isinstance(evaluation["top_k"], bool)
        or not isinstance(evaluation["top_k"], int)
        or evaluation["top_k"] <= 0
    ):
        raise ValueError("evaluation.top_k must be a positive integer.")

    benchmarks = config["benchmarks"]
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("benchmarks must be a non-empty list.")
    for benchmark in benchmarks:
        for field in ("name", "image_root", "query_file", "label_file"):
            if not isinstance(benchmark.get(field), str) or not benchmark[field]:
                raise ValueError(f"Invalid benchmark field: {field}")
        if not os.path.isdir(benchmark["image_root"]):
            raise FileNotFoundError(
                f"Benchmark image root does not exist: {benchmark['image_root']}"
            )
        for field in ("query_file", "label_file"):
            if not os.path.isfile(benchmark[field]):
                raise FileNotFoundError(
                    f"Benchmark file does not exist: {benchmark[field]}"
                )
    benchmark_names = [benchmark["name"] for benchmark in benchmarks]
    if len(benchmark_names) != len(set(benchmark_names)):
        raise ValueError("Benchmark names must be unique.")


def validate_recall_evaluation_config(config):
    for field in ("evaluation", "benchmarks"):
        if field not in config:
            raise ValueError(f"Missing evaluation config field: {field}")

    evaluation = config["evaluation"]
    for field in ("device", "batch_size", "num_workers", "top_k", "result_file"):
        if field not in evaluation:
            raise ValueError(f"Missing evaluation config field: evaluation.{field}")
    if not isinstance(evaluation["device"], str) or not evaluation["device"]:
        raise ValueError("evaluation.device must be a non-empty string.")
    if not isinstance(evaluation["result_file"], str) or not evaluation["result_file"]:
        raise ValueError("evaluation.result_file must be a non-empty string.")
    if (
        isinstance(evaluation["batch_size"], bool)
        or not isinstance(evaluation["batch_size"], int)
        or evaluation["batch_size"] <= 0
    ):
        raise ValueError("evaluation.batch_size must be a positive integer.")
    if (
        isinstance(evaluation["num_workers"], bool)
        or not isinstance(evaluation["num_workers"], int)
        or evaluation["num_workers"] < 0
    ):
        raise ValueError("evaluation.num_workers must be a non-negative integer.")

    if (
        isinstance(evaluation["top_k"], bool)
        or not isinstance(evaluation["top_k"], int)
        or evaluation["top_k"] <= 0
    ):
        raise ValueError("evaluation.top_k must be a positive integer.")

    benchmarks = config["benchmarks"]
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("benchmarks must be a non-empty list.")
    for benchmark in benchmarks:
        for field in ("name", "image_root", "label_file"):
            if not isinstance(benchmark.get(field), str) or not benchmark[field]:
                raise ValueError(f"Invalid benchmark field: {field}")
        if not os.path.isdir(benchmark["image_root"]):
            raise FileNotFoundError(
                f"Benchmark image root does not exist: {benchmark['image_root']}"
            )
        if not os.path.isfile(benchmark["label_file"]):
            raise FileNotFoundError(
                f"Benchmark file does not exist: {benchmark['label_file']}"
            )
    benchmark_names = [benchmark["name"] for benchmark in benchmarks]
    if len(benchmark_names) != len(set(benchmark_names)):
        raise ValueError("Benchmark names must be unique.")


def save_evaluation_results(result_file, content):
    result_directory = os.path.dirname(result_file)
    if result_directory:
        os.makedirs(result_directory, exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run_map_evaluation(
    model,
    preprocess,
    tokenizer,
    model_metadata,
    evaluation_config,
    benchmarks,
):
    evaluation_device = evaluation_config["device"]
    model.to(evaluation_device)
    model.eval()

    output = {
        "model": model_metadata,
        "evaluation": {
            "device": evaluation_device,
            "batch_size": evaluation_config["batch_size"],
            "num_workers": evaluation_config["num_workers"],
            "top_k": evaluation_config["top_k"],
        },
        "benchmarks": [],
    }
    for benchmark in benchmarks:
        print(
            f"Evaluating {benchmark['name']}: mAP@{evaluation_config['top_k']}",
            flush=True,
        )
        result, _top_indices = evaluate_map(
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            query_file=benchmark["query_file"],
            image_root=benchmark["image_root"],
            label_file=benchmark["label_file"],
            evaluation_device=evaluation_device,
            batch_size=evaluation_config["batch_size"],
            num_workers=evaluation_config["num_workers"],
            top_k=evaluation_config["top_k"],
        )
        benchmark_result = {"name": benchmark["name"], **result}
        output["benchmarks"].append(benchmark_result)
        save_evaluation_results(evaluation_config["result_file"], output)
        print(
            f"{benchmark['name']}: images={result['image_count']}, "
            f"queries={result['matched_query_count']}/{result['query_count']}, "
            f"mAP@{result['top_k']}={result['map_at_k']}",
            flush=True,
        )
    return output


def run_recall_evaluation(
    model,
    preprocess,
    tokenizer,
    model_metadata,
    evaluation_config,
    benchmarks,
):
    evaluation_device = evaluation_config["device"]
    model.to(evaluation_device)
    model.eval()

    output = {
        "model": model_metadata,
        "evaluation": {
            "device": evaluation_device,
            "batch_size": evaluation_config["batch_size"],
            "num_workers": evaluation_config["num_workers"],
            "top_k": evaluation_config["top_k"],
        },
        "benchmarks": [],
    }
    for benchmark in benchmarks:
        print(
            f"Evaluating {benchmark['name']}: text-to-image "
            f"Recall@{evaluation_config['top_k']}",
            flush=True,
        )
        image_paths, image_embeddings = extract_image_embeddings(
            model=model,
            preprocess=preprocess,
            image_root=benchmark["image_root"],
            evaluation_device=evaluation_device,
            batch_size=evaluation_config["batch_size"],
            num_workers=evaluation_config["num_workers"],
            preserve_model_dtype=True,
        )
        result = evaluate_text_to_image_recall(
            model=model,
            tokenizer=tokenizer,
            label_file=benchmark["label_file"],
            image_paths=image_paths,
            image_embeddings=image_embeddings,
            evaluation_device=evaluation_device,
            batch_size=evaluation_config["batch_size"],
            top_k=evaluation_config["top_k"],
        )
        benchmark_result = {"name": benchmark["name"], **result}
        output["benchmarks"].append(benchmark_result)
        save_evaluation_results(evaluation_config["result_file"], output)
        print(
            f"{benchmark['name']}: images={result['image_count']}, "
            f"captions={result['caption_count']}, "
            f"R@{result['top_k']}={result['text_to_image_recall_at_k']}",
            flush=True,
        )
    return output
