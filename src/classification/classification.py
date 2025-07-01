import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from src.classification.emb.embeder import ResNetEmbedder
from src.utils import load_config

class SpeakerClustering:
    def __init__(
            self,
            podcasts_path: str,
            threshold=0.85,
            model_path: str = 'voxblink2_samresnet100_ft',
            device: str = 'cuda:0'
        ):

        self.podcasts_path = podcasts_path
        df_path = os.path.join(podcasts_path, 'results.csv')
        self.full_df = pd.read_csv(df_path)  
        
        if 'fullness' not in self.full_df.columns:
            self.full_df['fullness'] = np.nan
        
        self.df = self.full_df[self.full_df['is_mono'] == True].copy() 
        self.threshold = threshold
        self.model_path = model_path
        self.device = device
        self.all_clusters = []  # {"embeddings": list, "paths": list}

    def norm_cos_sim(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)
        
        cosine_score = F.cosine_similarity(
            F.normalize(emb1, p=2, dim=1),
            F.normalize(emb2, p=2, dim=1),
            dim=1
        )
        cosine_score = cosine_score.item()
        return (cosine_score + 1.0) / 2  # normalize: [-1, 1] => [0, 1]
    
    def load_embedding(self, audio_path: str) -> torch.Tensor:
        emb_path = audio_path.rsplit('.', 1)[0] + '.emb'
        if os.path.exists(emb_path):
            try:
                emb, _ = torch.load(emb_path)
                return emb
            except Exception as e:
                logger.error(f"Failed to load embedding from {emb_path}: {str(e)}")

    def cluster_mono_podcast(self, podcast_id: int):
        group = self.df[self.df['podcast_id'] == podcast_id]
        temp_clusters = []

        for _, row in group.iterrows():
            audio_path = os.path.join(
                self.podcasts_path,
                row["audio_path"]
                )
            emb = self.load_embedding(audio_path)

            if emb is None:
                continue

            best_similarity = 0.0
            best_cluster = None

            for cluster in temp_clusters:
                cluster_embs = torch.stack(cluster["embeddings"])
                avg_emb = cluster_embs.mean(dim=0)
    
                sim = self.norm_cos_sim(emb, avg_emb)
                if sim > best_similarity:
                    best_similarity = sim
                    best_cluster = cluster

            if best_similarity >= self.threshold and best_cluster is not None:
                best_cluster["embeddings"].append(emb)
                best_cluster["paths"].append(row["audio_path"])
            else:
                new_cluster = {
                    "embeddings": [emb],
                    "paths": [row["audio_path"]]
                }
                temp_clusters.append(new_cluster)
        
        self.all_clusters.extend(temp_clusters)

    def cluster_all_podcast(self):
        for podcast_id in tqdm(self.df['podcast_id'].unique()):
            self.cluster_mono_podcast(int(podcast_id))

    def merge_clusters(self):
        changed = True
        while changed:
            changed = False
            new_clusters = []
            used = [False] * len(self.all_clusters)

            for i in range(len(self.all_clusters)):
                if used[i]:
                    continue
                base_cluster = self.all_clusters[i]
                base_embs = torch.stack(base_cluster["embeddings"])
                base_avg = base_embs.mean(dim=0)

                for j in range(i + 1, len(self.all_clusters)):
                    if used[j]:
                        continue
                    comp_cluster = self.all_clusters[j]
                    comp_embs = torch.stack(comp_cluster["embeddings"])
                    comp_avg = comp_embs.mean(dim=0)
                    sim = self.norm_cos_sim(base_avg, comp_avg)
                    if sim >= self.threshold:
                        base_cluster["embeddings"].extend(comp_cluster["embeddings"])
                        base_cluster["paths"].extend(comp_cluster["paths"])

                        base_embs = torch.stack(base_cluster["embeddings"])
                        base_avg = base_embs.mean(dim=0)
                        used[j] = True
                        changed = True
                new_clusters.append(base_cluster)
                used[i] = True
            self.all_clusters = new_clusters

        for cluster_id, cluster in enumerate(self.all_clusters):
            cluster['id'] = cluster_id
        
        return self.all_clusters

    def assign_cluster_ids(self):
        cluster_mapping = {}
        for cluster in self.all_clusters:
            for path in cluster["paths"]:
                cluster_mapping[path] = cluster["id"]

        self.full_df["speaker"] = (
            self.full_df["audio_path"]
            .map(cluster_mapping)
            .astype(pd.Int64Dtype())
        )

def init_worker(model_path, device_str):
    global worker_embedder
    worker_embedder = ResNetEmbedder(model_path, device_str)

def compute_embedding_for_file(audio_path):
    try:
        emb, fullness = worker_embedder(audio_path)
        if emb is not None:
            emb_path = audio_path.rsplit('.', 1)[0] + '.emb'
            torch.save((emb, fullness), emb_path)
            return audio_path, fullness
        return None
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {str(e)}")
        return None

def precompute_embeddings(
    audio_files: list,
    model_path: str,
    num_workers_per_gpu: int = 8
) -> dict:
    num_gpus = torch.cuda.device_count()
    available_gpu_ids = list(range(num_gpus))
    
    files_for_each_gpu = [[] for _ in range(num_gpus)]
    for i, path in enumerate(audio_files):
        gpu_assignment_index = i % num_gpus
        files_for_each_gpu[gpu_assignment_index].append(path)
    
    all_futures = []
    executors = []
    fullness_results = {}
    
    logger.info(f"Starting embedding computation on {num_gpus} GPUs with {num_workers_per_gpu} workers per GPU")
    
    for i, gpu_id in enumerate(available_gpu_ids):
        device_str = f'cuda:{gpu_id}'
        files_for_this_gpu = files_for_each_gpu[i]
        
        if not files_for_this_gpu:
            continue
            
        logger.info(f"Creating ProcessPoolExecutor for {device_str} with {num_workers_per_gpu} workers for {len(files_for_this_gpu)} files")

        executor = ProcessPoolExecutor(
            max_workers=num_workers_per_gpu,
            initializer=init_worker,
            initargs=(model_path, device_str)
        )
        executors.append(executor)
        
        for path in files_for_this_gpu:
            future = executor.submit(compute_embedding_for_file, path)
            all_futures.append(future)
    
    logger.info(f"Submitted {len(all_futures)} tasks across {len(executors)} GPUs")
    
    completed_count = 0
    for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="Computing embeddings"):
        try:
            result = future.result()
            if result:
                audio_path, fullness = result
                fullness_results[audio_path] = fullness
                completed_count += 1
        except Exception as e:
            logger.error(f"Task failed: {str(e)}")

    for executor in executors:
        executor.shutdown()
    
    logger.info(f"Completed {completed_count}/{len(all_futures)} embedding computations")
    return fullness_results

def main(args):
    config = load_config(args.config_path, 'classification')
    podcasts_path = config.get('podcasts_path', '../../../podcasts') if args.podcasts_path is None else args.podcasts_path 
    threshold = config.get('threshold', 0.85) if args.threshold is None else args.threshold 
    model_path = config.get('model_path', '/models/voxblink2_samresnet100_ft') if args.model_path is None else args.model_path
    device =  config.get('device', 'cuda') if args.device is None else args.device  
    num_workers =  config.get('num_workers', 8) if args.num_workers is None else args.num_workers  

    logger.info(
        f"""
        Used params:
        podcasts path: {podcasts_path}
        threshold: {threshold}
        emb model path: {model_path}
        device: {device}
        workers per GPU: {num_workers}
        """
    )

    sc = SpeakerClustering(
        podcasts_path=podcasts_path, 
        threshold=threshold,
        model_path=model_path,
        device=device
    )
    
    audio_files = [
        os.path.join(podcasts_path, path)
        for path in sc.df['audio_path'].unique().tolist()
    ]
    
    missing_files = []
    for path in audio_files:
        emb_path = os.path.join(podcasts_path,path.rsplit('.', 1)[0] + '.emb')
        if not os.path.exists(emb_path):
            missing_files.append(path)
    
    if missing_files:
        logger.info(f"Found {len(missing_files)} missing embeddings, starting computation...")
        fullness_results = precompute_embeddings(
            missing_files,
            model_path,
            num_workers_per_gpu=num_workers
        )

        for path, fullness in fullness_results.items():
            relative_path = os.path.relpath(path, podcasts_path)
            mask = sc.full_df['audio_path'] == relative_path
            sc.full_df.loc[mask, 'fullness'] = fullness
      
            mask_df = sc.df['audio_path'] == relative_path
            sc.df.loc[mask_df, 'fullness'] = fullness
        
        updated_csv_path = os.path.join(podcasts_path, 'results.csv')
        sc.full_df.to_csv(updated_csv_path, index=False)
        logger.info(f"Updated results.csv with fullness values")
    else:
        logger.info("All embeddings already precomputed")

    sc.cluster_all_podcast()
    sc.merge_clusters()
    sc.assign_cluster_ids()

    output_path = os.path.join(podcasts_path, 'results.csv')
    sc.full_df.to_csv(output_path, index=False)
    logger.info(f"Clustered DataFrame saved to {output_path}")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Classification speaker")
    parser.add_argument("--config_path", help="Path to the configuration file")
    parser.add_argument("--podcasts_path", help="Path to the podcast folder")
    parser.add_argument("--threshold", type=float, help="Threshold for clustering")
    parser.add_argument("--model_path", type=str, help="embedder model path")
    parser.add_argument("--device", type=str, help="embedder device")
    parser.add_argument("--num_workers", type=int, help="Number of workers per GPU")

    args = parser.parse_args()
    main(args)