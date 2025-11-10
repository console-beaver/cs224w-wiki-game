import os
import multiprocessing as mp
from tqdm import tqdm

def process_chunk(file_name, start, end, show_progress=False):
    partial = {}
    with open(file_name, 'r') as f:
        f.seek(start)
        pos = start
        if start != 0:
            f.readline()
            pos = f.tell()

        pbar = None
        if show_progress:
            total_bytes = end - pos
            pbar = tqdm(total=total_bytes, unit='B', unit_scale=True, desc='        processing chunk')

        while pos < end:
            line = f.readline()
            if not line:
                break
            line_size = len(line.encode('utf-8'))
            pos = f.tell()
            line = line.strip()
            if not line or line[0] == '#':
                if pbar:
                    pbar.update(line_size)
                continue
            u, v = line.split()
            u, v = int(u), int(v)
            if u not in partial:
                partial[u] = {v}
            else:
                partial[u].add(v)
            if pbar:
                pbar.update(line_size)

        if pbar:
            pbar.close()
    return partial

def merge_dicts(dict_list):
    out = {}
    for d in tqdm(dict_list, desc='        merging'):
        for u, neighbors in d.items():
            if u not in out:
                out[u] = neighbors
            else:
                out[u].update(neighbors)
    return out

def build_dict_parallel(filepath, tab=0):
    file_size = os.path.getsize(filepath)
    cpu_count = mp.cpu_count()
    chunk_size = file_size // cpu_count

    chunks = []
    with open(filepath, 'r') as f:
        start = 0
        for i in range(cpu_count):
            end = start + chunk_size
            if i == cpu_count - 1:
                end = file_size
            else:
                f.seek(end)
                f.readline()  # move to next line to avoid breaking lines
                end = f.tell()
            chunks.append((filepath, start, end, i == 0))
            start = end

    print('\t' * (tab+1) + f'processing {len(chunks)} chunks in parallel...')
    with mp.Pool(cpu_count) as pool:
        results = pool.starmap(process_chunk, chunks)

    print('\t' * (tab+1) + 'merging chunks...')
    return merge_dicts(results)

if __name__ == '__main__':
    filepath = 'graph_edges.txt'
    graph = build_dict_parallel(filepath)
    print(f"Total nodes: {len(graph)}")

