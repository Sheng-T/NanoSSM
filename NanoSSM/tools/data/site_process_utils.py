r"""
This module is a collection of functions needed to run m6anet dataprep
"""
import csv
import gc
import os
import multiprocessing
import random
import traceback
from io import StringIO
from itertools import groupby
from collections import defaultdict
from operator import itemgetter
from math import floor, log10
import numpy as np
import pandas as pd
import ujson
from typing import List, Tuple, Dict, Union, Callable
from itertools import product
from sklearn.model_selection import train_test_split

from numba import njit, prange
from tqdm import tqdm

from NanoSSM.tools.common.common_utils import common_log, ERROR

CENTER_MOTIFS = [['A', 'G', 'T'], ['G', 'A'], ['A'], ['C'], ['A', 'C', 'T']]
M6A_KMERS = ["".join(x) for x in product(*CENTER_MOTIFS)]

CENTER_MOTIFS_PSEUDO = [['A', 'C', 'G', 'T'], ['A', 'C', 'G', 'T'], ['T'], ['A', 'C', 'G', 'T'], ['A', 'C', 'G', 'T']]
PSEU_KMERS = ["".join(x) for x in product(*CENTER_MOTIFS_PSEUDO)]


CENTER_MOTIFS_A = [['A', 'C', 'G', 'T'], ['A', 'C', 'G', 'T'], ['A'], ['A', 'C', 'G', 'T'], ['A', 'C', 'G', 'T']]
A_KMERS = ["".join(x) for x in product(*CENTER_MOTIFS_A)]

motif_dic = {
    'm6a': M6A_KMERS,
    'pseu': PSEU_KMERS,
    'a': A_KMERS
}

# -------------------------------------
# index
# -------------------------------------
def parallel_index(eventalign_filepath: str, chunk_size: int, out_dir: str, n_processes: int):

    # Create output paths and locks.
    out_paths, locks = dict(), dict()
    for out_filetype in ['index']:
        out_paths[out_filetype] = os.path.join(out_dir, 'eventalign.%s' % out_filetype)
        locks[out_filetype] = multiprocessing.Lock()
    actual_header = pd.read_csv(eventalign_filepath, nrows=0, sep='\t').columns.tolist()
    read_col = 'read_name' if 'read_name' in actual_header else 'read_index'

    common_log(f"[INDEX] Indexing eventalign.txt file: {eventalign_filepath}")

    with open(out_paths['index'], 'w', encoding='utf-8') as f:
        f.write('transcript_id,read_index,pos_start,pos_end\n')  # header

    task_queue = multiprocessing.JoinableQueue(maxsize=n_processes * 2)

    consumers = [Consumer(task_queue=task_queue, task_function=index, locks=locks) for i in range(n_processes)]
    for process in consumers:
        process.start()

    ## Load tasks into task_queue. A task is eventalign information of one read.
    eventalign_file = open(eventalign_filepath, 'r', encoding='utf-8')
    pos_start = len(eventalign_file.readline())  # remove header
    chunk_split = None
    chunk_counter = 0
    index_features = ['contig', 'read_index', 'line_length']
    for chunk in pd.read_csv(eventalign_filepath, chunksize=chunk_size, sep='\t'):
        if read_col == 'read_name':
            chunk = chunk.rename(columns={'read_name': 'read_index'})
        chunk_counter += 1
        if chunk_counter % 100 == 0:
            common_log(f"[INDEX] Processing chunk #{chunk_counter}, pos_start = {pos_start}")
        chunk_complete = chunk[chunk['read_index'] != chunk.iloc[-1]['read_index']]
        chunk_concat = pd.concat([chunk_split, chunk_complete])
        chunk_concat_size = len(chunk_concat.index)
        ## read the file at where it left off because the file is opened once ##
        lines = [len(eventalign_file.readline()) for i in range(chunk_concat_size)]
        chunk_concat.loc[:, 'line_length'] = np.array(lines)
        task_queue.put((chunk_concat[index_features], pos_start, out_paths))
        pos_start += sum(lines)
        chunk_split = chunk[chunk['read_index'] == chunk.iloc[-1]['read_index']].copy()

    chunk_split_size = len(chunk_split.index)
    lines = [len(eventalign_file.readline()) for i in range(chunk_split_size)]
    chunk_split.loc[:, 'line_length'] = np.array(lines)
    task_queue.put((chunk_split[index_features], pos_start, out_paths))

    task_queue = end_queue(task_queue, n_processes)

    task_queue.join()


def index(eventalign_result: pd.DataFrame, pos_start: int, out_paths: Dict, locks: Dict):

    eventalign_result = eventalign_result.set_index(['contig', 'read_index'])
    pos_end = pos_start
    with locks['index'], open(out_paths['index'], 'a', encoding='utf-8') as f_index:
        for _index in list(dict.fromkeys(eventalign_result.index)):
            transcript_id, read_index = _index
            pos_end += eventalign_result.loc[_index]['line_length'].sum()
            f_index.write('%s,%s,%d,%d\n' % (transcript_id, read_index, pos_start, pos_end))
            pos_start = pos_end


# -------------------------------------
# process
# -------------------------------------

def filter_by_kmer(partition: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                   kmers: List[str], window_size: int) -> \
                        Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], List]:
    feature_arr, kmer_arr, tx_id_arr, tx_pos_arr = partition[:4]

    kmers_5 = kmer_arr[:, (2 * window_size + 1) // 2]
    mask = np.isin(kmers_5, kmers).flatten()

    filtered_feature_arr = feature_arr[mask]

    filtered_kmer_arr = kmer_arr[mask]
    filtered_tx_pos_arr = tx_pos_arr[mask]
    filtered_tx_id_arr = tx_id_arr[mask]
    filtered_read_id_arr = partition[-1][mask]

    if not np.any(mask): 
        return []
    else:
        return filtered_feature_arr,  filtered_kmer_arr, filtered_tx_id_arr, filtered_tx_pos_arr, filtered_read_id_arr


def roll(to_roll: np.ndarray, window_size: int) -> np.ndarray:
    nex = np.concatenate([np.roll(to_roll, i, axis=0) for i in range(-1, - window_size - 1, -1)],
                         axis=1)
    prev = np.concatenate([np.roll(to_roll, i, axis=0) for i in range(window_size, 0, -1)], axis=1)
    combined = np.concatenate((prev, to_roll, nex), axis=1)[window_size: -window_size, :]
    
    return combined

def roll_window_stack(to_roll: np.ndarray, window_size: int) -> np.ndarray:

    N, D = to_roll.shape
    total_window = 2 * window_size + 1

    if np.issubdtype(to_roll.dtype, np.number):
        fill_val = 0.0
    else:
        fill_val = "N" * 5 


    padded = np.pad(to_roll, ((window_size, window_size), (0, 0)),
                    mode='constant', constant_values=fill_val)
    stacked = np.stack([padded[i: i + total_window] for i in range(N)], axis=0)

    return stacked


def create_windowed_features(partition: Tuple[np.ndarray, np.ndarray,np.ndarray, np.ndarray, np.ndarray],
                             window_size: int) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    float_arr, kmer_arr, tx_id_arr, tx_pos_arr = partition[:4]

    windowed_results = (
        roll_window_stack(float_arr, window_size),  # (N, 2w+1, 5)
        roll_window_stack(kmer_arr, window_size),  # (N, 2w+1, 1)
        tx_id_arr,  # (N,)
        tx_pos_arr,  # (N,)
        partition[-1]  # (N,) read_idx
    )

    return windowed_results


def process_partitions(partitions: List[Tuple[np.ndarray, np.ndarray,np.ndarray, np.ndarray, np.ndarray]],
                       window_size: int, kmers: List[str]):

    windowed_partition = [create_windowed_features(partition, window_size) for partition in partitions]
    filtered_by_kmers = [filter_by_kmer(partition, kmers, window_size) for partition in windowed_partition]
    final_partitions = [x for x in filtered_by_kmers if len(x) > 0]

    return final_partitions


def partition_into_continuous_positions(arr: np.recarray, window_size: int) -> List[Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]]:
    if arr.size == 0:
        return []

    order = np.argsort(arr["transcriptomic_position"])
    arr_sorted = arr[order]
    pos = arr_sorted["transcriptomic_position"]

    diff = np.diff(pos)
    break_points = np.where(diff != 1)[0]  
    starts = np.concatenate(([0], break_points + 1))
    ends   = np.concatenate((break_points + 1, [arr.size]))


    min_len = 3 
    valid = (ends - starts) >= min_len
    starts = starts[valid]
    ends   = ends[valid]

    if len(starts) == 0:
        return []

    float_features = np.column_stack([arr_sorted[f] for f in ('dwell_time', 'norm_std', 'norm_mean', 'cv', 'std_level')])
    kmers   = arr_sorted["reference_kmer"].reshape(-1, 1)
    tx_id   = arr_sorted["transcript_id"]
    tx_pos  = pos
    read_idx = arr_sorted["read_index"]

    partitions = [
        (
            float_features[s:e],
            # samples[s:e],
            kmers[s:e],
            tx_id[s:e],
            tx_pos[s:e],
            read_idx[s:e]
        )
        for s, e in zip(starts, ends)
    ]
    return partitions


def filter_events(events: np.recarray, window_size: int, kmers: List[str]) -> \
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:

    events = partition_into_continuous_positions(events, window_size)
    events = process_partitions(events, window_size, kmers)

    return events


def combine_sequence(kmers: List[str]) -> str:
    kmer = kmers[0]
    for _kmer in kmers[1:]:
        kmer += _kmer[-1]
    return kmer



def combine(events_str: str, max_signal_len: int, seed: int = 42) -> np.recarray:
    f_string = StringIO(events_str)
    eventalign_result = pd.read_csv(f_string, delimiter='\t',
                                    names=['contig', 'position', 'reference_kmer', 'read_index', 'strand',
                                           'event_index',
                                           'event_level_mean', 'event_stdv', 'event_length', 'model_kmer', 'model_mean',
                                           # 'model_stdv', 'standardized_level', 'start_idx', 'end_idx', 'samples'])
                                           'model_stdv', 'standardized_level', 'start_idx', 'end_idx'])
    eventalign_result = eventalign_result[['contig', 'position', 'reference_kmer', 'read_index','event_level_mean',
                                           # 'event_stdv', 'event_length', 'model_kmer','standardized_level', 'start_idx', 'end_idx', 'samples']].copy()
                                           'event_stdv', 'event_length', 'model_kmer','standardized_level', 'start_idx', 'end_idx']].copy()
    f_string.close()
    cond_successfully_eventaligned = eventalign_result['reference_kmer'] == eventalign_result['model_kmer']

    if cond_successfully_eventaligned.sum() != 0:

        eventalign_result = eventalign_result[cond_successfully_eventaligned]

        keys = ['read_index','contig','position','reference_kmer'] # for groupby
        eventalign_result.loc[:, 'length'] = pd.to_numeric(eventalign_result['end_idx']) - \
                pd.to_numeric(eventalign_result['start_idx'])
        eventalign_result.loc[:, 'sum_norm_mean'] = pd.to_numeric(eventalign_result['event_level_mean']) \
                * eventalign_result['length']
        eventalign_result.loc[:, 'sum_norm_std'] = pd.to_numeric(eventalign_result['event_stdv']) \
                * eventalign_result['length']
        eventalign_result.loc[:, 'sum_dwell_time'] = pd.to_numeric(eventalign_result['event_length']) \
                * eventalign_result['length']
        eventalign_result.loc[:,'sum_cv'] = (pd.to_numeric(eventalign_result['event_stdv']) / pd.to_numeric(eventalign_result['event_level_mean'])) \
                                      * eventalign_result['length']
        eventalign_result.loc[:,'sum_std_level'] = pd.to_numeric(eventalign_result['standardized_level']) \
                                             * eventalign_result['length']



        grouped = eventalign_result.groupby(keys).agg({
            'sum_norm_mean': 'sum',
            'sum_norm_std': 'sum',
            'sum_dwell_time': 'sum',
            'sum_cv': 'sum',
            'sum_std_level': 'sum',
            'start_idx': 'min',
            'end_idx': 'max',
            'length': 'sum',
            # 'samples': list 
        }).reset_index()


        eventalign_result = pd.concat([grouped['start_idx'],grouped['end_idx']], axis=1)
        eventalign_result['norm_mean'] = (grouped['sum_norm_mean']/grouped['length']).round(1)
        eventalign_result["norm_std"] = grouped['sum_norm_std'] / grouped['length']
        eventalign_result["dwell_time"] = grouped['sum_dwell_time'] / grouped['length']
        # eventalign_result["mean_dev"] = sum_mean_dev / total_length
        eventalign_result["cv"] = grouped['sum_cv'] / grouped['length']
        eventalign_result["std_level"] = grouped['sum_std_level'] / grouped['length']
        # eventalign_result['samples'] = process_samples_column(grouped['samples'].values, 5)

        eventalign_result['transcript_id'] = grouped['contig']    #### CHANGE MADE ####
        eventalign_result['transcriptomic_position'] = \
                pd.to_numeric(grouped['position']) + 2 # the middle position of 5-mers.

        eventalign_result['reference_kmer'] = grouped['reference_kmer']
        eventalign_result['read_index'] = grouped['read_index']

        features = ['transcript_id', 'read_index',
                    'transcriptomic_position', 'reference_kmer',
                    # 'norm_mean', 'norm_std', 'dwell_time',  'cv', 'std_level', 'samples']
                    'norm_mean', 'norm_std', 'dwell_time',  'cv', 'std_level']
        df_events = eventalign_result.loc[:, features].copy()
        np_events = np.rec.fromrecords(df_events, names=[*df_events])
        return np_events

    return np.array([])



def parallel_preprocess_tx(eventalign_filepath: str, out_dir: str, n_processes: int, readcount_min: int,
                           readcount_max: int, n_neighbors: int, min_segment_count: int, compress: bool,
                           max_signal_len: int, seed: int = 42, motif: str = 'm6a',):
    # Create output paths and locks.
    common_log(f'Preparing to process eventalign.txt...')
    out_paths, locks = dict(), dict()

    common_log(f'Preparing output files[json/info/log]...')
    for out_filetype in ['json', 'info', 'log']:
        out_paths[out_filetype] = os.path.join(out_dir, 'data.%s' % out_filetype)
        locks[out_filetype] = multiprocessing.Lock()

    # Writing the starting of the files.

    open(out_paths['json'], 'w', encoding='utf-8').close()

    with open(out_paths['info'], 'w', encoding='utf-8') as f:
        f.write('transcript_id,transcript_position,motif,start,end,n_reads\n')  # header

    open(out_paths['log'], 'w', encoding='utf-8').close()

    # Create communication queues.
    common_log(f'Creating {n_processes} consumers...')
    task_queue = multiprocessing.JoinableQueue(maxsize=n_processes * 2)

    # Create and start consumers.
    consumers = [Consumer(task_queue=task_queue, task_function=preprocess_tx, locks=locks) for i in range(n_processes)]

    for process in consumers:
        process.start()
    df_eventalign_index = pd.read_csv(os.path.join(out_dir, 'eventalign.index'))
    df_eventalign_index['transcript_id'] = df_eventalign_index['transcript_id']
    tx_ids = df_eventalign_index['transcript_id'].values.tolist()
    tx_ids = list(dict.fromkeys(tx_ids))
    df_eventalign_index = df_eventalign_index.set_index('transcript_id')
    total_count = len(df_eventalign_index)
    total_tx = len(tx_ids)
    common_log(f'Processing {total_tx} transcripts, total reads count: {total_count}...')

    common_log(f'motif is {motif}, motif mask is {motif_dic[motif]}')
    with open(eventalign_filepath, 'r', encoding='utf-8') as eventalign_result:
        tx_count = 0
        for tx_id in tx_ids:
            data_dict = dict()
            readcount = 0
            tx_count += 1
            if tx_count % 100 == 0:
                percent = tx_count / total_tx * 100
                common_log(f'Processing {tx_count}/{total_tx} transcripts... ({percent:.2f}%)')

            for _, row in df_eventalign_index.loc[[tx_id]].iterrows():
                read_index, pos_start, pos_end = row['read_index'], row['pos_start'], row['pos_end']
                eventalign_result.seek(pos_start, 0)
                chunk_size = pos_end - pos_start
                events_str = eventalign_result.read(pos_end - pos_start)
                try:
                    data = combine(events_str, max_signal_len, seed)
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    common_log(f"combine failed: {e}, chunk_size: {chunk_size}, traceback: {stack_trace}", ERROR)
                if data.size > 1:
                    data_dict[read_index] = data
                readcount += 1
                if readcount > readcount_max:
                    break
            if readcount >= readcount_min:
                task_queue.put((tx_id, data_dict, n_neighbors, min_segment_count, out_paths, compress, motif))

    task_queue = end_queue(task_queue, n_processes)

    task_queue.join()
    common_log(f'Finished processing {len(tx_ids)} transcripts.')


def preprocess_tx(tx_id: str, data_dict: Dict, n_neighbors: int, min_segment_count: int, out_paths: Dict,
                  compress: bool, motif: str, locks: Dict):
    if len(data_dict) == 0:
        return

    features_arrays = []
    # signals_arrays = []
    reference_kmer_arrays = []
    reference_kmer_arrays_encoded = []
    transcriptomic_positions_arrays = []
    read_ids = []
    motif_mask = motif_dic[motif]

    for _, events_per_read in data_dict.items():

        events_per_read = filter_events(events_per_read, n_neighbors, motif_mask)
        for event_per_read in events_per_read:

            features_arrays.append(event_per_read[0])

            center_kmers = event_per_read[1][:, n_neighbors]
            reference_kmer_arrays.append(center_kmers.flatten())

            encoded_seq_list = []
            for read_window in event_per_read[1]:
                current_read_seq = []

                for kmer_item in read_window.flatten():
                    if isinstance(kmer_item, str) and len(kmer_item) > 0:

                        encoded_kmer = [alphabet.get(base, 0) for base in kmer_item.upper()]
                        current_read_seq.append(encoded_kmer)
                    else:

                        current_read_seq.append([0, 0, 0, 0, 0])

                encoded_seq_list.append(current_read_seq)

            reference_kmer_arrays_encoded.append(encoded_seq_list)

            transcriptomic_positions_arrays.append(event_per_read[3])
            read_ids.append(event_per_read[4])

    if len(features_arrays) == 0:
        return

    features_arrays = np.concatenate(features_arrays)
    # signals_arrays = np.concatenate(signals_arrays)
    reference_kmer_arrays = np.concatenate(reference_kmer_arrays)
    reference_kmer_arrays_encoded = np.concatenate(reference_kmer_arrays_encoded)
    transcriptomic_positions_arrays = np.concatenate(transcriptomic_positions_arrays)
    read_ids = np.concatenate(read_ids)

    assert (len(features_arrays) == len(reference_kmer_arrays) == \
            len(transcriptomic_positions_arrays) == len(read_ids))

    idx_sorted = np.argsort(transcriptomic_positions_arrays)
    positions, indices = np.unique(transcriptomic_positions_arrays[idx_sorted],
                                   return_index=True, axis=0)  # 'chr',

    features_arrays = np.split(features_arrays[idx_sorted], indices[1:])
    # signals_arrays = np.split(signals_arrays[idx_sorted], indices[1:])
    reference_kmer_arrays_encoded = np.split(reference_kmer_arrays_encoded[idx_sorted], indices[1:])
    reference_kmer_arrays = np.split(reference_kmer_arrays[idx_sorted], indices[1:])
    read_ids = np.split(read_ids[idx_sorted], indices[1:])

    # Prepare
    data = defaultdict(dict)
    motif = defaultdict(dict)

    for position, features_array, reference_kmer_array, reference_kmer_encoded, read_id in \
            zip(positions, features_arrays, reference_kmer_arrays, reference_kmer_arrays_encoded, read_ids):

        kmer = set(reference_kmer_array)

        if compress:
            features_array = features_array.round(decimals=3)


        if len(kmer) > 1:
            log_str = f"WARNING: Position {tx_id}:{position} skipped due to inconsistent K-mers: {kmer}"

            with locks['log'], open(out_paths['log'], 'a', encoding='utf-8') as f:
                f.write(log_str + '\n')
            continue 

        if (len(set(reference_kmer_array)) == 1) and ('NNNNN' in set(reference_kmer_array)) \
                or (len(features_array) == 0):
            continue
        if len(features_array) >= min_segment_count and len(reference_kmer_encoded) == len(features_array):
            kmer_str = kmer.pop()
            data[int(position)] = {kmer_str: {'seq': reference_kmer_encoded.tolist(), 'stat': features_array.tolist()},
                                   'read_id': read_id.tolist()}
            motif[int(position)] = kmer_str
    # write to file.
    log_str = '%s: Data preparation ... Done.' % (tx_id)
    with locks['json'], open(out_paths['json'], 'a', encoding='utf-8') as f:
        with locks['info'], open(out_paths['info'], 'a', encoding='utf-8') as g:
            for pos, dat in data.items():
                pos_start = f.tell()
                f.write('{')
                f.write('"%s":{"%d":' % (tx_id, pos))
                ujson.dump(dat, f)
                f.write('}}\n')
                pos_end = f.tell()
                n_reads = 0
                for kmer, features in dat.items():
                    n_reads += len(features)
                g.write('%s,%d,%s,%d,%d,%d\n' % (tx_id, pos, motif[pos], pos_start, pos_end, n_reads))

    with locks['log'], open(out_paths['log'], 'a', encoding='utf-8') as f:
        f.write(log_str + '\n')

    del data_dict
    del features_arrays, reference_kmer_arrays, reference_kmer_arrays_encoded, transcriptomic_positions_arrays, read_ids
    gc.collect()


# -------------------------
# utils
# -------------------------
alphabet = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4}

def combine_wrapper(args):
    events_str, max_signal_len, seed = args
    try:
        return combine(events_str, max_signal_len, seed)
    except Exception as e:
        return None


def str_to_float_array(s):
    return np.array([float(x) for x in s.split(',')])


def process_signals_numpy(signals: List[str], max_len: int, seed=42) -> np.ndarray:

    arrays = [np.fromstring(s, sep=',') for s in signals]

    result = np.zeros((len(arrays), max_len), dtype=np.float32)

    for i, arr in enumerate(arrays):
        n = len(arr)

        if n >= max_len:

            start_idx = (n - max_len) // 2
            end_idx = start_idx + max_len

            result[i] = arr[start_idx:end_idx]

        else:
            result[i, :n] = arr

    return result

def process_samples_column(data, decimals=5):
    results = []
    fmt = f"{{:.{decimals}f}}".format 
    for row in data:
        arr = np.asarray(row)  # shape: (n_reads, signal_len)

        mean_arr = arr.mean(axis=0)

        results.append(",".join(map(fmt, mean_arr)))

    return np.array(results, dtype=object)




def split_samples_arr(
        samples_list: List[str],
) -> np.ndarray:
    out = []
    for samples in samples_list:
        out.append(np.array([float(x) for x in samples.split(',')]))

    return np.stack(out, axis=0).astype(float)


@njit
def encode_sequence_numba(seq):
    out = np.empty(len(seq), dtype=np.uint8)
    for i, c in enumerate(seq):
        if c == 'A':
            out[i] = 1
        elif c == 'C':
            out[i] = 2
        elif c == 'G':
            out[i] = 3
        elif c == 'T':
            out[i] = 4
        else:
            out[i] = 0
    return out


def encode_sequence_numba_array(seq_array):
    return np.array([encode_sequence_numba(seq) for seq in seq_array])


class Consumer(multiprocessing.Process):
    """ For parallelisation """

    def __init__(self, task_queue, task_function, locks=None, result_queue=None):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.locks = locks
        self.task_function = task_function
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task_args = self.task_queue.get()
            if next_task_args is None:
                self.task_queue.task_done()
                break
            result = self.task_function(*next_task_args, self.locks)
            self.task_queue.task_done()
            if self.result_queue is not None:
                self.result_queue.put(result)


def end_queue(task_queue, n_processes):
    for _ in range(n_processes):
        task_queue.put(None)
    return task_queue


def load_bed_to_keys(bed_path: str, ratio_col: int = 8) -> dict:
    bed_dict = {}
    with open(bed_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            chrom, start = parts[0], int(parts[1])
            ratio = float(parts[ratio_col])
            bed_dict[(chrom, start)] = ratio
    return bed_dict





