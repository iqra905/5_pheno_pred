import gzip
import os
import itertools
import argparse
from multiprocessing import Pool, cpu_count
import logging
from typing import Set, Tuple, List, Dict, Generator
from pathlib import Path
from contextlib import contextmanager
import resource
import time
import gc
import io


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gwas_processing.log')
    ]
)
logger = logging.getLogger(__name__)

class ProcessingConfig:
    """Configuration class to store and validate processing parameters."""
    def __init__(self, 
                 input_folder: str,
                 output_folder: str,
                 reference_file: str,
                 final_output: str,
                 chunk_size: int,
                 n_processes: int):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.reference_file = Path(reference_file)
        self.final_output = Path(final_output)
        self.chunk_size = chunk_size
        self.n_processes = n_processes
        
        # Validate paths
        self._validate_paths()
        
    def _validate_paths(self):
        """Validate input paths exist and output paths can be created."""
        if not self.input_folder.exists():
            raise ValueError(f"Input folder does not exist: {self.input_folder}")
        if not self.reference_file.exists():
            raise ValueError(f"Reference file does not exist: {self.reference_file}")
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.final_output.parent.mkdir(parents=True, exist_ok=True)

@contextmanager
def memory_tracker(description: str):
    """Context manager to track memory usage of a code block."""
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mem_diff = end_mem - start_mem
        time_diff = end_time - start_time
        logger.info(f"{description} - Memory delta: {mem_diff/1024:.2f}MB, Time: {time_diff:.2f}s")

def read_reference_file(filename: Path, chunk_size: int = 100000) -> Set[Tuple[str, ...]]:
    """
    Read reference SNPs with explicit memory management.
    """
    reference_set = set()
    logger.info(f"Reading reference file: {filename}")
    
    try:
        with open(filename, 'r') as f:
            chunk = []
            while True:
                # Read chunk_size lines
                new_lines = list(itertools.islice(f, chunk_size))
                if not new_lines:
                    break
                
                # Process chunk
                for line in new_lines:
                    reference_set.add(tuple(line.split()[:5]))
                
                # Clear chunk explicitly
                new_lines.clear()
                del new_lines
                gc.collect()
                
                # Log memory usage periodically
                if len(reference_set) % (chunk_size * 10) == 0:
                    logger.info(f"Loaded {len(reference_set)} SNPs. "
                              f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f}MB")
        
        logger.info(f"Loaded {len(reference_set)} reference SNPs")
        return reference_set
        
    except Exception as e:
        logger.error(f"Error reading reference file: {str(e)}")
        raise


def process_chunk_generator(chunk: List[str], reference_set: Set[Tuple[str, ...]]) -> Generator[str, None, None]:
    """
    Process chunk using generator to avoid holding entire filtered chunk in memory.
    """
    try:
        for line in chunk:
            if tuple(line.split()[:5]) in reference_set:
                yield line
    finally:
        # Clear the chunk explicitly
        chunk.clear()
        del chunk
        gc.collect()

def process_single_file(args: Tuple[Path, Path, Set[Tuple[str, ...]], int]) -> Dict:
    """
    Process a single .gen.gz file with explicit memory management.
    """
    input_file, output_file, reference_set, chunk_size = args
    stats = {
        'input_file': str(input_file),
        'lines_processed': 0,
        'lines_written': 0,
        'success': False
    }
    
    logger.info(f"Processing {input_file}")
    
    try:
        with gzip.open(input_file, 'rt') as in_f, gzip.open(output_file, 'wt') as out_f:
            chunk = []
            lines_written = 0
            
            for line_count, line in enumerate(in_f, 1):
                chunk.append(line)
                
                if len(chunk) >= chunk_size:
                    # Process chunk using generator
                    for filtered_line in process_chunk_generator(chunk, reference_set):
                        out_f.write(filtered_line)
                        lines_written += 1
                    
                    # Create new chunk list
                    chunk = []
                    
                    # Force garbage collection after processing large chunks
                    if line_count % (chunk_size * 10) == 0:
                        gc.collect()
                        logger.info(f"Processed {line_count} lines in {input_file}. "
                                  f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f}MB")
            
            # Process remaining lines
            if chunk:
                for filtered_line in process_chunk_generator(chunk, reference_set):
                    out_f.write(filtered_line)
                    lines_written += 1
            
            stats.update({
                'lines_processed': line_count,
                'lines_written': lines_written,
                'success': True
            })
        
        logger.info(f"Successfully processed {input_file}: "
                   f"{stats['lines_written']}/{stats['lines_processed']} lines written")
        return stats
    
    except Exception as e:
        logger.error(f"Error processing {input_file}: {str(e)}")
        stats['error'] = str(e)
        return stats
    finally:
        # Ensure cleanup
        gc.collect()

# def merge_filtered_files(filtered_folder: Path, output_file: Path, chunk_size: int = 100000) -> bool:
#     """
#     Merge filtered files with explicit memory management.
#     """
#     logger.info("Starting file merge process")
    
#     try:
#         gen_files = sorted(
#             [f for f in filtered_folder.glob('*.gen.gz')],
#             key=lambda x: custom_sort(x.name)
#         )
        
#         with gzip.open(output_file, 'wt') as out_f:
#             for gen_file in gen_files:
#                 logger.info(f"Merging file: {gen_file}")
#                 with gzip.open(gen_file, 'rt') as in_f:
#                     chunk = []
#                     while True:
#                         new_lines = list(itertools.islice(in_f, chunk_size))
#                         if not new_lines:
#                             break
                        
#                         # Write chunk
#                         out_f.writelines(new_lines)
                        
#                         # Clear chunk explicitly
#                         new_lines.clear()
#                         del new_lines
#                         gc.collect()
                
#                 logger.info(f"Memory usage after merging {gen_file}: "
#                           f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f}MB")
        
#         logger.info("Merge completed successfully")
#         return True
        
#     except Exception as e:
#         logger.error(f"Error during merge: {str(e)}")
#         return False
#     finally:
#         # Ensure cleanup
#         gc.collect()

def merge_filtered_files(filtered_folder: Path, output_file: Path, lines_per_chunk: int = 5000) -> bool:
    logger.info(f"Starting file merge process with {lines_per_chunk:,} lines per chunk")
    
    try:
        # Sort files once at the start
        gen_files = sorted(
            [f for f in filtered_folder.glob('*.gen.gz')],
            key=lambda x: custom_sort(x.name)
        )

        total_files = len(gen_files)
        if total_files == 0:
            logger.warning("No files found to merge")
            return False
            
        logger.info(f"Found {total_files} files to merge")
        
        # Use a larger buffer for writing
        with gzip.open(output_file, 'wt') as out_f:
            total_chunks_processed = 0
            total_lines_processed = 0

            for file_num, gen_file in enumerate(gen_files, 1):
                logger.info(f"Merging file {file_num}/{total_files}: {gen_file}")
                chunk_count = 0
                file_lines = 0
                
                # Read and write in larger chunks
                with gzip.open(gen_file, 'rt') as in_f:
                    while True:
                        # Efficiently read chunks using itertools.islice
                        chunk = list(itertools.islice(in_f, lines_per_chunk))
                        if not chunk:
                            break
                            
                        chunk_count += 1
                        chunk_line_count = len(chunk)
                        file_lines += chunk_line_count
                        
                        # Log chunk processing
                        logger.info(f"File {file_num}/{total_files} - "
                                  f"Processing chunk {chunk_count} "
                                  f"(lines: {chunk_line_count:,})")
                        
                        # Write the chunk directly
                        out_f.writelines(chunk)
                
                total_chunks_processed += chunk_count
                total_lines_processed += file_lines
                
                # Log completion of each file
                logger.info(f"Completed file {file_num}/{total_files} - "
                          f"Processed {chunk_count:,} chunks, "
                          f"{file_lines:,} lines")
        
        # Log final statistics
        logger.info(f"Merge completed successfully - "
                   f"Total chunks: {total_chunks_processed:,}, "
                   f"Total lines: {total_lines_processed:,}")
        return True
        
    except Exception as e:
        logger.error(f"Error during merge: {str(e)}")
        return False
        
def custom_sort(file_name: str) -> Tuple[int, str]:
    """Sort files by chromosome number."""
    parts = file_name.split('_')
    try:
        chr_num = int(parts[0][3:])  # Extract number after 'chr'
    except ValueError:
        chr_num = float('inf')
    return (chr_num, file_name)

def main(args: argparse.Namespace) -> None:
    """
    Main function to coordinate parallel processing and merging with memory optimization.
    """
    try:
        # Initialize configuration
        config = ProcessingConfig(
            args.input_folder,
            args.output_folder,
            args.reference_file,
            args.final_output,
            args.chunk_size,
            args.n_processes if args.n_processes else max(1, cpu_count() - 1)
        )
        
        logger.info("Starting GWAS data processing")
        
        # logger.info(f"Using {config.n_processes} processes")
        
        # # Read reference file
        # with memory_tracker("Reference set creation"):
        #     reference_set = read_reference_file(config.reference_file)
        
        # # Get list of files to process
        # gen_files = sorted(
        #     [f for f in config.input_folder.glob('*.gen.gz')],
        #     key=lambda x: custom_sort(x.name)
        # )
        # #gen_files = list(config.input_folder.glob('*.gen.gz'))
        # if not gen_files:
        #     logger.error(f"No .gen.gz files found in {config.input_folder}")
        #     return
        
        # # Prepare arguments for parallel processing
        # process_args = [
        #     (
        #         input_path,
        #         config.output_folder / input_path.name,
        #         reference_set,
        #         config.chunk_size
        #     )
        #     for input_path in gen_files
        # ]
        
        # # Process files in parallel
        # with Pool(processes=config.n_processes) as pool:
        #     results = pool.map(process_single_file, process_args)
        
        # # Analyze results
        # successful = all(result['success'] for result in results)
        # total_processed = sum(result['lines_processed'] for result in results)
        # total_written = sum(result['lines_written'] for result in results)
        
        # logger.info(f"Processing complete. "
        #            f"Total lines processed: {total_processed}, "
        #            f"Total lines written: {total_written}")
        
        # # Merge if all processing was successful
        # if successful:
        #     merge_success = merge_filtered_files(
        #         config.output_folder,
        #         config.final_output,
        #         config.chunk_size
        #     )
        #     if merge_success:
        #         logger.info(f"Final merged file created: {config.final_output}")
        #     else:
        #         logger.error("Merge process failed")
        # else:
        #     logger.error("Some files failed to process")

        merge_success = merge_filtered_files(config.output_folder, config.final_output, config.chunk_size)
        if merge_success:
            logger.info(f"Final merged file created: {config.final_output}")
        else:
            logger.error("Merge process failed")

        
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and merge .gen.gz files")
    parser.add_argument('-input_folder', type=str, 
                       default='/vol/research/ucdatasets/gwas/gwas_mono_rm',
                       help='Path to input folder')
    parser.add_argument('-output_folder', type=str,
                       default='/vol/research/ucdatasets/gwas/gwas_mono_rm/gen_data_5M',
                       help='Path to output folder for filtered files')
    parser.add_argument('-reference_file', type=str,
                       default='/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_unq.gen',
                       help='Path to reference file')
    parser.add_argument('-final_output', type=str,
                       default='/vol/research/ucdatasets/gwas/gwas_mono_rm/gen_data_5M_merged/5D_merged.gen.gz',
                       help='Path to final merged file')
    parser.add_argument('-chunk_size', type=int, default=5000,
                       help='Number of SNPs to process at a time')
    parser.add_argument('-n_processes', type=int, default=15,
                       help='Number of processes to use')
    
    args = parser.parse_args()
    main(args)