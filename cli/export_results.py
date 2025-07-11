import itertools
import logging
from pathlib import Path
from typing import Any, Dict, List
import argparse

import ijson

from data_processing.data_analysis import create_table_plots



class Results:
    def __init__(self, subparser : argparse.ArgumentParser):
        self.BENCHMARKED_DIR_NAME = 'benchmarked'
        self.CURRENT_PATH = Path(__file__).resolve()
        self.BENCHMARKED_DIR = self.CURRENT_PATH.parent.parent / self.BENCHMARKED_DIR_NAME
        subparser.add_argument(
            '--output_dir',
            type=str,
            default='results_compare',
            help='The directory name to save the generated charts and tables.'
        )
        # Set the function to be executed when this command is chosen
        subparser.set_defaults(func=self.run)

        
    def find_available_gpus(self, benchmarked_dir: Path) -> List[Dict[str, Any]]:
        """
        Scans the benchmarked directory to find all available GPUs and their device IDs.
        """
        gpus_available = []
        unique_gpus = set()
        for trace_file in benchmarked_dir.glob('*/*/*/*.json'):
            try:
                gpu_folder = trace_file.parents[3]

                with open(trace_file, 'r') as f:
                    parser = ijson.parse(f)
                    for prefix, event, value in parser:
                        if prefix == 'benchmark_info.device_id':
                            gpu_info = (str(gpu_folder), value)
                            if gpu_info not in unique_gpus:
                                unique_gpus.add(gpu_info)
                                gpus_available.append({'path': str(gpu_folder), 'device_id': value})
                    
                            break
            except Exception as e:
                logging.warning(f"Could not parse {trace_file}: {e}")
                continue

        if not benchmarked_dir.exists():
            logging.error(f"Benchmarked directory not found at: {benchmarked_dir}")
            return []

        return gpus_available

    def find_common_items(self, selected_items: List[Dict[str, Any]], level: str, previous_selections: dict[str, List[str]] = {}) -> List[str]:
        """
        A generic function to find common scripts, models, or parameters.
        """
        if not selected_items:
            return []

        common_items: set[str] = set()
    
        first_gpu_path = Path(selected_items[0]['path'])
    
        initial_items = set()
    
        for item_path in first_gpu_path.rglob('*'):
            if item_path.is_dir():
                try:
                    if level == 'scripts':
                        if item_path.parent == first_gpu_path:
                            initial_items.add(item_path.name)
                    elif level == 'models':
                         if previous_selections and item_path.parent.name in previous_selections.get('scripts', []):
                            initial_items.add(item_path.name)
                    elif level == 'params':
                        if previous_selections and item_path.parent.name in previous_selections.get('models', []):
                             initial_items.add(item_path.name)

                except IndexError:
                    continue
        common_items = initial_items

        for gpu in selected_items[1:]:
            gpu_path = Path(gpu['path'])
            current_gpu_items = set()
            for item_path in gpu_path.rglob('*'):
                 if item_path.is_dir():
                    try:
                        if level == 'scripts' and item_path.parent == gpu_path:
                            current_gpu_items.add(item_path.name)
                        elif level == 'models' and previous_selections and item_path.parent.name in previous_selections.get('scripts', []):
                            current_gpu_items.add(item_path.name)
                        elif level == 'params' and previous_selections and item_path.parent.name in previous_selections.get('models', []):
                            current_gpu_items.add(item_path.name)

                    except IndexError:
                        continue

            common_items.intersection_update(current_gpu_items)

        return sorted(list(common_items))

    def find_all_trace_files(self, selected_gpus: List[Dict[str, Any]], common_scripts: List[str], common_models: List[str], common_params: List[str]) -> List[str]:
        """
        Gathers all trace file paths that match the selection criteria.
        """
        trace_files = []
        
        path_combinations = itertools.product(selected_gpus, common_scripts, common_models, common_params)

        for gpu_info, script_name, model_name, param_name in path_combinations:
            # python magic :)
            param_path = Path(gpu_info['path']) / script_name / model_name / param_name
    
            if not param_path.is_dir():
                continue

            for trace_file in param_path.glob('*.json'):
                try:
                    with open(trace_file, 'r') as f:
                        if any(prefix == 'benchmark_info.device_id' and value == gpu_info['device_id'] 
                            for prefix, _, value in ijson.parse(f)):
                            trace_files.append(str(trace_file))
                except Exception:
                    continue

    
        return trace_files


    def run(self, args: argparse.Namespace):
        """
        Entry point for analyzing results and generating charts.

        Args:
            args: The parsed command-line arguments.
        """
        self.BENCHMARKED_DIR.mkdir(exist_ok=True)
        all_gpus = self.find_available_gpus(self.BENCHMARKED_DIR)
        if not all_gpus:
            logging.info("No benchmarked GPUs found. Exiting.")
            return

        # The rest of the result analysis logic remains the same...
        selections: Dict[str, List[str]] = {'scripts': [], 'models': []}
        common_scripts = self.find_common_items(all_gpus, 'scripts')
        selections['scripts'] = common_scripts
        common_models = self.find_common_items(all_gpus, 'models', selections)
        selections['models'] = common_models
        common_params = self.find_common_items(all_gpus, 'params', selections)

        trace_files = self.find_all_trace_files(all_gpus, common_scripts, common_models, common_params)
        if not trace_files:
            logging.error("No trace files found for the common criteria.")
            return
        
        logging.info(f"Generating tables and plots from {len(trace_files)} files...")
        table_string, graph_dir = create_table_plots(trace_files, args.output_dir)

        print("\n" + "="*80)
        print(" " * 25 + "Benchmark Results Summary")
        print("="*80)
        print(f'Graphs & CSVs are available in: {graph_dir}\n')
        print(table_string)
        print("="*80)

