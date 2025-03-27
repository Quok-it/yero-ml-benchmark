import statistics
import orjson
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import textwrap
from datetime import datetime
from pathlib import Path

def count_significant_digits(number):
    str_number = str(number)

    if '.' not in str_number:
        return 0

    decimal_part = str_number.split('.')[1]
    significant_digits = len(decimal_part.rstrip('0'))

    return significant_digits

def smallest_decimal_precision_subtract_one(numbers):
    smallest_precision = float('inf')

    for number in numbers:
        precision = count_significant_digits(number)

        if precision < smallest_precision:
            smallest_precision = precision

    return smallest_precision

'''
create_results_lists function organizes a json trace file 
and calculates the average duration/value, standard deviation, variance, and coefficient of variation
for the particular unique keys if applicable.
'''
def create_results_lists(traceEvents_key, traceEvents_items, gpu_id):
    results_list = []

    for key, value in traceEvents_items:
        results_dict = {'GPU': gpu_id}

        for sub_key, sub_value in value.items():
            durations = []
            col_name = ''
            stdev_rounded = 0.0
            for sub_dictionary in sub_value:
                if col_name == '':
                    if traceEvents_key == 'kernel':
                        col_name = (sub_dictionary['name'] +
                                    ' @ Grid: [' + ', '.join(str(x) for x in sub_dictionary['args']['grid']) + ']' +
                                    ' & Block: [' + ', '.join(str(x) for x in sub_dictionary['args']['block']) + ']')
                    else:
                        col_name = sub_dictionary['name']

                if 'Power' not in sub_dictionary['name'] and 'Temperature' not in sub_dictionary['name'] and 'VRAM' not in sub_dictionary['name']:
                    # the dur in json trace file is in microseconds, converting to milliseconds
                    durations.append(float(sub_dictionary['dur'])/1000)
                else:
                    durations.append(float(sub_dictionary['dur']))

            rounding_to = smallest_decimal_precision_subtract_one(durations)

            avg_duration = statistics.mean(durations)
            avg_duration_rounded = round(avg_duration, rounding_to)

            if len(durations) > 1:
                stdev = statistics.stdev(durations)
                variance = statistics.variance(durations)

                if avg_duration != 0.0:
                    cv_duration = (stdev / avg_duration) * 100
                else:
                    cv_duration = 0.0

                stdev_rounded = round(stdev, rounding_to)
                variance_rounded = round(variance, rounding_to)

            else:
                stdev_rounded = 0.0
                variance_rounded = 0.0
                cv_duration = 0.0

            if traceEvents_key == 'kernel':
                results_dict['Kernel'] = col_name
            elif traceEvents_key == 'cuda_runtime':
                results_dict['Cuda Runtime'] = col_name
            elif traceEvents_key == 'gpu_memcpy':
                results_dict['GPU Memcpy'] = col_name
            elif traceEvents_key == 'gpu_user_annotation':
                results_dict['GPU User Annotation'] = col_name
            elif traceEvents_key == 'gpu_user_annotation_vram':
                results_dict['GPU User Annotation VRAM Use'] = col_name
            elif traceEvents_key == 'gpu_user_annotation_power':
                results_dict['GPU User Annotation Power Use'] = col_name
            elif traceEvents_key == 'gpu_user_annotation_temp':
                results_dict['GPU User Annotation Temperature'] = col_name


            if 'vram' not in traceEvents_key and 'power' not in traceEvents_key and 'temp' not in traceEvents_key:
                results_dict['Average Duration Rounded (ms)'] = avg_duration_rounded
                results_dict['Standard Deviation Rounded (ms)'] = stdev_rounded
                results_dict['Variance Rounded (ms^2)'] = variance_rounded
            elif 'vram' in traceEvents_key:
                results_dict['Average VRAM Use Rounded (GB)'] = avg_duration_rounded
                results_dict['Standard Deviation Rounded (GB)'] = stdev_rounded
                results_dict['Variance Rounded (GB^2)'] = variance_rounded
            elif 'power' in traceEvents_key:
                results_dict['Average Power Use Rounded (W)'] = avg_duration_rounded
                results_dict['Standard Deviation Rounded (W)'] = stdev_rounded
                results_dict['Variance Rounded (W^2)'] = variance_rounded
            elif 'temp' in traceEvents_key:
                results_dict['Average Temperature Rounded (°C)'] = avg_duration_rounded
                results_dict['Standard Deviation Rounded (°C)'] = stdev_rounded
                results_dict['Variance Rounded (°C^2)'] = variance_rounded

            results_dict['% Coefficient of Variation'] = cv_duration
            results_dict['Total Calls'] = len(sub_value)

        results_list.append(results_dict)

    return results_list

def wrap_text(text, width):
    return '\n'.join(textwrap.wrap(text, width, replace_whitespace=False, break_on_hyphens=False))

def apply_wrap_to_dataframe(df, columns, width):
    for column in columns:
        if df[column].dtype == 'object':
            df[column] = df[column].astype(str).apply(lambda x: wrap_text(x, width))

    return df

'''
dataframe_to_table function converts all dataframe into a table_string text.
tabe_string text is for display output detailing all dataframes in the GUI.
It also gets saved into a text file to be reopened in another tab, Open Results.
Monospace fonts are required to view the table_string text properly
if the text file is opened elsewhere.
'''
def dataframe_to_table(datafrom_list, results_type, timestamp):
    entire_table = []
    for sub_df in datafrom_list:
        for key, df_v in sub_df.items():
            table = []
            column_widths = {}

            for col in df_v.columns:
                max_value_length = df_v[col].astype(str).apply(len).max()
                max_length = max(max_value_length, len(str(col)))

                if max_length > 50:
                    max_length = 50

                column_widths[col] = max_length

            separator = ' | '.join(['-' * (column_widths[col]) for col in df_v.columns])

            table.append(f"| {separator} |")

            header = ' | '.join([str(col).ljust(column_widths[col]) for col in df_v.columns])

            table.append(f"| {header} |")

            table.append(f"| {separator} |")

            table.append(f"| {separator} |")

            for idx, row in df_v.iterrows():
                row_newlines = 0
                row_d = []

                for col in df_v.columns:
                    cell = str(row[col])
                    c_split = cell.split('\n')
                    row_newlines = max(row_newlines, len(c_split))

                for col in df_v.columns:
                    cell = str(row[col])
                    c_split = cell.split('\n')
                    max_col = 0
                    for n in range(len(c_split)):
                        if n not in range(len(row_d)):
                            row_d.append('| ' + c_split[n].ljust(column_widths[col]))

                        else:
                            row_d[n] += ' | ' + c_split[n].ljust(column_widths[col])

                    if len(c_split) < row_newlines:
                        for n in range(len(c_split), row_newlines):

                            if n not in range(len(row_d)):
                                row_d.append('| '.ljust(column_widths[col] + 2))

                            else:
                                row_d[n] += ' | '.ljust(column_widths[col] + 3)

                for idx, r in enumerate(row_d):
                    row_d[idx] = r[:-1] + '  |'

                row_data = '\n'.join(row_d)

                table.append(f"{row_data}")

                table.append(f"| {separator} |")

            if key == 'gpu_name':
                path = Path(f"benchmark_results/{results_type}/{timestamp}/table")
                path.mkdir(parents=True, exist_ok=True)
                with open(f'benchmark_results/{results_type}/{timestamp}/table/table_summary.txt', 'w') as f:
                    f.write('\n'.join(table))
            entire_table.append('\n'.join(table))

    return '\n\n\n\n\n\n\n\n\n\n'.join(entire_table) + '\n\n\n\n\n\n\n\n\n\n'

def create_table_plots(json_paths, results_type):

    entire_comparison = []

    for json_path in json_paths:

        benchmark_results_list = []
        kernel_results_list = []
        cuda_runtime_results_list = []
        gpu_memcpy_results_list = []
        gpu_user_annotation_results_list = []
        gpu_user_annotation_vram_results_list = []
        gpu_user_annotation_power_results_list = []
        gpu_user_annotation_temp_results_list = []
        benchmark_results_dict = {'GPU ID TimeStamp': ''}

        with open(json_path, "rb") as json_file:  # Use 'rb' mode for orjson
            data = orjson.loads(json_file.read())

        traceEvents = {}

        for d_row in data['traceEvents']:
            if 'cat' in d_row:
                if d_row['cat'] == 'ac2g' or d_row['cat'] == 'Trace':
                    continue

                if d_row['cat'] not in traceEvents:
                    traceEvents[d_row['cat']] = {}

                if d_row['name'] not in traceEvents[d_row['cat']] and 'ProfilerStep#' not in d_row['name']:
                    traceEvents[d_row['cat']][d_row['name']] = {}

                elif 'ProfilerStep#' in d_row['name']:
                    continue

                '''
                This further separates the same kernels based on their grid and block values.
                '''
                if 'args' in d_row:
                    if 'grid' in d_row['args'] and 'block' in d_row['args']:
                        grid_list = d_row['args']['grid']
                        block_list = d_row['args']['block']
                        combined_gb_list = ','.join(str(x) for x in grid_list) + '/' + ','.join(
                            str(x) for x in block_list)

                        if combined_gb_list not in traceEvents[d_row['cat']][d_row['name']]:
                            traceEvents[d_row['cat']][d_row['name']][combined_gb_list] = []  # should be a dict?

                        traceEvents[d_row['cat']][d_row['name']][combined_gb_list].append(d_row)

                    else:

                        if 'other_args' not in traceEvents[d_row['cat']][d_row['name']]:
                            traceEvents[d_row['cat']][d_row['name']]['other_args'] = []

                        traceEvents[d_row['cat']][d_row['name']]['other_args'].append(d_row)

                else:
                    if 'no_args' not in traceEvents[d_row['cat']][d_row['name']]:
                        traceEvents[d_row['cat']][d_row['name']]['no_args'] = []

                    traceEvents[d_row['cat']][d_row['name']]['no_args'].append(d_row)

        for k, v in data.items():
            if 'benchmark_info' == k:
                for benchmark_info_key, benchmark_info_value in data[k].items():
                    benchmark_results_dict[benchmark_info_key] = benchmark_info_value

            if 'displayTimeUnit' == k:
                benchmark_results_dict[k] = data[k]

            if 'deviceProperties' == k:
                for x in data[k]:
                    if x['name'] == json_path.split('/')[2]:
                        for device_properties_key, device_properties_value in x.items():
                            benchmark_results_dict[device_properties_key] = device_properties_value

            if 'cupti_version' == k:
                benchmark_results_dict[k] = data[k]

            if 'cuda_runtime_version' == k:
                benchmark_results_dict[k] = data[k]

            if 'cuda_driver_version' == k:
                benchmark_results_dict[k] = data[k]

            if 'trace_id' == k:
                benchmark_results_dict[k] = data[k]

            if 'schemaVersion' == k:
                benchmark_results_dict[k] = data[k]

            if 'traceName' == k:
                benchmark_results_dict[k] = data[k]

            if 'baseTimeNanoseconds' == k:
                benchmark_results_dict[k] = data[k]


        benchmark_results_list.append(benchmark_results_dict)

        '''
        gpu_id is the identifier for the values in both the table_string and plots.
        '''
        gpu_id = (benchmark_results_dict['gpu_name'].strip() + '\n'
                  + benchmark_results_dict['device_id'].strip() + '\n'
                  + 'Model:' +benchmark_results_dict['traceName'].split('/')[3].strip()
                  + '/Script:' + benchmark_results_dict['traceName'].split('/')[2].strip() + '\n'
                  + 'Batch_size:' + benchmark_results_dict['traceName'].split('/')[4].split('_')[0].strip()
                  + '/Image_size:' + benchmark_results_dict['traceName'].split('/')[4].split('_')[1].strip() + '\n'
                  + 'Compile_Mode:' + benchmark_results_dict['compile_mode'].strip() + '\n'
                  + benchmark_results_dict['traceName'].split('/')[5].split('.')[0].strip())

        benchmark_results_dict['GPU ID TimeStamp'] = gpu_id

        for traceEvents_key, traceEvents_value in traceEvents.items():

            if 'kernel' == traceEvents_key:
                kernel_results_list = create_results_lists(traceEvents_key, traceEvents[traceEvents_key].items(), gpu_id)

            if 'cuda_runtime' == traceEvents_key:
                cuda_runtime_results_list = create_results_lists(traceEvents_key, traceEvents[traceEvents_key].items(), gpu_id)

            if 'gpu_memcpy' == traceEvents_key:
                gpu_memcpy_results_list = create_results_lists(traceEvents_key, traceEvents[traceEvents_key].items(), gpu_id)

            if 'gpu_user_annotation' == traceEvents_key:
                gpu_user_annotation_results_list = create_results_lists(traceEvents_key, traceEvents[traceEvents_key].items(), gpu_id)

            if 'gpu_user_annotation_vram' == traceEvents_key:
                gpu_user_annotation_vram_results_list = create_results_lists(traceEvents_key, traceEvents[traceEvents_key].items(), gpu_id)

            if 'gpu_user_annotation_power' == traceEvents_key:
                gpu_user_annotation_power_results_list = create_results_lists(traceEvents_key, traceEvents[traceEvents_key].items(), gpu_id)

            if 'gpu_user_annotation_temp' == traceEvents_key:
                gpu_user_annotation_temp_results_list = create_results_lists(traceEvents_key, traceEvents[traceEvents_key].items(), gpu_id)

        entire_comparison.append({'benchmark_results_list': benchmark_results_list,
                                  'kernel_results_list': kernel_results_list,
                                  'cuda_runtime_results_list': cuda_runtime_results_list,
                                  'gpu_memcpy_results_list': gpu_memcpy_results_list,
                                  'gpu_user_annotation_results_list': gpu_user_annotation_results_list,
                                  'gpu_user_annotation_vram_results_list': gpu_user_annotation_vram_results_list,
                                  'gpu_user_annotation_power_results_list':gpu_user_annotation_power_results_list,
                                  'gpu_user_annotation_temp_results_list': gpu_user_annotation_temp_results_list,
                                  })

    entire_comparison.sort(key=lambda x: sum(part['Average Duration Rounded (ms)'] for part in x['gpu_user_annotation_results_list']))

    '''
    This for loop compares the same values, such as kernels, 
    and calculates the % difference relative to the slowest GPU.
    The slowest GPU is the last index of the entire_comparison as it was sorted above.
    '''
    for entry in entire_comparison:
        for results_key, results_value in entry.items():
            if 'benchmark_results_list' == results_key:
                continue

            for single_result in results_value:
                try:
                    if 'Kernel' in single_result:
                        base_list = entire_comparison[-1][results_key]
                        base_found = [found_ for found_ in base_list if found_['Kernel'] == single_result['Kernel']]
                        if base_found != []:
                            base_line = base_found[0]['Average Duration Rounded (ms)']

                        else:
                            single_result['% Avg Faster Than Avg Slowest'] = 'N/A'
                            continue

                    if 'Cuda Runtime' in single_result:
                        base_list = entire_comparison[-1][results_key]
                        base_found = [found_ for found_ in base_list if found_['Cuda Runtime'] == single_result['Cuda Runtime']]
                        if base_found != []:
                            base_line = base_found[0]['Average Duration Rounded (ms)']

                        else:
                            single_result['% Avg Faster Than Avg Slowest'] = 'N/A'
                            continue

                    if 'GPU Memcpy' in single_result:
                        base_list = entire_comparison[-1][results_key]
                        base_found = [found_ for found_ in base_list if found_['GPU Memcpy'] == single_result['GPU Memcpy']]
                        if base_found != []:
                            base_line = base_found[0]['Average Duration Rounded (ms)']

                        else:
                            single_result['% Avg Faster Than Avg Slowest'] = 'N/A'
                            continue

                    if 'GPU User Annotation' in single_result:
                        base_list = entire_comparison[-1][results_key]
                        base_found = [found_ for found_ in base_list if found_['GPU User Annotation'] == single_result['GPU User Annotation']]
                        if base_found != []:
                            base_line = base_found[0]['Average Duration Rounded (ms)']

                        else:
                            single_result['% Avg Faster Than Avg Slowest'] = 'N/A'
                            continue

                    if 'GPU User Annotation VRAM Use' in single_result:
                        base_list = entire_comparison[-1][results_key]
                        base_found = [found_ for found_ in base_list if found_['GPU User Annotation VRAM Use'] == single_result['GPU User Annotation VRAM Use']]
                        if base_found != []:
                            base_line = base_found[0]['Average VRAM Use Rounded (GB)']

                        else:
                            single_result['% Avg VRAM Use Than Avg Slowest'] = 'N/A'
                            continue

                    if 'GPU User Annotation Power Use' in single_result:
                        base_list = entire_comparison[-1][results_key]
                        base_found = [found_ for found_ in base_list if found_['GPU User Annotation Power Use'] == single_result['GPU User Annotation Power Use']]
                        if base_found != []:
                            base_line = base_found[0]['Average Power Use Rounded (W)']

                        else:
                            single_result['% Avg Power Use Than Avg Slowest'] = 'N/A'
                            continue

                    if 'GPU User Annotation Temperature' in single_result:
                        base_list = entire_comparison[-1][results_key]
                        base_found = [found_ for found_ in base_list if found_['GPU User Annotation Temperature'] == single_result['GPU User Annotation Temperature']]
                        if base_found != []:
                            base_line = base_found[0]['Average Temperature Rounded (°C)']

                        else:
                            single_result['% Avg Temperature Than Avg Slowest'] = 'N/A'
                            continue

                except:
                    continue

                if ('GPU User Annotation VRAM Use' not in single_result and
                        'GPU User Annotation Power Use' not in single_result and
                        'GPU User Annotation Temperature' not in single_result):
                    comparing_value = single_result['Average Duration Rounded (ms)']
                    rounding_to = smallest_decimal_precision_subtract_one([comparing_value, base_line])
                    if comparing_value != 0.0:
                        percent_diff = ((base_line / comparing_value) - 1 ) * 100
                        single_result['% Avg Faster Than Avg Slowest'] = round(percent_diff, rounding_to)

                    else:
                        single_result['% Avg Faster Than Avg Slowest'] ='N/A'

                elif 'GPU User Annotation VRAM Use' in single_result:
                    comparing_value = single_result['Average VRAM Use Rounded (GB)']
                    rounding_to = smallest_decimal_precision_subtract_one([comparing_value, base_line])
                    if comparing_value != 0.0:
                        percent_diff = ((base_line / comparing_value) - 1 ) * 100
                        single_result['% Avg VRAM Use Than Avg Slowest'] = round(percent_diff, rounding_to)

                    else:
                        single_result['% Avg VRAM Use Than Avg Slowest'] = 'N/A'

                elif 'GPU User Annotation Power Use' in single_result:
                    comparing_value = single_result['Average Power Use Rounded (W)']
                    rounding_to = smallest_decimal_precision_subtract_one([comparing_value, base_line])
                    if comparing_value != 0.0:
                        percent_diff = ((base_line / comparing_value) - 1 ) * 100
                        single_result['% Avg Power Use Than Avg Slowest'] = round(percent_diff, rounding_to)

                    else:
                        single_result['% Avg Power Use Than Avg Slowest'] = 'N/A'

                elif 'GPU User Annotation Temperature' in single_result:
                    comparing_value = single_result['Average Temperature Rounded (°C)']
                    rounding_to = smallest_decimal_precision_subtract_one([comparing_value, base_line])
                    if comparing_value != 0.0:
                        percent_diff = ((base_line / comparing_value) - 1 ) * 100
                        single_result['% Avg Temperature Than Avg Slowest'] = round(percent_diff, rounding_to)

                    else:
                        single_result['% Avg Temperature Than Avg Slowest'] = 'N/A'

    benchmark_results_ = [entry['benchmark_results_list'] for entry in entire_comparison]
    kernel_results_ = [entry['kernel_results_list'] for entry in entire_comparison]
    cuda_runtime_results_ = [entry['cuda_runtime_results_list'] for entry in entire_comparison]
    gpu_memcpy_results_= [entry['gpu_memcpy_results_list'] for entry in entire_comparison]
    gpu_user_annotation_results_ = [entry['gpu_user_annotation_results_list'] for entry in entire_comparison]
    gpu_user_annotation_vram_results_ = [entry['gpu_user_annotation_vram_results_list'] for entry in entire_comparison]
    gpu_user_annotation_power_results_ = [entry['gpu_user_annotation_power_results_list'] for entry in entire_comparison]
    gpu_user_annotation_temp_results_ = [entry['gpu_user_annotation_temp_results_list'] for entry in entire_comparison]

    dataframes_list = []
    result_list = [benchmark_results_,
                   gpu_user_annotation_results_,
                   gpu_user_annotation_vram_results_,
                   gpu_user_annotation_power_results_,
                   gpu_user_annotation_temp_results_,
                   kernel_results_,
                   cuda_runtime_results_,
                   gpu_memcpy_results_,
                   ]

    '''
    This part creates specific dataframes and stores it as a value in a dictionary with a known key.
    The table_string, plots, and additional plots are based on these dataframes created.
    '''
    for result_ in result_list:
        results = list(itertools.chain(*result_))
        results_col_n = [len(n) for n in result_]

        if not results:
            continue

        if 'Kernel' in results[0]:
            key_name = 'Kernel'

        elif 'Cuda Runtime' in results[0]:
            key_name = 'Cuda Runtime'

        elif 'GPU Memcpy' in results[0]:
            key_name = 'GPU Memcpy'

        elif 'Kernel' in results[0]:
            key_name = 'Kernel'

        elif 'GPU User Annotation' in results[0]:
            key_name = 'GPU User Annotation'

        elif 'GPU User Annotation VRAM Use' in results[0]:
            key_name = 'GPU User Annotation VRAM Use'

        elif 'GPU User Annotation Power Use' in results[0]:
            key_name = 'GPU User Annotation Power Use'

        elif 'GPU User Annotation Temperature' in results[0]:
            key_name = 'GPU User Annotation Temperature'

        elif 'gpu_name'in results[0]:
            key_name = 'gpu_name'

        else:
            key_name = 'Broken'

        results = sorted(results, key=lambda x: x[key_name])
        results = list(results)

        df = pd.DataFrame(results)

        if key_name == 'gpu_name':
            df = df.transpose()
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Info'}, inplace=True)

        dataframes_list.append({key_name: df})


    # Additional dataframes made from further analyzing already made dataframes
    for df_dictionary in dataframes_list:
        for df_key, df_value in df_dictionary.items():
            '''
            Creates a dataframe based on the amount of calls a value, such as a kernel, gets calls.
            This can help identify why some benchmarks have bad performance, 
            due to bad optimizations or missing features found in new GPU generations.
            '''
            if df_key != 'gpu_name':
                head_value = df_value.columns[1]
                col_values = df_value.columns
                if f'{head_value} Total Calls Per Epoch' != df_key and 'Total Calls' in col_values:
                    total_result = df_value.groupby('GPU').agg(Total_Calls=('Total Calls', 'sum')).reset_index()

                    total_result = total_result.rename(columns={'Total_Calls': f'{head_value} Total Calls Per Epoch'})

                    total_result = total_result.sort_values(by=f'{head_value} Total Calls Per Epoch', ascending=True)

                    epochs = -1
                    for x in dataframes_list:
                        for x_key, x_value in x.items():
                            if x_key == 'GPU User Annotation':
                                epochs = x_value['Total Calls'][0]
                                break
                        if epochs != -1:
                            break

                    total_result[f'{head_value} Total Calls Per Epoch'] = total_result[f'{head_value} Total Calls Per Epoch'].astype(float) / float(epochs)

                    base_line = total_result.iloc[-1][f'{head_value} Total Calls Per Epoch']
                    if base_line != 0.0:
                        percent_diff = ((base_line / total_result[f'{head_value} Total Calls Per Epoch']) - 1) * 100
                        rounding_to = smallest_decimal_precision_subtract_one(
                            [total_result[f'{head_value} Total Calls Per Epoch'], base_line])
                        total_result['% Total Calls'] = round(percent_diff, rounding_to)
                    else:
                        total_result['% Total Calls'] = 'N/A'

                    dataframes_list.append({f'{head_value} Total Calls Per Epoch': total_result})

            '''
            Creates a dataframe based on the predefined GPU User Annotation.
            The predefined GPU User Annotations is separated by name in a benchmark, 
            this combines the predefine parts into one to show a quick overview of performance between GPUs.
            '''
            if df_key == 'GPU User Annotation':
                total_result = df_value.groupby('GPU').agg(
                    Added_Value=('Average Duration Rounded (ms)', 'sum'),
                    Combined_Partial=('GPU User Annotation', ' --> '.join),
                    Combined_Calls=('Total Calls', 'sum')).reset_index()

                total_result = total_result.rename(columns={'Added_Value': 'Total Average Duration Rounded (ms)'})
                total_result = total_result.rename(columns={'Combined_Partial': 'Total GPU User Annotations'})
                total_result = total_result.rename(columns={'Combined_Calls': 'Total Combined Calls'})

                total_result['Total Combined Calls'] = total_result['Total Combined Calls'] / total_result['Total GPU User Annotations'].str.split(' --> ').apply(len)
                total_result['Total Combined Calls'] = total_result['Total Combined Calls'].apply(int)

                total_result = total_result.sort_values(by='Total Average Duration Rounded (ms)', ascending=True)

                base_line = total_result.iloc[-1]['Total Average Duration Rounded (ms)']
                if base_line != 0.0:
                    percent_diff = ((base_line / total_result['Total Average Duration Rounded (ms)']) - 1) * 100
                    rounding_to = smallest_decimal_precision_subtract_one([total_result['Total Average Duration Rounded (ms)'], base_line])
                    total_result['% Total Avg Faster Than Total Avg Slowest'] = round(percent_diff, rounding_to)
                else:
                    total_result['% Total Avg Faster Than Total Avg Slowest'] = 'N/A'

                dataframes_list.append({'Total GPU User Annotations': total_result})

    total_popped = dataframes_list.pop()
    dataframes_list.insert(1, total_popped)

    '''
    This limits how long a text can be displayed before needing to be wrapped.
    A width of 50 seems to look decent in the table_string when viewing in the GUI.
    '''
    for d in dataframes_list:
        for k, v, in d.items():
            apply_wrap_to_dataframe(df=v, columns=v.columns, width=50)

    timestamp = str(datetime.today().replace(microsecond=0)).replace(' ', '_')

    table_string = dataframe_to_table(dataframes_list, results_type, timestamp)

    path = Path(f"benchmark_results/{results_type}/{timestamp}/table")
    path.mkdir(parents=True, exist_ok=True)
    path = Path(f"benchmark_results/{results_type}/{timestamp}/graphs")
    path.mkdir(parents=True, exist_ok=True)

    with open(f'benchmark_results/{results_type}/{timestamp}/table/table_result.txt', 'w') as f:
        f.write(table_string)

    graph_dir = f"benchmark_results/{results_type}/{timestamp}/graphs/"

    '''
    This part is the start of creating the plots/graphs.
    Using the dataframes made previously, it takes the unique values in a column
    and uses the rows found to be combined into a plot/graph.
    Other plots/graphs are made here from existing dataframes
    such as "Combined GPU User Annotations", which groups the
    individual GPU User Annotations together without merging them,
    showing each part in comparison to the whole.
    '''
    for df_i in dataframes_list:
        for df_k, df_v in df_i.items():
            if  df_k != 'gpu_name':
                path = Path(f"benchmark_results/{results_type}/{timestamp}/graphs/{df_k}")
                path.mkdir(parents=True, exist_ok=True)

                sub_dfs = []
                '''
                kernel_values represents the unique values found in the specific column.
                sub_df represents a list of unique sub dataframes
                '''
                for kernel_value in df_v[df_k].unique():
                    kernel_rows = df_v[df_v[df_k] == kernel_value]
                    s_cols = kernel_rows.columns

                    if ('GPU User Annotation VRAM Use' not in s_cols and
                            'GPU User Annotation Power Use' not in s_cols and
                            'GPU User Annotation Temperature' not in s_cols and
                            'Total GPU User Annotations' not in s_cols and
                            'Total Calls' not in df_k):
                        sub_df = kernel_rows[['GPU', df_k, 'Average Duration Rounded (ms)',
                                              'Standard Deviation Rounded (ms)',
                                              'Variance Rounded (ms^2)',
                                              '% Coefficient of Variation',
                                              '% Avg Faster Than Avg Slowest',
                                              ]]

                    elif 'GPU User Annotation VRAM Use' in s_cols:
                        sub_df = kernel_rows[['GPU', df_k, 'Average VRAM Use Rounded (GB)',
                                              'Standard Deviation Rounded (GB)',
                                              'Variance Rounded (GB^2)',
                                              '% Coefficient of Variation',
                                              '% Avg VRAM Use Than Avg Slowest',
                                              ]]

                    elif 'GPU User Annotation Power Use' in s_cols:
                        sub_df = kernel_rows[['GPU', df_k, 'Average Power Use Rounded (W)',
                                              'Standard Deviation Rounded (W)',
                                              'Variance Rounded (W^2)',
                                              '% Coefficient of Variation',
                                              '% Avg Power Use Than Avg Slowest',
                                              ]]

                    elif 'GPU User Annotation Temperature' in s_cols:
                        sub_df = kernel_rows[['GPU', df_k, 'Average Temperature Rounded (°C)',
                                              'Standard Deviation Rounded (°C)',
                                              'Variance Rounded (°C^2)',
                                              '% Coefficient of Variation',
                                              '% Avg Temperature Than Avg Slowest',
                                              ]]

                    elif 'Total GPU User Annotations' in s_cols:
                        sub_df = kernel_rows[['GPU', df_k, 'Total Average Duration Rounded (ms)',
                                              '% Total Avg Faster Than Total Avg Slowest',
                                              ]]

                    elif ' Total Calls' in df_k:
                        break

                    sub_dfs.append(sub_df)

                for i, s_df in enumerate(sub_dfs):
                    height_plot = len(s_df.index) * 3
                    if height_plot < 25:
                        height_plot = 25

                    fig, axes = plt.subplots(2, 2,
                                             figsize=(15, height_plot))

                    s_df_cols = s_df.columns

                    if (('GPU User Annotation VRAM Use' not in s_df_cols and
                            'GPU User Annotation Power Use' not in s_df_cols and
                            'GPU User Annotation Temperature' not in s_df_cols) and
                            'Total GPU User Annotations' not in s_df_cols and
                            'Total Calls' not in df_k):
                        ax1 = s_df.plot(x='GPU',
                                        y='Average Duration Rounded (ms)',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        ax=axes[0, 0])

                        axes[0, 0].get_legend().remove()
                        axes[0, 0].set_ylabel('GPU')
                        axes[0, 0].set_xlabel('Average Duration Rounded (ms)\nlower is better')
                        axes[0, 0].invert_yaxis()

                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999,
                                 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold',
                                 transform=ax1.transAxes)

                        bar_labels_str = []
                        for x in s_df['% Avg Faster Than Avg Slowest']:
                            if x == 'N/A':
                                bar_labels_str.append(f"{x}")

                            elif float(x) > 0:
                                bar_labels_str.append(f"+{x}%\nFaster")

                            elif float(x) == 0:
                                bar_labels_str.append(f"{x}%\nBaseline")

                            else:
                                bar_labels_str.append(f"{x}%\nSlower")

                        axes[0, 0].bar_label(axes[0, 0].containers[0],
                                             labels=bar_labels_str)

                        ax2 = s_df.plot(x='GPU',
                                        y='Standard Deviation Rounded (ms)',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        ax=axes[0, 1])

                        axes[0, 1].get_legend().remove()
                        axes[0, 1].set_ylabel('GPU')
                        axes[0, 1].set_xlabel('Standard Deviation Rounded (ms)\nlower is better')
                        axes[0, 1].invert_yaxis()

                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999,
                                 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold',
                                 transform=ax2.transAxes)

                        ax3 = s_df.plot(x='GPU',
                                        y='% Coefficient of Variation',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        ax=axes[1, 0])

                        axes[1, 0].get_legend().remove()
                        axes[1, 0].set_ylabel('GPU')
                        axes[1, 0].set_xlabel('% Coefficient of Variation\nlower is better')
                        axes[1, 0].invert_yaxis()

                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999,
                                 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold',
                                 transform=ax3.transAxes)

                        ax4 = s_df.plot(x='GPU',
                                        y='Variance Rounded (ms^2)',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        ax=axes[1, 1])

                        axes[1, 1].get_legend().remove()
                        axes[1, 1].set_ylabel('GPU')
                        axes[1, 1].set_xlabel('Variance Rounded (ms^2)\nlower is better')
                        axes[1, 1].invert_yaxis()

                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999,
                                 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold',
                                 transform=ax4.transAxes)

                    elif 'GPU User Annotation VRAM Use' in s_df_cols:
                        ax1 = s_df.plot(x='GPU',
                                        y='Average VRAM Use Rounded (GB)',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        ax=axes[0, 0])

                        axes[0, 0].get_legend().remove()
                        axes[0, 0].set_ylabel('GPU')
                        axes[0, 0].set_xlabel('Average VRAM Use Rounded (GB)\nlower is better')
                        axes[0, 0].invert_yaxis()

                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999,
                                 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold', transform=ax1.transAxes)

                        bar_labels_str = []
                        for x in s_df['% Avg VRAM Use Than Avg Slowest']:
                            if x == 'N/A':
                                bar_labels_str.append(f"{x}")

                            elif float(x) > 0:
                                bar_labels_str.append(f"+{x}%\nLess")

                            elif float(x) == 0:
                                bar_labels_str.append(f"{x}%\nBaseline")

                            else:
                                bar_labels_str.append(f"{x}%\nMore")

                        axes[0, 0].bar_label(axes[0, 0].containers[0],
                                             labels=bar_labels_str)

                        ax2 = s_df.plot(x='GPU',
                                        y='Standard Deviation Rounded (GB)',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        ax=axes[0, 1])

                        axes[0, 1].get_legend().remove()
                        axes[0, 1].set_ylabel('GPU')
                        axes[0, 1].set_xlabel('Standard Deviation Rounded (GB)\nlower is better')
                        axes[0, 1].invert_yaxis()

                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999,
                                 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold',
                                 transform=ax2.transAxes)

                        ax3 = s_df.plot(x='GPU',
                                        y='% Coefficient of Variation',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        ax=axes[1, 0])

                        axes[1, 0].get_legend().remove()
                        axes[1, 0].set_ylabel('GPU')
                        axes[1, 0].set_xlabel('% Coefficient of Variation\nlower is better')
                        axes[1, 0].invert_yaxis()

                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999,
                                 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold',
                                 transform=ax3.transAxes)

                        ax4 = s_df.plot(x='GPU',
                                        y='Variance Rounded (GB^2)',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        ax=axes[1, 1])

                        axes[1, 1].get_legend().remove()
                        axes[1, 1].set_ylabel('GPU')
                        axes[1, 1].set_xlabel('Variance Rounded (GB^2)\nlower is better')
                        axes[1, 1].invert_yaxis()

                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999,
                                 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold',
                                 transform=ax4.transAxes)

                    elif 'GPU User Annotation Power Use' in s_df_cols:
                        ax1 = s_df.plot(x='GPU',
                                        y='Average Power Use Rounded (W)',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        ax=axes[0, 0])

                        axes[0, 0].get_legend().remove()
                        axes[0, 0].set_ylabel('GPU')
                        axes[0, 0].set_xlabel('Average Power Use Rounded (W)\nlower is better')
                        axes[0, 0].invert_yaxis()

                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999,
                                 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold',
                                 transform=ax1.transAxes)

                        bar_labels_str = []
                        for x in s_df['% Avg Power Use Than Avg Slowest']:
                            if x == 'N/A':
                                bar_labels_str.append(f"{x}")

                            elif float(x) > 0:
                                bar_labels_str.append(f"+{x}%\nLess")

                            elif float(x) == 0:
                                bar_labels_str.append(f"{x}%\nBaseline")

                            else:
                                bar_labels_str.append(f"{x}%\nMore")

                        axes[0, 0].bar_label(axes[0, 0].containers[0],
                                             labels=bar_labels_str)

                        ax2 = s_df.plot(x='GPU',
                                        y='Standard Deviation Rounded (W)',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        ax=axes[0, 1])

                        axes[0, 1].get_legend().remove()
                        axes[0, 1].set_ylabel('GPU')
                        axes[0, 1].set_xlabel('Standard Deviation Rounded (W)\nlower is better')
                        axes[0, 1].invert_yaxis()

                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999,
                                 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold',
                                 transform=ax2.transAxes)

                        ax3 = s_df.plot(x='GPU',
                                        y='% Coefficient of Variation',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        ax=axes[1, 0])

                        axes[1, 0].get_legend().remove()
                        axes[1, 0].set_ylabel('GPU')
                        axes[1, 0].set_xlabel('% Coefficient of Variation\nlower is better')
                        axes[1, 0].invert_yaxis()


                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999,
                                 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold',
                                 transform=ax3.transAxes)

                        ax4 = s_df.plot(x='GPU',
                                        y='Variance Rounded (W^2)',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        ax=axes[1, 1])

                        axes[1, 1].get_legend().remove()
                        axes[1, 1].set_ylabel('GPU')
                        axes[1, 1].set_xlabel('Variance Rounded (W^2)\nlower is better')
                        axes[1, 1].invert_yaxis()

                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999,
                                 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold',
                                 transform=ax4.transAxes)

                    elif 'GPU User Annotation Temperature' in s_df_cols:
                        ax1 = s_df.plot(x='GPU',
                                        y='Average Temperature Rounded (°C)',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        ax=axes[0, 0])

                        axes[0, 0].get_legend().remove()
                        axes[0, 0].set_ylabel('GPU')
                        axes[0, 0].set_xlabel('Average Temperature Rounded (°C)\nlower is better')
                        axes[0, 0].invert_yaxis()

                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999,
                                 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold',
                                 transform=ax1.transAxes)

                        bar_labels_str = []
                        for x in s_df['% Avg Temperature Than Avg Slowest']:
                            if x == 'N/A':
                                bar_labels_str.append(f"{x}")

                            elif float(x) > 0:
                                bar_labels_str.append(f"+{x}%\nLess")

                            elif float(x) == 0:
                                bar_labels_str.append(f"{x}%\nBaseline")

                            else:
                                bar_labels_str.append(f"{x}%\nMore")

                        axes[0, 0].bar_label(axes[0, 0].containers[0],
                                             labels=bar_labels_str)

                        ax2 = s_df.plot(x='GPU',
                                        y='Standard Deviation Rounded (°C)',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        ax=axes[0, 1])

                        axes[0, 1].get_legend().remove()
                        axes[0, 1].set_ylabel('GPU')
                        axes[0, 1].set_xlabel('Standard Deviation Rounded (°C)\nlower is better')
                        axes[0, 1].invert_yaxis()

                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999,
                                 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold',
                                 transform=ax2.transAxes)

                        ax3 = s_df.plot(x='GPU',
                                        y='% Coefficient of Variation',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        ax=axes[1, 0])

                        axes[1, 0].get_legend().remove()
                        axes[1, 0].set_ylabel('GPU')
                        axes[1, 0].set_xlabel('% Coefficient of Variation\nlower is better')
                        axes[1, 0].invert_yaxis()

                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999,
                                 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold',
                                 transform=ax3.transAxes)


                        ax4 = s_df.plot(x='GPU',
                                        y='Variance Rounded (°C^2)',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        ax=axes[1, 1])

                        axes[1, 1].get_legend().remove()
                        axes[1, 1].set_ylabel('GPU')
                        axes[1, 1].set_xlabel('Variance Rounded (°C^2)\nlower is better')
                        axes[1, 1].invert_yaxis()

                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999,
                                 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold',
                                 transform=ax4.transAxes)

                    plt.subplots_adjust(wspace=2)

                    fig.savefig(f'benchmark_results/{results_type}/{timestamp}/graphs/{df_k}/{str(i)}.png',
                                dpi=100,
                                bbox_inches='tight')

                    plt.clf()
                    plt.close()

                    if 'Total GPU User Annotations' in s_df_cols:

                        height_plot = len(s_df.index) * 2 + 5

                        ax0 = s_df.plot(x='GPU',
                                        y='Total Average Duration Rounded (ms)',
                                        kind='barh',
                                        title=df_k + '\n ' + s_df.loc[s_df.index[0], df_k],
                                        figsize=(15, height_plot))

                        ax0.get_legend().remove()
                        ax0.set_ylabel('GPU')
                        ax0.set_xlabel('Total Average Duration Rounded (ms) \nlower is better')

                        # Add watermark 'yero-ml-benchmark' to the plot
                        plt.text(0.5, 0.999, 'yero-ml-benchmark',
                                 fontsize=12,
                                 color='gray',
                                 alpha=0.8,
                                 ha='center',
                                 va='center',
                                 fontweight='bold',
                                 transform=ax0.transAxes)

                        bar_labels_str = []
                        for x in s_df['% Total Avg Faster Than Total Avg Slowest']:
                            if x == 'N/A':
                                bar_labels_str.append(f"{x}")

                            elif float(x) > 0:
                                bar_labels_str.append(f"+{x}%\nFaster")

                            elif float(x) == 0:
                                bar_labels_str.append(f"{x}%\nBaseline")

                            else:
                                bar_labels_str.append(f"{x}%\nSlower")

                        ax0.bar_label(ax0.containers[0],
                                      labels=bar_labels_str)

                        plt.gca().invert_yaxis()

                        plt.savefig(f'benchmark_results/{results_type}/{timestamp}/graphs/{df_k}/{str(i)}.png',
                                    dpi=100,
                                    bbox_inches='tight')

                        plt.clf()
                        plt.close()

                if 'Total Calls' in df_k:
                    height_plot = len(df_v.index) * 2 + 5

                    ax0 = df_v.plot(x='GPU',
                                    y=df_k,
                                    kind='barh',
                                    title=df_k + 'Per Epoch',
                                    figsize=(15, height_plot))

                    ax0.get_legend().remove()
                    ax0.set_ylabel('GPU')
                    ax0.set_xlabel('Total Calls')

                    # Add watermark 'yero-ml-benchmark' to the plot
                    plt.text(0.5, 0.999,
                             'yero-ml-benchmark',
                             fontsize=12,
                             color='gray',
                             alpha=0.8,
                             ha='center',
                             va='center',
                             fontweight='bold',
                             transform=ax0.transAxes)

                    bar_labels_str = []
                    for x in df_v['% Total Calls']:
                        if x == 'N/A':
                            bar_labels_str.append(f"{x}")

                        elif float(x) > 0:
                            bar_labels_str.append(f"+{x}%\nLess")

                        elif float(x) == 0:
                            bar_labels_str.append(f"{x}%\nBaseline")

                        else:
                            bar_labels_str.append(f"{x}%\nMore")

                    ax0.bar_label(ax0.containers[0],
                                  labels=bar_labels_str)

                    plt.gca().invert_yaxis()

                    plt.savefig(f'benchmark_results/{results_type}/{timestamp}/graphs/{df_k}/0.png',
                                dpi=100,
                                bbox_inches='tight')

                    plt.clf()
                    plt.close()

                if df_k == 'GPU User Annotation':
                    height_plot = len(df_v['GPU'].unique()) * 2 + 5

                    fig, ax = plt.subplots(figsize=(15, height_plot))

                    df_v.groupby(['GPU', 'GPU User Annotation'], sort=False) \
                        ['Average Duration Rounded (ms)'].sum() \
                        .to_frame() \
                        .pivot_table(index='GPU',
                                     columns='GPU User Annotation',
                                     values='Average Duration Rounded (ms)',
                                     sort=False) \
                        .plot(kind='barh',
                              stacked=True,
                              ax=ax)

                    ax.legend(title='GPU User Annotation',
                              bbox_to_anchor=(1.0, 1),
                              loc='upper left')

                    plt.text(0.5, 0.999,
                             'yero-ml-benchmark',
                             fontsize=12,
                             color='gray',
                             alpha=0.8,
                             ha='center',
                             va='center',
                             fontweight='bold',
                             transform=ax.transAxes)

                    ax.set_xlabel('Average Duration Rounded (ms)\nlower is better')
                    ax.set_ylabel('GPU')
                    ax.set_title('Combined GPU User Annotations')

                    plt.gca().invert_yaxis()

                    path = Path(f"benchmark_results/{results_type}/{timestamp}/graphs/Combined {df_k}s")
                    path.mkdir(parents=True, exist_ok=True)

                    plt.savefig(f'benchmark_results/{results_type}/{timestamp}/graphs/Combined {df_k}s/0.png',
                                dpi=100,
                                bbox_inches='tight')  # Save with high resolution

                    plt.clf()
                    plt.close()

    return table_string, graph_dir
