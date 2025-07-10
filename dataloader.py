import json
from operator import index
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from typing import Dict, List, Tuple

from pojo import CallEventNode
from utrace_data_export import utrace2CSV


def build_call_tree(events_df:DataFrame,timer_df:DataFrame)->CallEventNode:

    events_df = events_df.merge(
        timer_df[['TimerId','source_file','source_line']],how='left',on='TimerId'
    )
    min_start_time = events_df['StartTime'].min()
    max_end_time = events_df['EndTime'].max()
    total_duration = max_end_time - min_start_time

    # 获取第一个事件的信息作为根节点的线程信息
    first_event = events_df.iloc[0]

    # 3. 使用聚合信息创建内容更丰富的根节点
    root = CallEventNode(
        name='root',
        CallDepth=-1,  # 根节点深度应比任何真实节点都小
        TimerName='root',
        StartTime=min_start_time,
        EndTime=max_end_time,
        Duration=total_duration,
        ThreadId=first_event.ThreadId,
        ThreadName=first_event.ThreadName
    )
    stack = [root]

    # 使用itertuples以获得更好的性能
    for row in tqdm(events_df.itertuples(), total=len(events_df), desc="Building Call Tree"):
        # 使用 _asdict() 将命名元组高效地转换为字典
        row_dict = row._asdict() # TODO 对于source_file字段为nan的处理

        # 修正回溯逻辑
        # 当新节点深度小于等于栈顶节点时，需要弹出栈顶元素以找到正确的父节点
        while row.CallDepth <= stack[-1].CallDepth:
            stack.pop()

        # 正确的父节点现在是栈顶元素
        parent_node = stack[-1]

        # 创建新节点并直接关联父节点
        node = CallEventNode(
            name=row_dict['TimerName'],
            parent=parent_node,
            **row_dict
        )

        stack.append(node)

    # 返回真正的根节点
    return root




def load_events(file_path,thread_name='GameThread')->DataFrame:
    chunk_iterator = pd.read_csv(file_path, chunksize=10**6)
    # 创建一个列表来存储处理后的结果
    results_list = []

    # 循环处理每个数据块
    print("开始分块处理文件...")
    for i, chunk in tqdm(enumerate(chunk_iterator)):
        processed_chunk = chunk[chunk['ThreadName'] == thread_name].dropna()
        # 将处理后的结果添加到列表中
        results_list.append(processed_chunk)

    # 将所有处理过的、较小的结果合并成一个最终的DataFrame
    print("合并所有处理结果...")
    final_df = pd.concat(results_list)
    # 按 StartTime 排序并重置索引
    final_df = final_df.sort_values('StartTime')

    # 添加顺序编号作为 event_id (从0或1开始)
    final_df['event_id'] = final_df.index

    print("处理完成！")
    final_df.rename(columns={'Depth':'CallDepth'}, inplace=True)
    print(final_df.info())
    return final_df

def load_timer(file_path,timer_type='CPU')->DataFrame:
    df = pd.read_csv(file_path)
    df.rename(columns={'Name':'TimerName','Type':'TimerType','Id':'TimerId','File':'source_file','Line':'source_line'}, inplace=True)
    return df[df['TimerType'] == timer_type]

def loadTimerStats(file_path)->DataFrame:
    df = pd.read_csv(file_path)
    column_rename_map = {
        'Name': 'TimerName',
        'Count': 'count',
        'Incl': 'inclusive_total',
        'I.Min': 'inclusive_min',
        'I.Max': 'inclusive_max',
        'I.Avg': 'inclusive_avg',
        'I.Med': 'inclusive_median',
        'Excl': 'exclusive_total',
        'E.Min': 'exclusive_min',
        'E.Max': 'exclusive_max',
        'E.Avg': 'exclusive_avg',
        'E.Med': 'exclusive_median'
    }
    df.rename(columns=column_rename_map, inplace=True)
    return df


def load_insights_data(data_dir:str,timer_type='CPU',thread_name='GameThread'):
    timer_df,timer_stats,events_df = load_timer(os.path.join(data_dir,'Timers.csv'),timer_type=timer_type),\
            loadTimerStats(os.path.join(data_dir,'TimerStat.csv')), \
            load_events(os.path.join(data_dir, 'TimerEvents.csv'), thread_name=thread_name)

    timer_name_set = set(timer_df['TimerName'].tolist())
    timer_stats = timer_stats[timer_stats['TimerName'].apply(lambda x:x in timer_name_set)]

    events_df = events_df[events_df['TimerName'].apply(lambda x: x in timer_name_set)]
    call_tree = build_call_tree(events_df,timer_df)

    return timer_df,timer_stats,call_tree

def load_raw_top_stat(raw_stat_file:str)->Tuple[np.ndarray,np.ndarray]:
    """
    :param raw_stat_file: 原始top stat数据
    :return: 事件名称列表，
    """
    with open(raw_stat_file,'r',encoding='utf-8') as f:
        lines = f.readlines()
        event_name_list = np.array(lines[0].split(',')[1:]) # 删除第一个
        csv_profiler_data = []
        for line in lines[1:]:
            frame_array = np.array(line.strip().split(',')[1:]) # 删除第一个
            if frame_array[0]=='EVENTS':
                break
            frame_array = np.where(frame_array.astype(str) == '', '0.0', frame_array).astype(float)
            csv_profiler_data.append(frame_array)

        profiler_matrix = np.vstack(csv_profiler_data)  # row为一帧数据
        assert profiler_matrix.shape[1] == len(event_name_list)
        return event_name_list, profiler_matrix


def extra_exception_top_stat(event_name_list:np.ndarray, profiler_matrix:np.ndarray,windows_threshold:int=100,max_frame_cost:float=30.0)->Tuple[List[np.ndarray],np.ndarray]:
    """
    :param event_name_list: 事件名称
    :param profiler_matrix: 一级性能追踪数据矩阵，每行 为一帧
    :param windows_threshold: 持续低帧的窗口阈值，默认100帧，即超过100个持续的低帧区间才会被统计，单位为帧
    :param max_frame_cost: 低帧率阈值，默认为30fps,单位为毫秒，当帧率小于30帧时才视为低帧
    :return: List[Profiler matrix],event_name_list
    """

    def find_exception_blocks(bool_arr, windows_threshold):
        """
        :param bool_arr: bool数组，是否为异常帧
        :param windows_threshold: 最小窗口
        :return:
        valid_blocks = （异常窗口起始索引，异常窗口终止索引）
        """
        # 确保输入是NumPy数组
        bool_arr = np.asarray(bool_arr, dtype=bool)

        # 计算True区间的起始和结束索引
        diff = np.diff(np.concatenate(([False], bool_arr, [False])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        # 计算区间长度并筛选
        lengths = ends - starts + 1
        valid_blocks = [(start, end) for start, end, length in zip(starts, ends, lengths) if length >= windows_threshold]

        return valid_blocks


    # GameThread/GameEngineTick
    game_engine_tick_index = np.where(event_name_list == 'GameThread/GameEngineTick')[0]
    is_exception_frame = profiler_matrix[:,game_engine_tick_index] > 1000.0/max_frame_cost
    exception_interval_index = find_exception_blocks(is_exception_frame,windows_threshold)
    return [profiler_matrix[start:end+1,:] for start, end in exception_interval_index],event_name_list



def top_stat_data2DataFrame(top_stat_profiler_matrix:np.ndarray[float],event_name_list:List[str],module_dict_file:str)->DataFrame:
    """
    top_stat_profiler_matrix: 矩阵，行为一帧的数据，
    event_name_list: 事件名称
    module_dict_file: 顶级插桩名称与模块对应字典
    :return: DataFrame: {
            Event:
            min:
            max:
            avg:
            pct:
            count:
            module:
        }
    """
    with open(module_dict_file,'r',encoding='utf-8') as f:
        module_dict = json.load(f)
    game_engine_tick_index = np.where(event_name_list == 'GameThread/GameEngineTick')[0]
    sum_cost = np.sum(top_stat_profiler_matrix[:,game_engine_tick_index])
    top_stat_df = pd.DataFrame({
        'Event':event_name_list,
        'min':np.min(top_stat_profiler_matrix,axis=0),
        'max':np.max(top_stat_profiler_matrix,axis=0),
        'avg':np.average(top_stat_profiler_matrix,axis=0),
        'pct':np.sum(top_stat_profiler_matrix,axis=0)/sum_cost,
        'count':np.count_nonzero(top_stat_profiler_matrix,axis=0)
    })

    top_stat_df['module'] = top_stat_df['Event'].apply(lambda x:module_dict.get(x,'UNKNOWN'))
    return top_stat_df









import os
from typing import Dict, Any, Tuple


class TraceDataManager:
    """
    Manages loading and caching of trace data from .utrace files.

    This class acts as a dictionary where keys are paths to .utrace files.
    It automatically handles the conversion of .utrace to CSV format using
    UnrealInsights.exe if the CSVs are missing or outdated. The insights
    data is then loaded from the CSVs, cached, and returned.
    """

    _SOURCE_FILES = ['Timers.csv', 'TimerStat.csv', 'TimerEvents.csv', 'Threads.csv']

    def __init__(self, unreal_insights_exe_path: str, timer_type: str = 'CPU', thread_name: str = 'GameThread'):
        """
        Initializes the TraceDataManager.

        Args:
            unreal_insights_exe_path (str): The full path to UnrealInsights.exe.
            timer_type (str): The timer type to filter for (e.g., 'CPU').
            thread_name (str): The thread name to filter events for.
        """
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self.unreal_insights_exe_path = unreal_insights_exe_path
        self.timer_type = timer_type
        self.thread_name = thread_name
        print(f"TraceDataManager initialized for timer_type='{timer_type}', thread_name='{thread_name}'")
        if not Path(self.unreal_insights_exe_path).exists() and "MOCK" not in self.unreal_insights_exe_path:
            print(
                f"WARNING: UnrealInsights.exe not found at '{self.unreal_insights_exe_path}'. Operations will be simulated.")

    def _get_csv_dir(self, utrace_path: str) -> str:
        """Determines the directory for storing derived CSV files."""
        base_name = os.path.splitext(os.path.basename(utrace_path))[0]
        return os.path.join(os.path.dirname(utrace_path), base_name)

    def __getitem__(self, utrace_path: str) -> Any:
        """
        Retrieves trace data for a given .utrace file.
        Handles .utrace -> CSV conversion, caching, and lazy loading.
        """
        if not os.path.exists(utrace_path):
            raise FileNotFoundError(f"Source .utrace file not found: '{utrace_path}'")

        current_utrace_mtime = os.path.getmtime(utrace_path)

        # 1. Check cache validity against the .utrace file's modification time
        if utrace_path in self._cache:
            cached_utrace_mtime, cached_data = self._cache[utrace_path]
            if current_utrace_mtime == cached_utrace_mtime:
                print(f"[Cache] HIT: Returning cached data for '{os.path.basename(utrace_path)}'.")
                return cached_data
            else:
                print(f"[Cache] STALE: .utrace file '{os.path.basename(utrace_path)}' has changed. Reloading.")
        else:
            print(f"[Cache] MISS: Data for '{os.path.basename(utrace_path)}' not in cache. Loading.")

        # 2. Determine CSV directory and check if up-to-date CSVs exist
        csv_dir = self._get_csv_dir(utrace_path)
        source_paths = [os.path.join(csv_dir, fname) for fname in self._SOURCE_FILES]

        csvs_exist = all(os.path.exists(p) for p in source_paths)
        # Check if any CSV is older than the master .utrace file
        csvs_outdated = csvs_exist and any(os.path.getmtime(p) < current_utrace_mtime for p in source_paths)

        if not csvs_exist or csvs_outdated:
            if not csvs_exist:
                print(f"--> CSVs not found in '{csv_dir}'.")
            if csvs_outdated:
                print(f"--> CSVs in '{csv_dir}' are outdated.")
            print(f"--> Starting .utrace to CSV conversion...")
            utrace2CSV(
                trace_file=utrace_path,
                exe_path=self.unreal_insights_exe_path,
                csv_save_dir=csv_dir
            )
        else:
            print(f"--> Found up-to-date CSV files in '{csv_dir}'.")

        # 3. Load the data from the (now existing) CSVs
        print(f"--> Loading insights from CSV data...")
        fresh_data = load_insights_data(
            data_dir=csv_dir,
            timer_type=self.timer_type,
            thread_name=self.thread_name
        )

        # 4. Update the cache with the new data and the .utrace file's modification time
        self._cache[utrace_path] = (current_utrace_mtime, fresh_data)
        print(f"--> Caching complete for '{utrace_path}'.")

        return fresh_data

    # --- Other methods like invalidate, clear, __len__ would be here ---
    def invalidate(self, utrace_path: str):
        if utrace_path in self._cache:
            del self._cache[utrace_path]
            print(f"[Cache] Manually invalidated: '{os.path.basename(utrace_path)}'.")

    def clear(self):
        self._cache.clear()
        print("[Cache] All cached data has been cleared.")


# --- DEMONSTRATION ---
if __name__ == "__main__":
    # Use a mock path for demonstration if you don't have Unreal Insights installed
    # Or provide the real path to your UnrealInsights.exe
    UI_EXE_PATH = r"C:\Program Files\Epic Games\UE_5.6\Engine\Binaries\Win64\UnrealInsights.exe"  # <-- CHANGE THIS if you have it
    # UI_EXE_PATH = r"C:\Program Files\Epic Games\UE_5.x\Engine\Binaries\Win64\UnrealInsights.exe"

    # Create a dummy .utrace file
    DUMMY_UTRACE = r"C:\Users\lyq\Desktop\Work\CodePerformanceAnalysis\data\CSV\Test\20250626_215834.utrace"

    # 1. Instantiate the manager with the required exe path
    manager = TraceDataManager(unreal_insights_exe_path=UI_EXE_PATH)

    # 2. First access: Will trigger utrace2CSV, then load_insights_data
    print("\n" + "=" * 60 + "\n1. FIRST ACCESS (EXPECT CONVERSION AND LOADING)\n" + "=" * 60)
    data_tuple = manager[DUMMY_UTRACE]
    print(f"\n--> Successfully loaded data. Received tuple with {len(data_tuple)} elements.")

    # 3. Second access: Should be a fast cache hit
    print("\n" + "=" * 60 + "\n2. SECOND ACCESS (EXPECT CACHE HIT)\n" + "=" * 60)
    cached_data_tuple = manager[DUMMY_UTRACE]

    # # 4. Invalidate due to file change
    # print("\n" + "=" * 60 + "\n3. FILE CHANGE (EXPECT RE-CONVERSION)\n" + "=" * 60)
    # time.sleep(1)  # Ensure modification timestamp is different
    # with open(DUMMY_UTRACE, "w") as f:
    #     f.write("new dummy utrace content")
    #
    # reloaded_data_tuple = manager[DUMMY_UTRACE]
    #
    # # Clean up
    # import shutil
    #
    # csv_dir_to_remove = manager._get_csv_dir(DUMMY_UTRACE)
    # if os.path.exists(csv_dir_to_remove):
    #     shutil.rmtree(csv_dir_to_remove)
    # os.remove(DUMMY_UTRACE)
    # if os.path.exists('cmd.log'):
    #     os.remove('cmd.log')
    # print("\n--> Cleaned up dummy files.")