import os
import subprocess
from pathlib import Path


RSP_TEMPLATE = """TimingInsights.ExportThreads {csv_dir}/Threads.csv
TimingInsights.ExportTimers {csv_dir}/Timers.csv
TimingInsights.ExportTimingEvents {csv_dir}/TimerEvents.csv -columns="*" -threads="*" -timers="*"
TimingInsights.ExportTimerStatistics {csv_dir}/TimerStat.csv
"""



def create_temp_rsp_file(csv_dir: str):
    if os.path.isabs(csv_dir):
        csv_dir = os.path.abspath(csv_dir)
    rsp_content = RSP_TEMPLATE.format(csv_dir=csv_dir)
    os.makedirs(csv_dir, exist_ok=True)
    temp_rsp_file = os.path.join(csv_dir, 'temp.rsp')
    with open(temp_rsp_file, 'w',encoding='utf-8') as f:
        f.write(rsp_content)
    return temp_rsp_file




def run_unreal_insights_with_rsp(exe_path: str, trace_file: str, abslog_file: str, rsp_file: str):
    cmd = (
        f'"{exe_path}" '
        f'-OpenTraceFile="{trace_file}" '
        f'-ABSLOG="{abslog_file}" '
        f'-AutoQuit -NoUI '
        f'-ExecOnAnalysisCompleteCmd="@={rsp_file}"'
    )
    print("[>] 正在执行命令：")
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)
    print("[>] End")


def utrace2CSV(trace_file: str,exe_path: str, csv_save_dir:str='',abslog_file: str='cmd.log'):
    """
        将跟踪文件转换为CSV格式

        Args:
            trace_file: 输入的跟踪文件路径
            exe_path: 可执行文件路径
            csv_save_dir: CSV文件保存目录（默认为当前目录）
            abslog_file: 日志文件路径

        Returns:
            bool: 是否成功执行
        """
    # 1. 检查并创建日志文件目录
    log_path = Path(abslog_file)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"无法创建日志目录 {log_path.parent}: {e}")
        return False
    # 3. 检查输入文件是否存在
    if not Path(trace_file).exists():
        print(f"跟踪文件不存在: {trace_file}")
        return False

    if not Path(exe_path).exists():
        print(f"可执行文件不存在: {exe_path}")
        return False

    # 4. 处理输出目录
    output_dir = Path(csv_save_dir) if csv_save_dir else Path.cwd()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"输出目录已确认: {output_dir}")
    except Exception as e:
        print(f"无法创建输出目录 {output_dir}: {e}")
        return False

    temp_rsp_file = create_temp_rsp_file(csv_save_dir)
    run_unreal_insights_with_rsp(exe_path, trace_file, abslog_file, rsp_file=temp_rsp_file)




if __name__ == "__main__":
    # === 路径设定 ===
    trace_file = r"C:\Users\lyq\Desktop\Work\CodePerformanceAnalysis\data\CSV\test.utrace"
    exe_path   = r"C:\Program Files\Epic Games\UE_5.6\Engine\Binaries\Win64\UnrealInsights.exe"
    base_dir   = r"C:\Users\lyq\Desktop\Work\CodePerformanceAnalysis\data\CSV\TestCSV"

    utrace2CSV(trace_file, exe_path, base_dir)
