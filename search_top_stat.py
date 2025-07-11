import os
import re
import argparse
from pathlib import Path

import pandas as pd


class UECSVTimingStatFinder:
    def __init__(self, project_path):
        self.project_path = Path(project_path)
        self.pattern = re.compile(
            r'CSV_SCOPED_TIMING_STAT\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',
            re.IGNORECASE | re.MULTILINE
        )

        # UE项目中常见的源代码文件扩展名
        self.file_extensions = {'.cpp', '.h', '.hpp', '.c', '.cc', '.cxx'}

        # 需要排除的目录
        self.exclude_dirs = {
            'Binaries', 'Build', 'DerivedDataCache', 'Intermediate',
            'Saved', '.git', '.vs', '.vscode', 'node_modules'
        }

    def should_process_file(self, file_path):
        """判断是否应该处理该文件"""
        # 检查文件扩展名
        if file_path.suffix.lower() not in self.file_extensions:
            return False

        # 检查是否在排除的目录中
        for part in file_path.parts:
            if part in self.exclude_dirs:
                return False

        return True

    def find_in_file(self, file_path):
        """在单个文件中查找匹配的语句"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            matches = []
            for match in self.pattern.finditer(content):
                # 计算行号
                line_number = content[:match.start()].count('\n') + 1

                # 提取参数
                category = match.group(1).strip()
                stat_name = match.group(2).strip()

                # 获取完整的匹配文本
                full_match = match.group(0)

                matches.append({
                    'file_path': file_path,
                    'line': line_number,
                    'category': category,
                    'stat_name': stat_name,
                    'full_text': full_match,
                })

            return matches

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return []

    def search_project(self):
        """
        在整个项目中搜索
        :return:Dict = {
            file_path: {
                line:
                category:
                stat_name:
                full_text:
            }
        }
        """
        results = []
        total_files = 0
        processed_files = 0

        print(f"开始搜索项目: {self.project_path}")
        print("=" * 50)

        # 遍历项目目录
        for root, dirs, files in os.walk(self.project_path):
            # 过滤排除的目录
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]

            for file in files:
                file_path = Path(root) / file
                total_files += 1

                if self.should_process_file(file_path):
                    processed_files += 1
                    matches = self.find_in_file(file_path)

                    if matches:
                        results.extend(matches)

        print(f"扫描完成! 总文件数: {total_files}, 处理文件数: {processed_files}")
        return results

def search_top_stat_and_save(save_csv_file,project_path,ue_source_path):
    print('='*5+'start search project top stat'+'='*5+'\n')
    finder = UECSVTimingStatFinder(project_path)
    project_stats = finder.search_project()
    project_top_stat_df = pd.DataFrame.from_records(project_stats)

    print('='*5+'start search UE Source top stat'+'='*5+'\n')
    finder = UECSVTimingStatFinder(ue_source_path)
    ue_source_stats = finder.search_project()
    ue_source_top_stat_df = pd.DataFrame.from_records(ue_source_stats)

    pd.concat([project_top_stat_df,ue_source_top_stat_df]).to_csv(save_csv_file, index=False)




if __name__ == '__main__':
    ue_source = r'C:\Users\lyq\UESource\UnrealEngine-5.4.4-release'
    project_path = r'C:\Users\lyq\Documents\Unreal Projects\LyraStarterGame3 5.4'
    search_top_stat_and_save('output.csv', project_path, ue_source)