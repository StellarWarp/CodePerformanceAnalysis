import os
import re
import argparse
from pathlib import Path


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
                param1 = match.group(1).strip()
                param2 = match.group(2).strip()

                # 获取完整的匹配文本
                full_match = match.group(0)

                matches.append({
                    'line': line_number,
                    'param1': param1,
                    'param2': param2,
                    'full_text': full_match,
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })

            return matches

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return []

    def search_project(self):
        """在整个项目中搜索"""
        results = {}
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
                        # 计算相对路径
                        rel_path = file_path.relative_to(self.project_path)
                        results[str(rel_path)] = matches

        print(f"扫描完成! 总文件数: {total_files}, 处理文件数: {processed_files}")
        return results

    def print_results(self, results):
        """打印搜索结果"""
        if not results:
            print("未找到任何 CSV_SCOPED_TIMING_STAT 语句")
            return

        total_matches = sum(len(matches) for matches in results.values())
        print(f"\n找到 {total_matches} 个匹配项，分布在 {len(results)} 个文件中:")
        print("=" * 60)

        for file_path, matches in results.items():
            print(f"\n📁 文件: {file_path}")
            print("-" * 40)

            for i, match in enumerate(matches, 1):
                print(f"  {i}. 行 {match['line']}: {match['full_text']}")
                print(f"     参数1: {match['param1']}")
                print(f"     参数2: {match['param2']}")
                print()

    def export_to_csv(self, results, output_file="csv_timing_stats.csv"):
        """导出结果到CSV文件"""
        import csv

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['文件路径', '行号', '参数1', '参数2', '完整语句']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for file_path, matches in results.items():
                for match in matches:
                    writer.writerow({
                        '文件路径': file_path,
                        '行号': match['line'],
                        '参数1': match['param1'],
                        '参数2': match['param2'],
                        '完整语句': match['full_text']
                    })

        print(f"结果已导出到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='在UE项目中查找 CSV_SCOPED_TIMING_STAT 语句')
    parser.add_argument('--project_path', help='UE项目根目录路径')
    parser.add_argument('--export', '-e', help='导出结果到CSV文件', metavar='OUTPUT_FILE')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式，只显示结果')

    args = parser.parse_args()

    # 检查项目路径是否存在
    if not os.path.exists(args.project_path):
        print(f"错误: 项目路径 '{args.project_path}' 不存在")
        return

    # 创建查找器实例
    finder = UECSVTimingStatFinder(args.project_path)

    # 执行搜索
    results = finder.search_project()

    # 显示结果
    if not args.quiet:
        finder.print_results(results)

    # 导出结果
    if args.export:
        finder.export_to_csv(results, args.export)


if __name__ == "__main__":
    # 如果直接运行脚本，可以在这里设置默认路径进行测试
    # 例如: finder = UECSVTimingStatFinder("C:/YourUEProject")
    # results = finder.search_project()
    # finder.print_results(results)

    main()