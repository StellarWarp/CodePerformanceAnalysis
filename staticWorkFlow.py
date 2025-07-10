import re
from typing import List, Dict

import anytree
import pandas as pd
from rapidfuzz.fuzz import ratio

from pojo import CallEventNode


class StaticWorkFlow:


    def __init__(self,
                 secondary_stat_timer_df: pd.DataFrame,
                 secondary_stat_call_tree: CallEventNode,
                 top_stat_df: pd.DataFrame,
                 last_version_top_stat_df: pd.DataFrame):
        """
        top stat df = {
            Event:
            min:
            max:
            avg:
            pct:
            count:
            module:
        }
        :param secondary_stat_timer_df:
        :param secondary_stat_call_tree:
        :param top_stat_df:
        :param last_version_top_stat_df:
        """
        self.secondary_stat_timer_df = secondary_stat_timer_df
        self.secondary_stat_call_tree = secondary_stat_call_tree
        self.top_stat_df = top_stat_df
        self.last_version_top_stat_df = last_version_top_stat_df

    def get_func_context_by_keyword(self, keyword):
        # TODO


    def search_secondary_stat_by_top_stat(self,keywords:List[str])->Dict[str,Dict[str,str]]:
        """
        :param keywords: 一级数据关键词
        :return: {
            一级数据keyword: {
                二级数据category:
                二级数据stat_name:
            }
        }
        """
        pattern = r"CSV_SCOPED_TIMING_STAT\(([^,]+),\s*([^)]+)\)"  # TODO 考虑注释情况？考虑其他语句类型？
        secondary_stat_dict = {}
        for keyword in keywords:
            context = self.get_func_context_by_keyword(keyword)
            if not context:
                print(f"<UNK>{keyword}<UNK>")
                continue
            match = re.search(pattern, context)  # TODO 考虑多个匹配？
            match_dict = {}
            if match:
                category = match.group(1).strip()
                stat_name = match.group(2).strip()
                match_dict = {
                    'category': category,
                    'stat_name': stat_name,
                }

            secondary_stat_dict[keyword] = match_dict

        return secondary_stat_dict

    def query_secondary_stat_by_keyword(self, keyword,k=1)->List[str]:
        # TODO 文本匹配后续实现向量匹配
        sim_ratio_list = [(c,ratio(c,keyword))  for c in self.secondary_stat_timer_df['TimerName'].to_numpy()]
        sorted_sim_ratio_list = sorted(sim_ratio_list, key=lambda x: x[1], reverse=True)[:k]
        return [w[0] for w in sorted_sim_ratio_list ]

    def is_exception_node(self,node:CallEventNode,top_stat_name:str,max_ratio_threshold=2.0)->bool:
        avg_cost = self.last_version_top_stat_df[self.last_version_top_stat_df['Event'] == top_stat_name].iloc[0]['avg']
        return node.duration * 1000.0 >= max_ratio_threshold * avg_cost



    def second_stage(self,keywords:List[str]):
        """
        :param keywords: 一级打点中获取得到的异常打点关键词列表
        :return: {
            一级异常打点关键词:{
                 category:二级数据
                 stat_name:二级数据打点名称
                 TimerName:通过在UTrace中查询的二级打点结果
                 exception_node_list: List[CallEventNode]，UTrace中筛选得到的异常调用
            }
        }
        """
        secondary_stat_dict = self.search_secondary_stat_by_top_stat(keywords)
        # TODO TOP k处理
        for top_stat_name,record in secondary_stat_dict.items():
            secondary_stat_name_list = self.query_secondary_stat_by_keyword(record['stat_name'])
            record['TimerName'] = secondary_stat_name_list[0] # TODO top K处理
            record['exception_node_list'] = anytree.findall(self.secondary_stat_call_tree,filter_=lambda node: node.TimerName == record['TimerName'] and self.is_exception_node(node,top_stat_name))

        return secondary_stat_dict



