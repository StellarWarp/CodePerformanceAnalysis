from typing import List

from anytree import PreOrderIter, findall

from pojo import CallEventNode


class OptimizedCandidateDiscovery:

    def __init__(self, root_node: CallEventNode,min_cost: float):
        self.root_node = root_node
        self.min_cost = min_cost
        self.opt_value_dict = {}

    @staticmethod
    def getConsumptionRatio(parent_node:CallEventNode,child_node:CallEventNode)->float:
        return child_node.Duration / parent_node.Duration


    def isSharedFunction(self,node:CallEventNode)->bool:
        # TODO 暂时不排除共享函数
        return False

    def _calculateOptValue(self,node:CallEventNode)->float:

        if node in self.opt_value_dict:
            return self.opt_value_dict[node]
        if self.isSharedFunction(node) or OptimizedCandidateDiscovery.getConsumptionRatio(self.root_node,node) < self.min_cost:
            return 0.0
        childTree_opt_values = []
        for child_node in PreOrderIter(node,filter_=lambda x:x is not node):
            # child_node.opt_value = self._calculateOptValue(child_node)
            opt_value = self._calculateOptValue(child_node)
            self.opt_value_dict.update({child_node:opt_value})
            childTree_opt_values.append(opt_value)


        node_opt_value = node.Duration - sum(childTree_opt_values)
        return node_opt_value

    def getKeyNodes(self)->List[CallEventNode]:
        return findall(self.root_node,filter_=lambda x:self._calculateOptValue(x) > self.root_node.Duration * self.min_cost and x is not self.root_node) # TODO 是否可以讲root_node直接从key Node中删除