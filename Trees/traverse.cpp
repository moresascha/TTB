#include "ct.h"
#include <queue>
#include "output.h"

void TraverseBreathFirst(ICTTreeNode* node, OnNodeTraverse callBack)
{
    std::queue<ICTTreeNode*> queue;

    queue.push(node);

    while(!queue.empty())
    {
        ICTTreeNode* node = queue.front();
        queue.pop();
        ct_printf("%p\n", node);
        callBack(node, NULL);
        if(!node->IsLeaf())
        {
            queue.push(node->LeftNode());
            queue.push(node->RightNode());
        }
    }
}

void TraverseDepthFirst(ICTTreeNode* node, OnNodeTraverse callBack)
{
    callBack(node, NULL);
    if(!node->IsLeaf())
    {
        TraverseDepthFirst(node->LeftNode(), callBack);
        TraverseDepthFirst(node->RightNode(), callBack);
    }
}

void TraverseVEB(ICTTreeNode* root, OnNodeTraverse callBack)
{

}